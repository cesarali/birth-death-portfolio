import json
import os
from datetime import datetime

import numpy as np
from deep_fields import data_path
from deep_fields import project_path
from deep_fields.data.gaussian_processes.datasets import PossionGPDataset
from deep_fields.models import utils
from deep_fields.models.abstract_models import DeepBayesianModel
from deep_fields.models.basic_utils import all_metrics_to_floats

from deep_fields.models.gaussian_processes.gp_utils import polya_gamma_mean

from matplotlib import pyplot as plt
from torch.distributions import Gamma, Poisson
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Adam
from tqdm import tqdm

from gpytorch.kernels import RBFKernel, ScaleKernel
from deep_fields.models.gaussian_processes.gaussian_processes import calculate_posterior
from deep_fields.models.gaussian_processes.gaussian_processes import multivariate_normal, white_noise_kernel

class PoissonGP(DeepBayesianModel):
    """
    here we follow (overleaf link)

    https://www.overleaf.com/read/dcggwrccpttq

    (Synthetic Data)
    """

    def __init__(self, model_dir=None, data_loader=None, **kwargs):
        model_name = "poisson_gp"
        DeepBayesianModel.__init__(self, model_name, model_dir=model_dir, data_loader=data_loader, **kwargs)

    def set_parameters(self, **kwargs):
        self.k = kwargs.get("k")
        self.theta = kwargs.get("theta")
        self.kernel_sigma = kwargs.get("kernel_sigma")
        self.kernel_l = kwargs.get("kernel_l")

        self.train_steps = kwargs.get("number_of_train_steps")
        self.number_all_steps = kwargs.get("number_all_steps")
        self.val_steps = kwargs.get("number_of_val_steps")
        self.train_support = kwargs.get("train_support")
        self.val_support = kwargs.get("val_support")
        self.number_of_processes = kwargs.get("number_of_processes")

        self.data_size = torch.Size((self.number_of_processes, self.train_steps))

    def update_parameters(self, data_set, **kwargs):
        kwargs.update({"number_all_steps": data_set.number_all_steps})
        kwargs.update({"number_of_processes": data_set.number_of_processes})
        kwargs.update({"number_of_train_steps": data_set.number_of_train_steps})
        kwargs.update({"number_of_val_steps": data_set.number_of_val_steps})

        self.train_input = data_set.train_input
        self.val_input = data_set.val_input

        self.train_support = data_set.train_support
        self.val_support = data_set.val_support

        return kwargs

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        parameters_sample = {"k": 5.,
                             "theta": 1.,
                             "kernel_sigma": 1.,
                             "kernel_l": 1.,
                             "train_support": 8.,
                             "val_support": 10.,
                             "number_of_processes": 9,
                             "number_all_steps": 100,
                             "number_of_train_steps": 80,
                             "number_of_val_steps": 20,
                             "model_path": os.path.join(project_path, 'results')}

        return parameters_sample

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"train_gp": True})
        inference_parameters.update({"train_lamba0": True})
        inference_parameters.update({"train_omega": True})
        inference_parameters.update({"train_big_omega": True})
        inference_parameters.update({"train_rho": True})

        inference_parameters.update({"model_eval": "loss"})
        return inference_parameters

    def define_deep_models(self):
        windows_size = 1
        self.f_kernel = ScaleKernel(RBFKernel(ard_num_dims=windows_size, requires_grad=True),
                                    requires_grad=True) + white_noise_kernel()

        f_hypers = {"raw_outputscale": torch.tensor(self.kernel_sigma),
                    "base_kernel.raw_lengthscale": torch.tensor(np.repeat(self.kernel_l, windows_size))}

        self.f_kernel.kernels[0].initialize(**f_hypers)

    def sample(self, data_dir=data_path, number_of_processes=10):
        """
        returns
        ------
        (documents,documents_z,thetas,phis)
        """
        self.define_deep_models()
        self.train_input = torch.tensor(np.linspace(0., self.val_support, self.number_all_steps))
        gamma = Gamma(torch.Tensor([self.theta]), torch.Tensor([self.k]))
        lambda0 = gamma.sample()

        f_variance = self.f_kernel(self.train_input, self.train_input).evaluate().float()
        f_mean = torch.zeros((1, self.train_steps))  # [number_of_batches,mean_dimension]
        f_distribution = multivariate_normal(f_mean, f_variance)
        f = f_distribution.rsample(torch.Size((self.number_of_processes,))).squeeze().float()

        LAMBDA = lambda0 * torch.sigmoid(f)
        poisson = Poisson(LAMBDA)
        n = poisson.sample()

        sampled_values = {"n": n.detach().numpy().tolist(),
                          "f": f.detach().numpy().tolist(),
                          "lambda0": lambda0.item(),
                          "theta": self.theta,
                          "k": self.k,
                          "kernel_sigma": self.kernel_sigma,
                          "kernel_l": self.kernel_l,
                          "all_support": self.time_support,
                          "all_steps": self.train_steps}

        path_name = os.path.join(data_dir, "poisson_gp")
        if not os.path.isdir(path_name):
            os.makedirs(path_name, exist_ok=True)

        path_name = os.path.join(path_name, "poisson_gp.json")
        json.dump(sampled_values, open(path_name, "w"))

        return n

    def metrics(self, batch_data, forward_results, epoch, mode="evaluation", data_loader=None):
        if mode == "train" or mode == "validation":
            with torch.no_grad():
                log_likelihood = 0.
                return {"best_log_likelihood": log_likelihood}
        if mode == "validation_global" and epoch % self.metrics_logs == 0:
            return {}
        return {}

    def loss(self, databatch, forward_results, data_loader, epoch):
        """
        nll [batch_size, max_lenght]
        """
        EQ_L = self.average_likelihood(databatch)
        KL_g = self.gp_kl()
        KL_lambda0 = self.lambda0_kl()

        elbo = EQ_L + KL_g + KL_lambda0

        return {"loss": -elbo, "elbo": elbo}

    def data_to_device(self, databatch):
        databatch = databatch.to(self.device)
        return databatch

    def lambda0_averages(self, databatch):
        n = databatch
        rho_average = self.mean_field_values['rho']

        n_sum = n.sum()
        rho_average_sum = rho_average.sum()

        new_theta = PGP.theta + n_sum
        new_alpha = PGP.k + n_sum + rho_average_sum

        self.mean_field_values["lambda0"] = new_alpha / new_theta
        self.mean_field_values["log_lambda0"] = torch.digamma(new_alpha) - torch.log(new_theta)

        self.mean_field_values["alpha_lambda"] = new_alpha
        self.mean_field_values["beta_lambda"] = new_theta

    def rho_averages(self, databatch):
        log_lambda0 = self.mean_field_values["log_lambda0"]
        gp_mean = self.mean_field_values['gp_mean']
        gp_mean_square = self.mean_field_values['gp_mean_square']
        A = torch.exp(-.5 * gp_mean)
        B = 2 * torch.cosh(torch.sqrt(gp_mean_square) * .5)
        rho_average = torch.exp(log_lambda0) * (A / B)

        self.mean_field_values["rho"] = rho_average

    def big_omega_averages(self, databatch):
        gp_mean_square = self.mean_field_values['gp_mean_square']
        rho_average = self.mean_field_values['rho']

        big_omega = (1 / (torch.sqrt(gp_mean_square) * 2)) * torch.tanh(torch.sqrt(gp_mean_square) * .5)
        big_omega = rho_average * big_omega

        self.mean_field_values["big_omega"] = big_omega

    def omega_averages(self, databatch):
        batch_size = databatch.shape[0]
        gp_mean_square = self.mean_field_values['gp_mean_square']
        root_gp_mean_square = torch.sqrt(gp_mean_square).repeat(batch_size, 1)
        omega = polya_gamma_mean(databatch, root_gp_mean_square)

        self.mean_field_values["omega"] = omega

    def gp_averages(self, databatch):
        n = databatch
        rho_average = self.mean_field_values['rho']
        omega = self.mean_field_values["omega"]
        big_omega = self.mean_field_values["big_omega"]

        mat = big_omega + omega
        Lambda = torch.zeros((self.number_of_processes, self.train_steps, self.train_steps))
        Lambda.as_strided(mat.size(), [Lambda.stride(0), Lambda.size(2) + 1]).copy_(mat)
        a = -rho_average + n
        K = PGP.f_kernel(self.train_input, self.train_input).evaluate().float()

        f_mean = torch.zeros((1, self.train_steps))  # [number_of_batches,mean_dimension]
        f_distribution = multivariate_normal(f_mean, K)
        Sigma0_inv = f_distribution.varinv()
        Sigma_inv = Sigma0_inv + Lambda
        Sigma = torch.inverse(Sigma_inv)
        gp_mean = torch.matmul(Sigma_inv, a.unsqueeze(-1)).squeeze()
        sigma_square = torch.diagonal(Sigma, offset=0, dim1=-2, dim2=-1)
        gp_mean_square = gp_mean ** 2 + sigma_square

        Sigma0 = K.unsqueeze(0).repeat(PGP.number_of_processes, 1, 1)

        self.mean_field_values["Sigma"] = Sigma
        self.mean_field_values["Sigma0"] = Sigma0
        self.mean_field_values['gp_mean'] = gp_mean
        self.mean_field_values['gp_mean_square'] = gp_mean_square

    def gp_kl(self):
        gp_mean = self.mean_field_values['gp_mean']
        Sigma = self.mean_field_values["Sigma"]
        Sigma0 = self.mean_field_values["Sigma0"]

        gp_posterior = Normal(gp_mean.unsqueeze(-1), Sigma)
        gp_prior = Normal(torch.zeros_like(gp_mean.unsqueeze(-1)), Sigma0)

        return kl_divergence(gp_posterior, gp_prior).sum(dim=2).sum(dim=1).par_1()

    def lambda0_kl(self):
        if self.train_lamba0:
            alpha0 = self.k
            beta0 = self.theta

            LOG_LAMBDA_0_AVERAGE = self.mean_field_values["log_lambda0"]
            LAMBDA_0_AVERAGE = self.mean_field_values["lambda0"]
            alpha_lambda = self.mean_field_values["alpha_lambda"]
            beta_lambda = self.mean_field_values["beta_lambda"]

            a = alpha0 * np.log(beta0)
            # b = -np.log(gamma_function(self.alpha_0))
            b = -alpha0 * np.log(alpha0) + alpha0
            c = (alpha0 - 1.) * LOG_LAMBDA_0_AVERAGE
            d = -(beta0 * LAMBDA_0_AVERAGE)
            e = alpha_lambda - np.log(beta_lambda)

            f = alpha_lambda * np.log(alpha_lambda) - alpha_lambda
            g = (1. - alpha_lambda) * torch.digamma(alpha_lambda)
            LAMBDA_0_BOUND = a + b + c + d + e + f + g
            return LAMBDA_0_BOUND
        else:
            return 0.

    def average_likelihood(self, databatch):
        lambda0 = self.mean_field_values["lambda0"]
        log_lambda0 = self.mean_field_values["log_lambda0"]
        gp_mean = self.mean_field_values['gp_mean']
        gp_mean_square = self.mean_field_values['gp_mean_square']

        A = torch.exp(-.5 * gp_mean)
        B = 2 * torch.cosh(torch.sqrt(gp_mean_square) * .5)

        Log_Z_Wrho = torch.exp(log_lambda0) * ((A / B) - 1.)
        log_Z_w = -databatch * torch.log(torch.cosh(torch.sqrt(gp_mean_square) * .5))
        C = (databatch.sum() + 1) * log_lambda0 - lambda0

        EQ_L = Log_Z_Wrho + log_Z_w + C
        return EQ_L.sum(dim=1).par_1()

    def mean_field_loop(self, databatch):
        """
        parameters
        ----------
        data ()
        returns
        -------
        z (batch_size*sequence_lenght,self.hidden_state_dim)
        recognition_parameters = (z_mean,z_var)
        (batch_size*sequence_lenght,self.hidden_state_dim)
        likelihood_parameters = (likelihood_mean,likelihood_variance)
        """
        if self.train_gp:
            self.gp_averages(databatch)

        if self.train_lamba0:
            self.lambda0_averages(databatch)

        if self.train_omega:
            self.omega_averages(databatch)

        if self.train_big_omega:
            self.big_omega_averages(databatch)

        if self.train_rho:
            self.train_rho(databatch)

    def inference_step(self, optimizer, data_set, epoch, inference_variables, **inference_parameters):
        self.train()

        data = data_set.get_count()
        optimizer.zero_grad()

        self.initialize_steps(data)
        data = self.data_to_device(data)

        self.mean_field_loop(data)
        forward_results = self(data)
        losses = self.loss(data, forward_results, data_set, epoch)
        metrics = self.metrics(data, forward_results, epoch, mode="train")

        loss = losses["loss"]
        loss.backward()

        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_norm)

        parameters_and_grad_stats = {}
        if self.debug:
            self.check_gradients_and_params()
            parameters_and_grad_stats = self.check_gradients_and_params_stats()
        optimizer.step()
        self.detach_history()

        self.update_writer({**losses, **metrics, **parameters_and_grad_stats}, label="train")
        self.number_of_iterations += 1

    def inference(self, data_loader, **inference_parameters):
        if self.INFERENCE:
            self.generate_training_message()
            inference_variables = self.initialize_inference(data_loader, **inference_parameters)
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=1.2 * 1e-6)

            for epoch in tqdm(range(1, self.number_of_epochs + 1), desc="Train Epoch", unit='epoch'):
                self.inference_step(optimizer, data_loader, epoch, inference_variables, **inference_parameters)
                # model_evaluation, all_metrics = self.validation_step(optimizer, data_loader, epoch, inference_variables, **inference_parameters)
                all_metrics = all_metrics_to_floats(all_metrics)
                # if model_evaluation < self.best_eval:
                #    self.best_eval = model_evaluation
                #    self.time_f = datetime.now()
                #    self.inference_results["best_eval_time"] = (self.time_f - self.time_0).total_seconds()
                #    self.inference_results["best_eval_criteria"] = self.best_eval
                #    json.dump(self.inference_results, open(self.inference_path, "w"))
                #    self.save_model()

                if self.debug and (epoch > self.reduced_num_batches):
                    break

            final_time = datetime.now()
            self.inference_results["final_time"] = (final_time - self.time_0).total_seconds()
            self.inference_results.update(all_metrics)
            json.dump(self.inference_results, open(self.inference_path, "w"))
            return self.inference_results
        else:
            print("MODEL OPEN IN RESULTS FOLDER, INFERENCE WILL OVERRIDE OLD RESULTS")
            print("CREATE NEW MODEL")
            raise Exception

    def initialize_inference(self, poisson_gp_dataset, **inference_parameters):
        super().initialize_inference(data_loader=poisson_gp_dataset, **inference_parameters)
        self.define_deep_models()

        self.train_gp = inference_parameters.get("train_gp")
        self.train_lamba0 = inference_parameters.get("train_lamba0")
        self.train_omega = inference_parameters.get("train_omega")
        self.train_big_omega = inference_parameters.get("train_big_omega")
        self.train_rho = inference_parameters.get("train_rho")
        self.train_kernel = inference_parameters.get("train_kernel")

        # ============================================
        # Initialize Mean Field
        # ============================================
        if not self.train_lamba0 and poisson_gp_dataset.get_lambda0() is not None:
            lambda0 = poisson_gp_dataset.get_lambda0()
        else:
            gamma = Gamma(torch.Tensor([self.theta]), torch.Tensor([self.k]))
            lambda0 = gamma.sample()

        if not self.train_gp and poisson_gp_dataset.get_gp() is not None:
            mean = poisson_gp_dataset.get_gp()
        else:
            mean = Normal(torch.zeros(self.data_size), torch.ones(self.data_size)).sample()

        if not self.train_kernel and poisson_gp_dataset.get_gp_hyperparameters()[0] is not None:
            windows_size = 1
            self.kernel_l, self.kernel_sigma = poisson_gp_dataset.get_gp_hyperparameters()

            f_hypers = {"raw_outputscale": torch.tensor(self.kernel_sigma),
                        "base_kernel.raw_lengthscale": torch.tensor(np.repeat(self.kernel_l, windows_size))}

            self.f_kernel.kernels[0].initialize(**f_hypers)
            K = self.f_kernel(self.train_input, self.train_input)
        else:
            K = self.f_kernel(self.train_input, self.train_input)

        rho = Poisson(lambda0).sample(self.data_size)

        self.mean_field_values = {"gp_mean": mean,
                                  "gp_mean_square": mean ** 2,
                                  "K": K,
                                  "lambda0": lambda0,
                                  "log_lambda0": torch.log(lambda0),
                                  "rho": rho,
                                  "omega": rho,
                                  "big_omega": rho}

        # SET CUDA
        utils.set_cuda(self, **inference_parameters)
        self.to(self.device)
        json.dump(inference_parameters, open(self.inference_parameters_path, "w"))

    def plot_gps(self, poisson_gp_dataset):
        train_input = poisson_gp_dataset.train_input
        gp = poisson_gp_dataset.get_gp()

        if gp is not None:
            gp_sample = gp[0]
            test_input = torch.tensor(np.linspace(0., poisson_gp_dataset.train_support, 100))
            predictive_mean, predictive_variance = calculate_posterior(test_input, gp_sample, train_input, self.kernel)
            upper = predictive_mean + predictive_variance
            lower = predictive_mean - predictive_variance
            # PLOT
            f, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(train_input, gp_sample.detach().numpy().T, "o")
            ax.plot(test_input, predictive_mean.detach().numpy(), "r-")
            ax.fill_between(test_input.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
            plt.show()


if __name__ == "__main__":
    import torch
    from deep_fields import models_path

    ss_param = PoissonGP.get_parameters()
    ss_param.update({"model_path": models_path})
    ss_param.update({"k": 0.01})

    ss_inference_param = PoissonGP.get_inference_parameters()
    ss_inference_param.update({"learning_rate": .001})

    from deep_fields import data_path

    path_name = os.path.join(data_path, "poisson_gp", "poisson_gp.json")

    pgp_dataset = PossionGPDataset(path_name)
    n = pgp_dataset.get_count()

    # MODEL
    PGP = PoissonGP(data_loader=pgp_dataset, **ss_param)
    PGP.initialize_inference(pgp_dataset, **ss_inference_param)
    sample = PGP.sample()
    # print(sample.shape)
