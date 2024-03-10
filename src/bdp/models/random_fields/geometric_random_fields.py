import os
import numpy as np
import torch
from torch.distributions import Uniform
from deep_fields import project_path
from deep_fields.models.abstract_models import DeepBayesianModel

from matplotlib import pyplot as plt
from gpytorch.kernels import RBFKernel, ScaleKernel
from deep_fields.models.gaussian_processes.gaussian_processes import multivariate_normal, white_noise_kernel
from deep_fields.models.random_measures.old_stuff.point_measures import uniform_point
from torch.distributions import Gamma

class geometric_kernel_field(DeepBayesianModel):

    def __init__(self, model_dir=None, data_loader=None, **kwargs):
        DeepBayesianModel.__init__(self, "geometric_kernel_field", model_dir=model_dir, data_loader=data_loader, **kwargs)
        self.define_deep_models()

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        number_of_assets = 100
        parameters_sample = {"number_of_assets": number_of_assets,  # z
                             "interest_rate_alpha": 1.,
                             "interest_rate_beta": 1.,
                             "delta_t": 0.001,
                             "max_time": 1.,
                             "kernel_sigma": 1.,
                             "kernel_lenght_scales": [0.2],
                             "z_dim": 1,
                             "model_path": os.path.join(project_path, 'results')}

        return parameters_sample

    def set_parameters(self, **kwargs):
        self.number_of_assets = kwargs.get("number_of_assets")
        self.interest_rate_alpha = kwargs.get("interest_rate_alpha")
        self.interest_rate_beta = kwargs.get("interest_rate_beta")
        self.delta_t = kwargs.get("delta_t")
        self.max_time = kwargs.get("max_time")
        self.kernel_sigma = kwargs.get("kernel_sigma")
        self.kernel_lenght_scales = kwargs.get("kernel_lenght_scales")
        self.z_dim = kwargs.get("z_dim")

    def update_parameters(self, data_loader, **kwargs):
        self.data_loader = data_loader
        return kwargs

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        inference_parameters.update({"regularizers": {"nll": None,
                                                      "kl_eta": {"lambda_0": 1., "percentage": .2},
                                                      "kl_alpha": {"lambda_0": 1., "percentage": .2},

                                                      "kl_eta_0": {"lambda_0": 1., "percentage": .2},
                                                      "kl_alpha_0": {"lambda_0": 1., "percentage": .2},
                                                      "kl_theta": {"lambda_0": 1., "percentage": .2}}})
        inference_parameters.update({"model_eval": "perplexity_blei"})
        inference_parameters.update({"gumbel": .0005})

        return inference_parameters

    def initialize_inference(self, data_loader, **inference_parameters):
        super().initialize_inference(data_loader=data_loader, **inference_parameters)

    def define_deep_models(self):
        self.f_kernel = ScaleKernel(RBFKernel(ard_num_dims=self.z_dim, requires_grad=True),
                                    requires_grad=True) + white_noise_kernel()

        self.f_hypers = {"raw_outputscale": torch.tensor(self.kernel_sigma),
                         "base_kernel.raw_lengthscale": torch.tensor(self.kernel_lenght_scales)}

        self.f_kernel.kernels[0].initialize(**self.f_hypers)

    def sample(self):
        self.define_deep_models()
        z_prior = Uniform(0., 1.)
        z = z_prior.sample(torch.Size([self.number_of_assets, self.z_dim]))
        f_variance = self.f_kernel(z, z).evaluate().float() * self.delta_t

        f_mean = torch.zeros((1, self.number_of_assets))  # [number_of_batches,mean_dimension]
        f_distribution = multivariate_normal(f_mean, f_variance)
        f = f_distribution.rsample().float()

        log_assets = [f]
        for i in np.arange(0., self.max_time, self.delta_t):
            f_distribution = multivariate_normal(log_assets[-1], f_variance)
            f = f_distribution.rsample().float()
            log_assets.append(f)
        log_assets = torch.cat(log_assets, dim=0)
        return log_assets

class poisson_gaussian_portfolio_process(DeepBayesianModel):

    def __init__(self, model_dir=None, data_loader=None, **kwargs):
        DeepBayesianModel.__init__(self, "poisson_gaussian_process", model_dir=model_dir, data_loader=data_loader, **kwargs)
        self.define_deep_models()

    @classmethod
    def get_parameters(cls):
        """
        here we provide an example of the minimum set of parameters requiered to instantiate the model
        """
        parameters_sample = {"birth_rate": 1.,  # z
                             "death_rate":0.,
                             "interest_rate_mean":1.,
                             "allocation_alpha": 1.,
                             "allocation_beta": 1.,
                             "time_support": 10.,
                             "z_dim": 1,
                             "z_support": 1.,
                             "kernel_sigma": 1.,
                             "kernel_lenght_scales": [0.001],
                             "model_path": os.path.join(project_path, 'results')}

        return parameters_sample

    def set_parameters(self, **kwargs):
        self.birth_rate = kwargs.get("birth_rate")
        self.death_rate = kwargs.get("death_rate")
        self.interest_rate_mean = kwargs.get("interest_rate_mean")

        self.time_support = kwargs.get("time_support")
        self.z_support = kwargs.get("z_support")
        self.z_dim = kwargs.get("z_dim")

        self.kernel_sigma = kwargs.get("kernel_sigma")
        self.kernel_lenght_scales = kwargs.get("kernel_lenght_scales")

        self.allocation_alpha = kwargs.get("allocation_alpha")
        self.allocation_beta = kwargs.get("allocation_beta")

    def update_parameters(self, data_loader, **kwargs):
        self.data_loader = data_loader
        return kwargs

    def define_deep_models(self):
        self.control_distribution = Gamma(self.allocation_alpha, self.allocation_beta)

        # covariance kernel
        self.kernel = ScaleKernel(RBFKernel(ard_num_dims=self.z_dim, requires_grad=True),
                                  requires_grad=True) + white_noise_kernel()

        self.kernel_hypers = {"raw_outputscale": torch.tensor(self.kernel_sigma),
                              "base_kernel.raw_lengthscale": torch.tensor(self.kernel_lenght_scales)}

        self.kernel.kernels[0].initialize(**self.kernel_hypers)

    def sample(self):
        points = uniform_point(self.birth_rate,
                               self.time_support,
                               self.z_support)

        number_of_assets = points.shape[1]
        allocation_ = self.control_distribution.sample(sample_shape=([number_of_assets])).squeeze()
        z = points[1:, :].T
        assets_covariance = self.kernel(z, z).evaluate().float()
        assets_mean = torch.ones((1, number_of_assets))
        assets_distribution = multivariate_normal(assets_mean, assets_covariance)
        return allocation_,assets_distribution


if __name__ == "__main__":
    model = "poisson_gaussian"
    #model = "geometric_field"

    if model == "geometric_field":
        model_param = geometric_kernel_field.get_parameters()
        GKF = geometric_kernel_field(**model_param)
        log_assets = GKF.sample()

        assets_price = torch.exp(log_assets).detach().numpy()

        plt.plot(assets_price[:, 0])
        plt.show()
    else:
        model_param = poisson_gaussian_portfolio_process.get_parameters()
        pgp = poisson_gaussian_portfolio_process(**model_param)
        allocation_,assets_distribution = pgp.sample()


