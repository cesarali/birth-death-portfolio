import json
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from pprint import pprint
from sklearn import metrics
from datetime import datetime
from gpytorch.kernels import RBFKernel, ScaleKernel
from torch.distributions import Normal, MultivariateNormal
from torch.distributions import Bernoulli, Beta, Poisson
from torch import einsum

from scipy.stats import wishart as Wishart
from bdp.utils.debugging import timeit
from scipy.stats import invwishart as Invwishart
from bdp.models.random_fields.utils import new_kernel
from bdp.models.abstract_models import DeepBayesianModel
from bdp.models.gaussian_processes.gaussian_processes import multivariate_normal, white_noise_kernel

from torch import matmul as m
from bdp import project_path
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import invwishart, bernoulli

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

EPSILON = 1e-5

from bovy_mcmc.elliptical_slice import elliptical_slice

class PoissonCovariance(DeepBayesianModel):
    """
    # MCMC type of algorithm for a simple model of covariance matrix defined by latent arrivals
    """
    def __init__(self, model_dir=None, data_loader = None, model_name=None, **kwargs):
        if model_name is None:
            model_name = "poisson_covariance"

        DeepBayesianModel.__init__(self,
                                   model_name,
                                   model_dir=model_dir,
                                   data_loader=data_loader,
                                   **kwargs)

    def define_deep_models(self):
        inference_parameters = {}

        inference_parameters.update({"prior_locations_mean": 0.})
        inference_parameters.update({"prior_locations_std": 1.})

        inference_parameters.update({"prior_sigma_mean": 0.})
        inference_parameters.update({"prior_sigma_std": 1.})

        inference_parameters.update({"prior_length_mean": 0.})
        inference_parameters.update({"prior_length_std": 1.})

        # LOCATIONS PRIORS
        prior_locations_mean = inference_parameters.get("prior_locations_mean")
        prior_locations_std = inference_parameters.get("prior_locations_std")

        self.locations_prior = Normal(torch.full((self.locations_dimension,), prior_locations_mean),
                                      torch.full((self.locations_dimension,), prior_locations_std))

        # KERNEL PRIORS
        prior_sigma_mean = inference_parameters.get("prior_sigma_mean")
        prior_sigma_std = inference_parameters.get("prior_sigma_std")

        prior_length_mean = inference_parameters.get("prior_length_mean")
        prior_length_std = inference_parameters.get("prior_length_std")

        self.sigma_prior = Normal(torch.full((1,), prior_sigma_mean),
                                  torch.full((1,), prior_sigma_std))

        self.lenght_prior = Normal(torch.full((self.locations_dimension,), prior_length_mean),
                                   torch.full((self.locations_dimension,), prior_length_std))

    def define_kernel(self, kernel_sigma, kernel_lenght_scales):
        kernel = ScaleKernel(RBFKernel(ard_num_dims=self.locations_dimension, requires_grad=True),
                             requires_grad=True) + white_noise_kernel()

        kernel_hypers = {"raw_outputscale": torch.tensor(kernel_sigma),
                         "base_kernel.raw_lengthscale": torch.tensor(kernel_lenght_scales)}

        kernel.kernels[0].initialize(**kernel_hypers)
        kernel_eval = lambda locations: kernel(locations, locations).evaluate().float()
        return kernel, kernel_eval

    def sample(self):
        locations = self.locations_prior.sample(sample_shape=(self.number_of_arrivals,))
        kernel, kernel_eval = self.define_kernel(self.kernel_sigma, self.kernel_lenght_scales)
        K = kernel_eval(locations)
        realizations = MultivariateNormal(torch.zeros(K.shape[0]), K).sample(
            sample_shape=(self.number_of_realizations,))

        data_loader = {"realizations": realizations, "locations": locations,
                       "kernel_sigma": self.kernel_sigma, "kernel_lenght_scales": self.kernel_lenght_scales,
                       "K": K, "kernel": kernel}
        return data_loader

    @classmethod
    def get_parameters(cls):
        locations_dimension = 2
        kernel_parameters = {"kernel_sigma": 0.5,
                             "kernel_lenght_scales": [1., 2.]}
        number_of_arrivals = 4
        number_of_realizations = 500

        kwargs = {"locations_dimension": locations_dimension,
                  "kernel_parameters": kernel_parameters,
                  "number_of_arrivals": number_of_arrivals,
                  "number_of_realizations": number_of_realizations,
                  "model_path": os.path.join(project_path, 'results')}

        return kwargs

    def set_parameters(self, **kwargs):
        self.locations_dimension = kwargs.get("locations_dimension")
        self.kernel_parameters = kwargs.get("kernel_parameters")
        self.kernel_sigma = self.kernel_parameters.get("kernel_sigma")
        self.kernel_lenght_scales = self.kernel_parameters.get("kernel_lenght_scales")

        self.number_of_arrivals = kwargs.get("number_of_arrivals")
        self.number_of_realizations = kwargs.get("number_of_realizations")

    def update_parameters(self, dataloader, **kwargs):
        realizations = dataloader.get("realizations")
        number_of_realizations = realizations.shape[0]
        locations = dataloader.get("locations")
        if locations is not None:
            locations_dimension = locations.shape[1]
            kwargs.update({"number_of_arrivals": locations.shape[0]})
            kwargs.update({"locations_dimension": locations_dimension})
        kwargs.update({"number_of_realizations": number_of_realizations})
        return kwargs

    def gibbs_sigma(self, monte_carlo_values):
        return monte_carlo_values

    def gibbs_lenght(self, monte_carlo_values):
        return monte_carlo_values

    def gibbs_locations(self, data_loader, location_index, monte_carlo_values):
        realizations = data_loader.get("realizations")
        locations_sample_now = monte_carlo_values["locations"]
        K = monte_carlo_values["K"]
        kernel = monte_carlo_values["kernel"]

        locations = self.locations_sample_to_tensor(locations_sample_now)
        location_now = locations_sample_now[location_index][-1]
        location_proposal = self.locations_prior.sample().detach().numpy()

        new_location, ll = elliptical_slice(initial_theta=location_now,
                                            prior=location_proposal,
                                            lnpdf=self.log_likelihood_realizations,
                                            pdf_params=(locations, location_index, realizations, monte_carlo_values))

        K_new = new_kernel(locations, location_index, torch.Tensor(new_location).unsqueeze(0), self.number_of_arrivals,
                           K, kernel)
        monte_carlo_values["locations"][location_index].append(new_location)
        monte_carlo_values["K"] = K_new

        return monte_carlo_values

    def log_likelihood_realizations(self, new_location, locations, location_index, realizations, monte_carlo_values):
        K = monte_carlo_values["K"]
        kernel = monte_carlo_values["kernel"]
        new_location = torch.Tensor(new_location).unsqueeze(0)

        K_new = new_kernel(locations, location_index, new_location, self.number_of_arrivals, K, kernel)
        realizations_pdf_new = MultivariateNormal(torch.zeros(torch.Tensor(K_new).shape[0]), torch.Tensor(K_new))
        return realizations_pdf_new.log_prob(realizations).sum().item()

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = {}
        inference_parameters["nmc"] = 1000
        inference_parameters["train_sigma"] = False
        inference_parameters["train_lenght"] = False
        inference_parameters["train_locations"] = True
        return inference_parameters

    def initialize_inference(self, data_loader=None, parameters=None, **inference_parameters):
        self.nmc = inference_parameters.get("nmc")
        self.train_sigma = inference_parameters.get("train_sigma")
        self.train_lenght = inference_parameters.get("train_lenght")
        self.train_locations = inference_parameters.get("train_locations")

        locations_0 = [[self.locations_prior.sample().detach().numpy()] for i in range(self.number_of_arrivals)]
        kernel_sigma_0 = self.sigma_prior.sample().detach().numpy()
        kernel_lenghts_0 = self.lenght_prior.sample().detach().numpy()
        kernel, kernel_eval = self.define_kernel(kernel_sigma_0, kernel_lenghts_0)

        locations = self.locations_sample_to_tensor(locations_0)
        K = kernel_eval(torch.Tensor(locations))

        monte_carlo_parameters = {"K": K.detach().numpy(),
                                  "kernel": kernel,
                                  "kernel_sigma": [kernel_sigma_0],
                                  "kernel_lenght_scales": [kernel_lenghts_0],
                                  "locations": locations_0}

        return monte_carlo_parameters

    def locations_sample_to_tensor(self, locations_mcmc):
        locations_now = []
        for locations_series in locations_mcmc:
            locations_now.append(torch.tensor(locations_series[-1]).unsqueeze(0))
        return torch.cat(locations_now, dim=0)

    def inference(self, data_loader, **inference_parameters):
        monte_carlo_parameters = self.initialize_inference(data_loader, None, **inference_parameters)
        print("#      ---------------- ")
        print("#      Start of MCMC    ")
        print("#      ---------------- ")
        for montecarlo_index in tqdm(range(self.nmc)):
            print("Monte Carlo {0}".format(montecarlo_index))
            if self.train_sigma:
                monte_carlo_parameters = self.gibbs_sigma(monte_carlo_parameters)
            if self.train_lenght:
                monte_carlo_parameters = self.gibbs_lenght(monte_carlo_parameters)
            if self.train_locations:
                for location_index in range(self.number_of_arrivals):
                    monte_carlo_parameters = self.gibbs_locations(data_loader, location_index, monte_carlo_parameters)

        return monte_carlo_parameters
    
class MertonJumpsPoissonCovariance(DeepBayesianModel):
    """
    # MCMC type of algorithm for a simple model of covariance matrix defined by latent arrivals
    """
    def __init__(self, model_dir=None, data_loader = None, model_name=None, **kwargs):
        if model_name is None:
            model_name = "merton_jumps_poisson_covariance"

        DeepBayesianModel.__init__(self,
                                   model_name,
                                   model_dir=model_dir,
                                   data_loader=data_loader,
                                   **kwargs)

    def define_deep_models(self):
        inference_parameters = {}

        inference_parameters.update({"prior_locations_mean": 0.})
        inference_parameters.update({"prior_locations_std": 1.})

        inference_parameters.update({"prior_sigma_mean": 0.})
        inference_parameters.update({"prior_sigma_std": 1.})

        inference_parameters.update({"prior_length_mean": 0.})
        inference_parameters.update({"prior_length_std": 1.})

        # JUMPS
        self.arrivals_intensity_prior = Beta(self.jump_arrival_alpha, self.jump_arrival_beta)

        nu = self.number_of_processes + 1.
        Psi = np.random.rand(self.number_of_processes, self.number_of_processes)
        Psi = np.dot(Psi, Psi.transpose())
        self.covariance_jump_prior = invwishart(nu,Psi)

        #COVARIANCE PRIOR
        nu = self.number_of_processes + 1.
        Psi = np.random.rand(self.number_of_processes, self.number_of_processes)
        Psi = np.dot(Psi, Psi.transpose())
        self.covariance_diffusion_prior = invwishart(nu,Psi)

    def sample(self):
        # Jumps -------------------------

        # sample from intensity prior
        arrivals_intensity = self.arrivals_intensity_prior.sample(sample_shape=(self.number_of_processes,))
        self.arrivals_indicator_prior = Bernoulli(arrivals_intensity)
        arrivals_indicator = self.arrivals_indicator_prior.sample(sample_shape=(self.number_of_realizations,))

        # sample from jump size prior
        jump_covariance = self.covariance_jump_prior.rvs()
        jump_covariance =  torch.Tensor(jump_covariance)
        jump_mean_prior = torch.ones(self.number_of_processes) * self.jump_size_a
        self.jump_mean_prior = MultivariateNormal(jump_mean_prior, self.jump_size_b *jump_covariance)
        jump_mean = self.jump_mean_prior.sample()

        self.jump_distribution = MultivariateNormal(jump_mean, jump_covariance)
        jumps_size = self.jump_distribution.sample(sample_shape=(self.number_of_realizations,))

        # covariance -------------------
        diffusion_covariance = self.diffusion_covariance_normalization*self.covariance_diffusion_prior.rvs()

        diffusion_covariance = torch.Tensor(diffusion_covariance)
        diffusion_mean_prior = torch.ones(self.number_of_processes) * self.returns_mean_a
        self.diffusion_mean_prior = MultivariateNormal(diffusion_mean_prior, self.returns_mean_b * diffusion_covariance)
        expected_return = self.diffusion_mean_prior.sample()

        # are we missing time scale Delta t? Assumed == 1
        diffusive_log_returns = expected_return[None, :] + \
                                MultivariateNormal(torch.zeros(diffusion_covariance.shape[0]), diffusion_covariance).sample(
                                    sample_shape=(self.number_of_realizations,))

        log_returns = diffusive_log_returns + jumps_size * arrivals_indicator

        data_loader = {"arrivals_intensity": arrivals_intensity,
                       "arrivals_indicator": arrivals_indicator,
                       "jump_mean": jump_mean,
                       "jump_covariance": jump_covariance,
                       "jump_size": jumps_size,
                       "diffusive_log_returns": diffusive_log_returns,
                       "log_returns": log_returns,
                       "diffusion_covariance":diffusion_covariance,
                       "expected_returns": expected_return}

        return data_loader

    @classmethod
    def get_parameters(cls):
        number_of_processes = 4
        number_of_realizations = 1000

        kwargs = {"jump_size_scale_prior": 1.,
                  "jump_size_a": 0.5,
                  "jump_size_b": 1.,
                  "jump_arrival_alpha": .5,
                  "jump_arrival_beta": .5,
                  "returns_mean_a": 1.,
                  "returns_mean_b": 1.,
                  "diffusion_covariance_normalization":0.5,
                  "number_of_processes": number_of_processes,
                  "number_of_realizations": number_of_realizations,
                  "model_path": os.path.join(project_path, 'results')}

        return kwargs

    def set_parameters(self, **kwargs):
        self.diffusion_covariance_normalization = kwargs.get("diffusion_covariance_normalization")
        self.number_of_processes = kwargs.get("number_of_processes")
        self.number_of_realizations = kwargs.get("number_of_realizations")
        self.T = self.number_of_realizations

        self.jump_size_scale_prior = kwargs.get("jump_size_scale_prior")
        self.jump_size_a = kwargs.get("jump_size_a")
        self.jump_size_b = kwargs.get("jump_size_b")
        self.jump_arrival_alpha = kwargs.get("jump_arrival_alpha")
        self.jump_arrival_beta = kwargs.get("jump_arrival_beta")

        self.returns_mean_a = kwargs.get("returns_mean_a")
        self.returns_mean_b = kwargs.get("returns_mean_b")

        self.average_bernoulli_probability = self.jump_arrival_alpha / (
                    self.jump_arrival_alpha + self.jump_arrival_beta)

    def update_parameters(self, dataloader, **kwargs):
        realizations = dataloader.get("log_returns")
        number_of_realizations = realizations.shape[0]
        locations = dataloader.get("locations")
        if locations is not None:
            locations_dimension = locations.shape[1]
            kwargs.update({"number_of_arrivals": locations.shape[0]})
            kwargs.update({"locations_dimension": locations_dimension})
        kwargs.update({"number_of_realizations": number_of_realizations})
        return kwargs

    def diffusive_and_jump_distribution(self, sigma, data_loader, monte_carlo_values):
        if isinstance(sigma, np.ndarray):
            sigma = torch.Tensor(sigma)

        jumps_size = torch.Tensor(monte_carlo_values["jumps_size"][-1])  # H
        jump_mean = torch.Tensor(monte_carlo_values["jumps_mean"][-1])

        arrivals_indicator = torch.Tensor(monte_carlo_values["arrivals_indicator"][-1]) # J
        jump_covariance = torch.Tensor(monte_carlo_values["jumps_covariance"][-1])

        interest_rate = data_loader["expected_returns"]
        log_returns = data_loader["log_returns"]

        sigma_inverse = torch.inverse(sigma)
        jump_covariance_inverse = torch.inverse(jump_covariance)
        diffusion_and_jump_covariance = torch.inverse(sigma_inverse + jump_covariance_inverse)

        jumps_size_posterior_covariance = torch.zeros(jumps_size.shape[0], jumps_size.shape[1], jumps_size.shape[1])
        jumps_size_posterior_covariance[torch.where(arrivals_indicator == 1.)[0]] = diffusion_and_jump_covariance[None,:, :]

        jumps_size_posterior_covariance[torch.where(arrivals_indicator == 0.)[0]] = jump_covariance[None, :, :]

        diffusive_mean_posterior = torch.matmul(sigma_inverse[None, :, :],
                                                (log_returns - interest_rate[None, :]).unsqueeze(-1))

        diffusive_mean_posterior = arrivals_indicator[:, None, None] * diffusive_mean_posterior

        jump_size_mean_posterior = torch.matmul(jump_covariance_inverse, jump_mean.unsqueeze(-1))
        jump_size_mean_posterior = diffusive_mean_posterior + jump_size_mean_posterior
        jump_size_mean_posterior = torch.matmul(jumps_size_posterior_covariance, jump_size_mean_posterior).squeeze()
        jump_posterior = MultivariateNormal(jump_size_mean_posterior, jumps_size_posterior_covariance)

        return jump_posterior

    def gibbs_jump_mean_covariance(self,data_loader, monte_carlo_values):
        """
        Here we follow:
        https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution

        :param data_loader:
        :param monte_carlo_values:
        :return:
        """
        a_J = self.jump_size_a
        b_J = self.jump_size_b
        H_t = monte_carlo_values["jumps_size"][-1]

        lambda_ = 1 / b_J
        mu_0 = a_J
        n = H_t.shape[0]
        y_av = H_t.mean(axis=0)

        mu_n = (lambda_ * mu_0 + n * y_av) / (lambda_ + n)
        lambda_n = lambda_ + n
        nu_n = self.nu + n

        s_0 = torch.Tensor(y_av - mu_0).unsqueeze(1)
        S_0 = torch.matmul(s_0, s_0.T).numpy()

        s = torch.Tensor(H_t - y_av).unsqueeze(-1)
        S = torch.matmul(s, s.permute(0, 2, 1)).sum(axis=0).numpy()

        Psi_n = self.Psi + S + ((lambda_ * n) / (lambda_ + n)) * S_0

        jumps_Sigma = invwishart(nu_n, Psi_n).rvs()
        jumps_mu = multivariate_normal(mu_n, jumps_Sigma / lambda_n).rvs()

        monte_carlo_values['jumps_mean'].append(jumps_mu)
        monte_carlo_values['jumps_covariance'].append(jumps_Sigma)

        return monte_carlo_values

    def gibbs_jump_size(self, data_loader, monte_carlo_values):
        sigma = torch.Tensor(monte_carlo_values["diffusion_covariance"][-1])
        jump_posterior = self.diffusive_and_jump_distribution(sigma, data_loader, monte_carlo_values)
        jump_values = jump_posterior.sample()
        monte_carlo_values["jumps_size"].append(jump_values.numpy())

        return monte_carlo_values

    def gibbs_arrival_indicator(self, data_loader, monte_carlo_values):
        jumps_size = torch.Tensor(monte_carlo_values["jumps_size"][-1])  # H
        sigma = torch.Tensor(monte_carlo_values["diffusion_covariance"][-1])
        arrivals_intensity = torch.Tensor(monte_carlo_values["arrivals_intensity"][-1])
        arrivals_indicator = torch.clone(torch.Tensor(monte_carlo_values["arrivals_indicator"][-1]))

        sigma_inverse = torch.inverse(sigma)

        log_returns = data_loader["log_returns"]
        interest_rate = data_loader["expected_returns"]

        for process_index in range(self.number_of_processes):
            arrivals_intensity_ = arrivals_intensity[process_index].item()

            # probability of arriving
            arrivals_indicator[:, process_index] = 1.
            indicator_mean = log_returns - interest_rate[None, :] - jumps_size * arrivals_indicator
            indicator_mean = einsum("bi,bij,bj->b", indicator_mean, sigma_inverse[None, :, :], indicator_mean)
            bernoulli_probability_1 = arrivals_intensity_ * torch.exp(-.5 * indicator_mean)

            # probability of not arriving
            arrivals_indicator[:, process_index] = 0.
            indicator_mean = (log_returns - interest_rate[None, :] - jumps_size * arrivals_indicator)
            indicator_mean = einsum("bi,bij,bj->b", indicator_mean, sigma_inverse[None, :, :], indicator_mean)
            indicator_mean = torch.exp(-.5 * indicator_mean)
            bernoulli_probability_0 = (1. - arrivals_intensity_) * indicator_mean
            bernoulli_probability = bernoulli_probability_1 / (
                        bernoulli_probability_0 + bernoulli_probability_1 + EPSILON)

            indicator_posterior = Bernoulli(bernoulli_probability)
            indicator = indicator_posterior.sample()
            arrivals_indicator[:, process_index] = indicator

        monte_carlo_values["arrivals_indicator"].append(arrivals_indicator.numpy())
        return monte_carlo_values

    def gibbs_arrivals_intensity(self, data_loader, monte_carlo_values):
        arrivals_indicator = monte_carlo_values["arrivals_indicator"][-1]
        indicator_sum = arrivals_indicator.sum()
        alpha_posterior = self.jump_arrival_alpha + indicator_sum
        beta_posterior = self.T - indicator_sum + self.jump_arrival_beta

        intensity = Beta(alpha_posterior, beta_posterior).sample()
        monte_carlo_values["arrivals_intensity"].append(intensity.item())

        return monte_carlo_values

    def gibbs_expected_returns_and_covariance(self,data_loader, monte_carlo_values):
        """
        Here we follow:

        https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution
        :param data_loader:
        :param monte_carlo_values:
        :return:
        """
        a_J = self.returns_mean_a
        b_J = self.returns_mean_b
        log_returns = torch.clone(data_loader["log_returns"])
        jumps_size = torch.Tensor(monte_carlo_values["jumps_size"][-1])  # H
        arrivals_indicator = torch.Tensor(monte_carlo_values["arrivals_indicator"][-1]) # J

        lambda_ = 1 / b_J
        mu_0 = a_J
        n = log_returns.shape[0]
        jumps_part = arrivals_indicator[:, None] * jumps_size
        log_returns = log_returns - jumps_part
        y_av = log_returns.mean(axis=0)

        mu_n = (lambda_ * mu_0 + n * y_av) / (lambda_ + n)
        lambda_n = lambda_ + n
        nu_n = self.nu + n

        s_0 = torch.Tensor(y_av - mu_0).unsqueeze(1)
        S_0 = torch.matmul(s_0, s_0.T).numpy()

        s = torch.Tensor(log_returns - y_av).unsqueeze(-1)
        S = torch.matmul(s, s.permute(0, 2, 1)).sum(axis=0).numpy()

        Psi_n = self.Psi + S + ((lambda_ * n) / (lambda_ + n)) * S_0

        diffusion_covariance = invwishart(nu_n, Psi_n).rvs()
        expected_returns = multivariate_normal(mu_n, diffusion_covariance / lambda_n).rvs()

        monte_carlo_values['expected_returns'].append(expected_returns)
        monte_carlo_values['diffusion_covariance'].append(diffusion_covariance)

        return monte_carlo_values

    def log_likelihood(self, data_loader, monte_carlo_parameters):
        """
        Log Likelihood of the Data

        :param data_loader:
        :param monte_carlo_values:

        :return:
        """
        diffusion_covariance = monte_carlo_parameters["diffusion_covariance"][-1]
        jumps_size = torch.Tensor(monte_carlo_parameters["jumps_size"][-1])  # Z
        arrivals_indicator = torch.Tensor(monte_carlo_parameters["arrivals_indicator"][-1])

        interest_rate = data_loader["expected_returns"]
        log_returns = data_loader["log_returns"]

        likelihood_mean = interest_rate[None, :] + jumps_size * arrivals_indicator
        realizations_pdf_new = MultivariateNormal(likelihood_mean, torch.Tensor(diffusion_covariance))

        return realizations_pdf_new.log_prob(log_returns).sum().item()

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = {}
        inference_parameters["nmc"] = 1000
        inference_parameters["burning"] = 200
        inference_parameters["metrics_logs"] = 200

        inference_parameters["train_diffusion_covariance"] = True
        inference_parameters["train_expected_returns"] = True #mu

        inference_parameters["train_jumps_arrival"] = True  # J
        inference_parameters["train_jumps_size"] = True  # Z
        inference_parameters["train_jumps_intensity"] = True  # Intensity
        inference_parameters["train_jumps_mean"] = True
        inference_parameters["train_jumps_covariance"] = True

        return inference_parameters

    def initialize_inference(self, data_loader=None, parameters=None, **inference_parameters):
        self.nmc = inference_parameters.get("nmc")
        self.metrics_logs = inference_parameters.get("metrics_logs")
        self.burning = inference_parameters.get("burning")

        assert self.nmc > self.burning,"MCMC time shorter than Burning"

        # covariance structure
        self.train_expected_returns = inference_parameters.get("train_expected_returns")
        self.train_diffusion_covariance = inference_parameters.get("train_diffusion_covariance")

        # discrete jumps structure
        self.train_jumps_intensity = inference_parameters.get("train_jumps_intensity")
        self.train_jumps_size = inference_parameters.get("train_jumps_size")
        self.train_jumps_arrival = inference_parameters.get("train_jumps_arrival")
        self.train_jumps_mean = inference_parameters.get("train_jumps_mean")
        self.train_jumps_covariance = inference_parameters.get("train_jumps_covariance")

        # =====================================
        # jumps structure
        # =====================================
        if self.train_jumps_intensity or data_loader["arrivals_intensity"] is None:
            arrivals_intensity = self.arrivals_intensity_prior.sample((self.number_of_processes,))
            self.train_jumps_intensity = True
        else:
            arrivals_intensity = data_loader["arrivals_intensity"]

        self.arrivals_indicator_prior = Bernoulli(arrivals_intensity)
        if self.train_jumps_arrival or data_loader["arrivals_indicator"] is None:
            arrivals_indicator = self.arrivals_indicator_prior.sample(sample_shape=(self.number_of_realizations,))
            self.train_jumps_arrival = True
        else:
            arrivals_indicator = data_loader["arrivals_indicator"]

        # sample from jump size prior
        if self.train_jumps_covariance or data_loader["jump_covariance"] is None:
            jump_covariance = self.covariance_jump_prior.rvs()
            jump_covariance = torch.Tensor(jump_covariance)
            self.train_jumps_covariance = True
        else:
            jump_covariance = data_loader["jump_covariance"]

        if self.train_jumps_mean or data_loader["jump_mean"] is None:
            jump_mean_prior = torch.ones(self.number_of_processes) * self.jump_size_a
            self.jump_mean_prior = MultivariateNormal(jump_mean_prior, self.jump_size_b * jump_covariance)
            jump_mean = self.jump_mean_prior.sample()
            self.train_jumps_mean = True
        else:
            jump_mean = data_loader["jump_mean"]

        if self.train_jumps_size or data_loader["jump_size"] is None:
            self.jump_distribution = MultivariateNormal(jump_mean, jump_covariance)
            jumps_size = self.jump_distribution.sample(sample_shape=(self.number_of_realizations,))
            self.train_jumps_size = True
        else:
            jumps_size = data_loader["jump_size"]

        if not self.train_jumps_size:
            if not self.train_jumps_arrival:
                expected_return = data_loader["expected_returns"]
                self.likelihood_mean = expected_return[None, :] + jumps_size * arrivals_indicator[:, None]

        arrivals_indicator = [arrivals_indicator.numpy()]
        arrivals_intensity = [arrivals_intensity.numpy()]
        jumps_size = [jumps_size.numpy()]
        jump_mean = [jump_mean.numpy()]
        jump_covariance = [jump_covariance.numpy()]

        # =======================================================================
        # DIFFUSION COVARIANCE
        # =======================================================================

        # sample from jump size prior
        if self.train_diffusion_covariance or data_loader["jump_covariance"] is None:
            diffusion_covariance = self.covariance_diffusion_prior.rvs()
            diffusion_covariance = torch.Tensor(diffusion_covariance)
            self.train_diffusion_covariance = True
        else:
            diffusion_covariance = data_loader["diffusion_covariance"]

        if self.train_expected_returns or data_loader["expected_returns"] is None:
            diffusion_mean_prior = torch.ones(self.number_of_processes) * self.returns_mean_a
            self.diffusion_mean_prior = MultivariateNormal(diffusion_mean_prior,
                                                           self.returns_mean_b * diffusion_covariance)
            expected_returns = self.diffusion_mean_prior.sample()
            self.train_expected_returns = True
        else:
            expected_returns = data_loader["expected_returns"]

        expected_returns = [expected_returns.numpy()]
        diffusion_covariance = [diffusion_covariance.numpy()]

        self.nu = self.number_of_processes + 1.
        self.Psi = np.random.rand(self.number_of_processes, self.number_of_processes)
        self.Psi = np.dot(self.Psi, self.Psi.transpose())

        monte_carlo_parameters = {"expected_returns":expected_returns,
                                  "diffusion_covariance":diffusion_covariance,
                                  "arrivals_indicator": arrivals_indicator,
                                  "arrivals_intensity": arrivals_intensity,
                                  "jumps_size": jumps_size,
                                  "jumps_mean": jump_mean,
                                  "jumps_covariance": jump_covariance}

        return monte_carlo_parameters

    def inference(self, data_loader, **inference_parameters):
        monte_carlo_parameters = self.initialize_inference(data_loader, None, **inference_parameters)
        print("#      ---------------- ")
        print("#      Start of MCMC    ")
        print("#      ---------------- ")
        for montecarlo_index in tqdm(range(self.nmc)):
            print("Monte Carlo {0}".format(montecarlo_index))
            #=============================================
            # Hyper parameters
            #=============================================
            if self.train_expected_returns and self.train_diffusion_covariance:
                monte_carlo_parameters = self.gibbs_expected_returns_and_covariance(data_loader, monte_carlo_parameters)

            if self.train_jumps_mean and self.train_jumps_covariance:
                monte_carlo_parameters = self.gibbs_jump_mean_covariance(data_loader, monte_carlo_parameters)
            #=============================================
            # Latent variables
            #=============================================
            if self.train_jumps_size:
                monte_carlo_parameters = self.gibbs_jump_size(data_loader, monte_carlo_parameters)
            if self.train_jumps_arrival:
                monte_carlo_parameters = self.gibbs_arrival_indicator(data_loader, monte_carlo_parameters)
            if self.train_jumps_intensity:
                monte_carlo_parameters = self.gibbs_arrivals_intensity(data_loader, monte_carlo_parameters)

            # METRICS
            if montecarlo_index > self.burning:
                if montecarlo_index % self.metrics_logs == 0:
                    self.inference_metrics(data_loader, monte_carlo_parameters, inference_parameters,montecarlo_index=montecarlo_index)
                    torch.save(monte_carlo_parameters,self.best_model_path)

        # METRICS END
        self.inference_metrics(data_loader, monte_carlo_parameters, inference_parameters, True,montecarlo_index=montecarlo_index)
        torch.save(monte_carlo_parameters, self.best_model_path)

        return monte_carlo_parameters

    #===========================================================
    # MODEL METRICS
    #===========================================================
    def arrivals_indicator_metrics(self, monte_carlo_parameters, data_loader, metrics_dict, end=False):
        real_arrivals_indicator = data_loader.get('arrivals_indicator')

        arrivals_indicator_stats = np.vstack(monte_carlo_parameters['arrivals_indicator'])
        arrivals_indicator_stats = arrivals_indicator_stats[self.burning:, :]

        arrivals_indicator_mean = arrivals_indicator_stats.mean(axis=1)
        arrivals_indicator_mean = arrivals_indicator_mean.mean().item()
        metrics_dict["mean_number_of_arrivals"] = arrivals_indicator_mean

        if real_arrivals_indicator is not None:
            arrivals_indicator_mean = arrivals_indicator_stats.mean(axis=0)
            fpr, tpr, thresholds = metrics.roc_curve(real_arrivals_indicator, arrivals_indicator_mean, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            metrics_dict["arrivals_indicator_auc"] = auc
            metrics_dict["mean_number_of_arrivals_real"] = real_arrivals_indicator.mean().item()

        return metrics_dict

    def arrivals_intensity_metrics(self, monte_carlo_parameters, data_loader, metrics_dict, end=False):
        arrivals_intensity_real = data_loader.get("arrivals_intensity")
        arrivals_intensity_stats = monte_carlo_parameters["arrivals_intensity"]

        metrics_dict["arrivals_intensity_mean"] = np.asarray(arrivals_intensity_stats).mean()
        metrics_dict["arrivals_intensity_std"] = np.asarray(arrivals_intensity_stats).std()

        if arrivals_intensity_real is not None:
            metrics_dict["arrivals_intensity_real"] = arrivals_intensity_real

        if end:
            stuff = plt.hist(arrivals_intensity_stats, alpha=0.3,bins=100)
            plt.axvline(x=metrics_dict["arrivals_intensity_mean"], c="red", label="mean")
            if arrivals_intensity_real is not None:
                plt.axvline(x=arrivals_intensity_real, c="green", label="real")
            plt.legend(loc="best")
            plt.savefig(os.path.join(self.model_dir, "arrivals_intensity.pdf"))
            plt.show()

        return metrics_dict

    def jumps_sizes_metrics(self, monte_carlo_parameters, data_loader, metrics_dict, end=False):
        # estimate errors from the full sample (including places with no arrivals)
        jumps_size_stats = np.dstack(monte_carlo_parameters["jumps_size"]).transpose(2, 0, 1)
        jumps_size_real = data_loader.get("jump_size")

        jumps_size_average = np.str_(jumps_size_stats.mean(axis=0).mean(axis=0)).tolist()
        jumps_size_std = np.str_(jumps_size_stats.std(axis=0).mean(axis=0)).tolist()
        metrics_dict["jumps_size_average"] = jumps_size_average
        metrics_dict["jumps_size_std"] = jumps_size_std

        if jumps_size_real is not None:
            jumps_size_real = jumps_size_real.detach().numpy()
            error_per_step = np.sqrt((jumps_size_stats.mean(axis=0) - jumps_size_real) ** 2).mean()
            metrics_dict["full_jumps_size_errors_per_step"] = str(error_per_step)

            jumps_size_average_real = np.str_(jumps_size_real.mean(axis=0)).tolist()
            jumps_size_std_real = np.str_(jumps_size_real.std(axis=0)).tolist()
            metrics_dict["jumps_size_average_real"] = jumps_size_average_real
            metrics_dict["jumps_size_std_std"] = jumps_size_std_real

        if self.train_jumps_arrival:
            pass
        else:
            arrivals_indicator_real = data_loader.get("arrivals_indicator")
            if arrivals_indicator_real is not None:
                observed_jumps_size_real = jumps_size_real[np.where(arrivals_indicator_real == 1)]
                observed_jumps_size_real_average = np.str_(observed_jumps_size_real.mean(axis=0)).tolist()
                observed_jumps_size_real_std = np.str_(observed_jumps_size_real.std(axis=0)).tolist()

                observer_jumps_size = jumps_size_stats.mean(axis=0)[np.where(arrivals_indicator_real == 1)]
                observer_jumps_size_average = np.str_(observer_jumps_size.mean(axis=0)).tolist()
                observer_jumps_size_std = np.str_(observer_jumps_size.std(axis=0)).tolist()

                metrics_dict["observed_jumps_size_real_average"] = observed_jumps_size_real_average
                metrics_dict["observer_jumps_size_average"] = observer_jumps_size_average

                metrics_dict["observed_jumps_size_real_std"] = observed_jumps_size_real_std
                metrics_dict["observer_jumps_size_std"] = observer_jumps_size_std

        return metrics_dict

    def likelihood_metric(self, monte_carlo_parameters, data_loader, metrics_dict, end=False):
        """

        :param monte_carlo_parameters:
        :param data_loader:
        :param metrics_dict:
        :param end:

        :return:
        """
        log_likelihood = self.log_likelihood(data_loader, monte_carlo_parameters)
        metrics_dict.update({"log_likelihood":log_likelihood})
        return metrics_dict

    def inference_metrics(self, data_loader, monte_carlo_parameters, inference_parameters, end=False,
                          montecarlo_index=None):
        """
        Here we also include possible plots if we call at the end of
        the inference procedure (end == True)

        :param data_loader:
        :param monte_carlo_parameters:
        :param inference_parameters:
        :param end:
        :param montecarlo_index:
        :return:
        """
        metrics_dict = {}
        with open(self.inference_path, "a+") as f:
            if self.train_jumps_size:
                metrics_dict = self.jumps_sizes_metrics(monte_carlo_parameters, data_loader, metrics_dict, end)
            if self.train_jumps_arrival:
                pass
                #metrics_dict = self.arrivals_indicator_metrics(monte_carlo_parameters, data_loader, metrics_dict, end)
            if self.train_jumps_intensity:
                pass
            #    metrics_dict = self.arrivals_intensity_metrics(monte_carlo_parameters, data_loader, metrics_dict, end)

            metrics_dict = self.likelihood_metric(monte_carlo_parameters, data_loader, metrics_dict, end)

            metrics_dict.update({"montecarlo_index": montecarlo_index})
            json.dump(metrics_dict, f)
            f.write("\n")
            f.flush()


if __name__ == "__main__":
    from bdp import data_path
    #===================================================================
    # MERTON JUMPS
    #===================================================================
    model_param = MertonJumpsPoissonCovariance.get_parameters()
    new_sample = True
    from bdp import data_path
    from pathlib import Path

    data_dir = Path(data_path) /"raw" / "merton_poisson_covariance"
    #data_dir = os.path.join(data_path,"raw\merton_poisson_covariance")
    #my_data_path = os.path.join(data_dir, "merton_jump_simulation_.tr")
    my_data_path = data_dir / "merton_jump_simulation_.tr"

    if new_sample:
        model_param.update({"jump_size_a": 5.})
        model_param.update({"jump_size_b": 1.})
        model_param.update({"jump_arrival_alpha": .5,})
        model_param.update({"jump_arrival_beta": .5,})
        model_param.update({"returns_mean_a": 1.,})
        model_param.update({"returns_mean_b": 1.,})
        model_param.update({"diffusion_covariance_normalization": 1.})

        mjpc_s = MertonJumpsPoissonCovariance(None, None, None, **model_param)
        data_loader = mjpc_s.sample()
        data_ = {"data_loader":data_loader,"model_param":model_param}
        torch.save(data_,my_data_path)
    else:
        raise Exception("New Data is Not ready")
        #data_ = torch.load(my_data_path)
        #data_loader = data_["data_loader"]
        #model_param = data_["model_param"]
    #========================================================================
    # inference from simulated date
    inference_param = MertonJumpsPoissonCovariance.get_inference_parameters()

    inference_param.update({"train_jumps_intensity": False})
    inference_param.update({"train_jumps_arrival": True})
    inference_param.update({"train_jumps_size": False})
    inference_param.update({"train_expected_returns": False})
    inference_param.update({"train_diffusion_covariance": False})
    inference_param["train_jumps_mean"] = False
    inference_param["train_jumps_covariance"] = False

    inference_param.update({"nmc": 1000,
                            "burning": 500,
                            "metrics_logs": 50})

    mjpc = MertonJumpsPoissonCovariance(None, data_loader, None, **model_param)
    monte_carlo_parameters = mjpc.inference(data_loader, **inference_param)

    #===================================================================
    # COLLECT REAL DATA
    #===================================================================
    """
    date_string = "2021-06-14"
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder,date_string)

    kwargs = {"path_to_data": data_folder,
              "date_string":date_string,
              "batch_size": 29,
              "steps_ahead": 10,
              "span": "full"}

    crypto_data_loader = CryptoDataLoader('cpu', **kwargs)
    data_batch = next(crypto_data_loader.train.__iter__())

    # defines portfolio to study
    date0 = datetime(2018, 1, 1)
    datef = datetime(2019, 1, 1)
    crypto_data_loader.set_portfolio_assets("2021-06-14",
                                            "full",
                                            predictor=None,
                                            top=4,
                                            date0=None,
                                            datef=None,
                                            max_size=4)
    """


    #===================================================================
    # MERTON JUMPS REAL
    #===================================================================
    """
    from deep_fields.models_.random_fields.mcmc_preprocessing import estimate_mcmc_dataloader_and_parameters_mertonjump
    from deep_fields.models_.random_fields.mcmc_preprocessing import preprocess_cryptoportfolio_for_mertonjump

    inference_param = MertonJumpsPoissonCovariance.get_inference_parameters()
    inference_param.update({"nmc": 1000,
                            "burning": 5,
                            "metrics_logs": 1})

    corrected_prices_data = preprocess_cryptoportfolio_for_mertonjump(crypto_data_loader)
    model_param,data_loader = estimate_mcmc_dataloader_and_parameters_mertonjump(corrected_prices_data,False)
    mjpc = MertonJumpsPoissonCovariance(None, data_loader, None, **model_param)
    monte_carlo_parameters = mjpc.inference(data_loader, **inference_param)
    """
    #======================================================
    # SIMPLE POISSON COVARIANCE
    #======================================================
    """
    model_param = PoissonCovariance.get_parameters()
    inference_param = PoissonCovariance.get_inference_parameters()

    pc_s = PoissonCovariance(None, None, None, **model_param)
    data_loader = pc_s.sample()

    pprint(inference_param)
    pc = PoissonCovariance(None, data_loader, None, **model_param)
    monte_carlo_parameters = pc.inference(data_loader, **inference_param)

    model_param = PoissonCovariance.get_parameters()
    inference_param = PoissonCovariance.get_inference_parameters()
    """