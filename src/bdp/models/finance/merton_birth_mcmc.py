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
from bdp.data.crypto.dataloaders import CryptoDataLoader, ADataLoader
from bdp.models.gaussian_processes.gaussian_processes import multivariate_normal, white_noise_kernel

from torch import matmul as m
from bdp import project_path
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import invwishart, bernoulli


class MertonBirthPoissonCovariance(DeepBayesianModel):
    """
    Here we learn a market growth process essentially

    """
    def __init__(self, model_dir=None, data_loader: ADataLoader = None, model_name=None, **kwargs):
        if model_name is None:
            model_name = "merton_birth_poisson_covariance"

        DeepBayesianModel.__init__(self,
                                   model_name,
                                   model_dir=model_dir,
                                   data_loader=data_loader,
                                   **kwargs)

    @classmethod
    def get_parameters(cls):
        locations_dimension = 2
        kernel_parameters = {"kernel_sigma": 0.5,
                             "kernel_lenght_scales": [1., 2.]}

        number_of_realizations = 50

        kwargs = {"locations_dimension": locations_dimension,
                  "jump_size_scale_prior": 1.,
                  "birth_intensity":4.,
                  "returns_mean_a": 1.,
                  "returns_mean_b": 1.,
                  "prior_locations_mean": 0.,
                  "prior_locations_std": 1.,
                  "prior_sigma_mean": 0.,
                  "prior_sigma_std": 1.,
                  "prior_length_mean": 0.,
                  "prior_length_std": 1.,
                  "kernel_parameters": kernel_parameters,
                  "number_of_realizations": number_of_realizations,
                  "model_path": os.path.join(project_path, 'results')}

        return kwargs

    def define_deep_models(self):
        inference_parameters = {}

        # BIRTH
        self.arrivals_intensity_prior = Poisson(self.birth_intensity)

        # LOCATIONS PRIORS
        self.locations_prior = Normal(torch.full((self.locations_dimension,), self.prior_locations_mean),
                                      torch.full((self.locations_dimension,), self.prior_locations_std))

        # KERNEL PRIORS
        self.sigma_prior = Normal(torch.full((1,), self.prior_sigma_mean),
                                  torch.full((1,), self.prior_sigma_std))

        self.lenght_prior = Normal(torch.full((self.locations_dimension,), self.prior_length_mean),
                                   torch.full((self.locations_dimension,), self.prior_length_std))

        # KERNEL
        self.kernel = ScaleKernel(RBFKernel(ard_num_dims=self.locations_dimension, requires_grad=True),
                                  requires_grad=True) + white_noise_kernel()
        self.initialize_kernel()

    def set_parameters(self, **kwargs):
        self.number_of_realizations = kwargs.get("number_of_realizations")
        self.T = self.number_of_realizations

        #birth
        self.birth_intensity = kwargs.get("birth_intensity")

        #locations
        self.locations_dimension = kwargs.get("locations_dimension")
        self.prior_locations_mean = kwargs.get("prior_locations_mean")
        self.prior_locations_std = kwargs.get("prior_locations_std")

        #kernel
        self.prior_sigma_mean = kwargs.get("prior_sigma_mean")
        self.prior_sigma_std = kwargs.get("prior_sigma_std")
        self.prior_length_mean = kwargs.get("prior_length_mean")
        self.prior_length_std = kwargs.get("prior_length_std")

        self.kernel_parameters = kwargs.get("kernel_parameters")
        self.kernel_sigma = self.kernel_parameters.get("kernel_sigma")
        self.kernel_lenght_scales = self.kernel_parameters.get("kernel_lenght_scales")

        # expected returns
        self.returns_mean_a = kwargs.get("returns_mean_a")
        self.returns_mean_b = kwargs.get("returns_mean_b")

    def update_parameters(self, dataloader, **kwargs):
        realizations = dataloader.get("log_returns")
        number_of_realizations = realizations.shape[0]
        locations = dataloader.get("locations_history")
        if locations is not None:
            locations_dimension = locations.shape[1]
            kwargs.update({"number_of_births": locations.shape[0]})
            kwargs.update({"locations_dimension": locations_dimension})
        kwargs.update({"number_of_realizations": number_of_realizations})

        self.birth_numbers = dataloader.get("birth_numbers")
        self.assets_in_the_market = dataloader.get("assets_in_the_market")
        self.total_assets_in_history = dataloader.get("total_assets_in_history")

        return kwargs

    def initialize_kernel(self, kernel_sigma, kernel_lenght_scales):
        kernel_hypers = {"raw_outputscale": torch.tensor(kernel_sigma),
                         "base_kernel.raw_lengthscale": torch.tensor(kernel_lenght_scales)}
        self.kernel.kernels[0].initialize(**kernel_hypers)

    def define_kernel(self, kernel_sigma, kernel_lenght_scales):
        self.initialize_kernel(kernel_sigma,kernel_lenght_scales)
        kernel_eval = lambda locations: self.kernel(locations, locations).evaluate().float()
        return self.kernel, kernel_eval

    def sample(self):
        """
        :return:
        """
        number_of_realizations = self.number_of_realizations

        # Births
        birth_distribution = Poisson(self.birth_intensity)
        birth_numbers = birth_distribution.sample((number_of_realizations,)).long()
        assets_in_the_market = birth_numbers.cumsum(dim=0)
        total_assets_in_history = assets_in_the_market[-1]

        # Locations
        locations_history = self.locations_prior.sample(sample_shape=(total_assets_in_history,))

        # Kernel
        kernel, kernel_eval = self.define_kernel(self.kernel_sigma, self.kernel_lenght_scales)
        covariance_diffusion_history = kernel_eval(locations_history)

        # expected returns -------------
        returns_mean_prior = torch.ones(total_assets_in_history) * self.returns_mean_a
        returns_covariance_prior = self.returns_mean_b * covariance_diffusion_history
        expected_returns_distribution = MultivariateNormal(returns_mean_prior, returns_covariance_prior)
        expected_returns_history = expected_returns_distribution.sample()

        log_returns = torch.zeros((self.number_of_realizations, total_assets_in_history))

        print("Total number of assets {0}".format(total_assets_in_history))
        # market birth process
        for time_index in range(self.number_of_realizations):
            print(time_index)
            current_number_of_assets = assets_in_the_market[time_index]
            print("{0} Assets out of {1}".format(current_number_of_assets,total_assets_in_history))
            if current_number_of_assets > 0:
                current_expected_returns = expected_returns_history[:current_number_of_assets]
                current_covariance = covariance_diffusion_history[:current_number_of_assets, :current_number_of_assets]
                current_log_returns_distribution = MultivariateNormal(current_expected_returns, current_covariance)
                current_log_returns = current_log_returns_distribution.sample()

                log_returns[time_index, :current_number_of_assets] = current_log_returns

        data_loader = {"birth_numbers":birth_numbers,
                       "assets_in_the_market":assets_in_the_market,
                       "total_assets_in_history":total_assets_in_history,
                       "log_returns": log_returns,
                       "locations_history": locations_history,
                       "kernel_sigma": self.kernel_sigma,
                       "kernel_lenght_scales": self.kernel_lenght_scales,
                       "covariance_diffusion_history": covariance_diffusion_history,
                       "expected_returns_history": expected_returns_history,
                       "kernel": kernel}

        return data_loader

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
        self.timef = datetime.now()
        time_of_inference = (self.timef - self.time0).total_seconds()
        metrics_dict = {}
        with open(self.inference_path, "a+") as f:
            if self.train_sigma_kernel:
                pass
            if self.train_lenght_kernel:
                pass
            if self.train_locations:
                metrics_dict = self.locations_metrics(monte_carlo_parameters, data_loader, metrics_dict, end)

            metrics_dict.update({"montecarlo_index": montecarlo_index,
                                 "time_of_inference":time_of_inference})

            json.dump(metrics_dict, f)
            f.write("\n")
            f.flush()
            if end:
                print("Time to Inference {0}".format(time_of_inference))

    def locations_metrics(self, monte_carlo_parameters, data_loader, metrics_dict, end=False):
        locations_mcmc = monte_carlo_parameters["locations_history"]
        covariance_real = data_loader['covariance_diffusion_history']

        locations_mcmc = torch.Tensor(locations_mcmc).permute(1, 0, 2)
        locations_mcmc = locations_mcmc[self.burning:, :, :]

        number_of_steps = 0
        covariance_stats = torch.zeros_like(
            monte_carlo_parameters["kernel"](locations_mcmc[0], locations_mcmc[0]).evaluate().detach())
        for locations in locations_mcmc:
            covariance = monte_carlo_parameters["kernel"](locations, locations).evaluate().detach().numpy()
            covariance_stats += covariance
            number_of_steps += 1

        covariance_stats /= number_of_steps
        metrics_dict["covariance_metrics"] = covariance_stats.detach().numpy().tolist()
        metrics_dict["covariance_real"] = covariance_real.detach().numpy().tolist()

        return metrics_dict

    def inference(self, data_loader, **inference_parameters):
        monte_carlo_parameters = self.initialize_inference(data_loader, None, **inference_parameters)
        print("#      ---------------- ")
        print("#      Start of MCMC    ")
        print("#      ---------------- ")

        for montecarlo_index in tqdm(range(self.nmc)):
            print("Monte Carlo {0}".format(montecarlo_index))
            if self.train_sigma_kernel:
                monte_carlo_parameters = self.gibbs_sigma(monte_carlo_parameters)
            if self.train_lenght_kernel:
                monte_carlo_parameters = self.gibbs_lenght(monte_carlo_parameters)
            if self.train_locations:
                for location_index in range(self.total_assets_in_history):
                    print("Location {0} out of {1}".format(location_index,self.total_assets_in_history))
                    monte_carlo_parameters = self.gibbs_locations(data_loader, location_index, monte_carlo_parameters)
            # METRICS
            if montecarlo_index > self.burning:
                if montecarlo_index % self.metrics_logs == 0:
                    self.inference_metrics(data_loader, monte_carlo_parameters, inference_parameters,montecarlo_index=montecarlo_index)
                    torch.save(monte_carlo_parameters,self.best_model_path)

        timef = datetime.now()
        # METRICS END
        self.inference_metrics(data_loader, monte_carlo_parameters, inference_parameters, True,montecarlo_index=montecarlo_index)
        torch.save(monte_carlo_parameters, self.best_model_path)

        return monte_carlo_parameters

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = {}
        inference_parameters["nmc"] = 1000
        inference_parameters["burning"] = 200
        inference_parameters["metrics_logs"] = 200

        inference_parameters["train_sigma_kernel"] = False
        inference_parameters["train_lenght_kernel"] = False
        inference_parameters["train_locations"] = False
        inference_parameters["train_birth_intensity"] = False

        return inference_parameters

    def initialize_inference(self, data_loader: ADataLoader, parameters=None, **inference_parameters):
        """
        HERE WE DEFINE THE MONTE CARLO PARAMETERS

        :param data_loader:
        :param parameters:
        :param inference_parameters:

        :return: monte_carlo_parameters dict
            here we carry the statistics of the mcmc simulation
        """
        self.time0 = datetime.now()

        self.nmc = inference_parameters.get("nmc")
        self.metrics_logs = inference_parameters.get("metrics_logs")
        self.burning = inference_parameters.get("burning")

        # covariance structure
        self.train_birth_intensity = inference_parameters.get("train_birth_intensity")
        self.train_sigma_kernel = inference_parameters.get("train_sigma_kernel")
        self.train_lenght_kernel = inference_parameters.get("train_lenght_kernel")
        self.train_locations = inference_parameters.get("train_locations")

        # birth numbers
        self.birth_numbers = data_loader.get("birth_numbers")
        self.assets_in_the_market = data_loader.get("assets_in_the_market")
        self.total_assets_in_history = data_loader.get("total_assets_in_history")

        # =======================================================================
        # DIFFUSION COVARIANCE THROUGH HIDDEN POISSON PROCESS
        # =======================================================================

        if self.train_locations or data_loader["locations_history"] is None:
            locations_0 = [[self.locations_prior.sample().detach().numpy()] for i in range(self.total_assets_in_history)]
        else:
            locations_0 = data_loader["locations_history"]
            locations_0 = [[locations_0[i].detach().numpy()] for i in range(self.total_assets_in_history)]

        if self.train_sigma_kernel:
            kernel_sigma_0 = self.sigma_prior.sample().detach().numpy()
        else:
            kernel_sigma_0 = data_loader["kernel_sigma"]

        if self.train_lenght_kernel:
            kernel_lenghts_0 = self.lenght_prior.sample().detach().numpy()
        else:
            kernel_lenghts_0 = data_loader["kernel_lenght_scales"]

        kernel, kernel_eval = self.define_kernel(kernel_sigma_0, kernel_lenghts_0)

        K = kernel_eval(torch.Tensor(locations))

        monte_carlo_parameters = {"K": K.detach().numpy(),
                                  "kernel": kernel,
                                  "kernel_sigma": [kernel_sigma_0],
                                  "kernel_lenght_scales": [kernel_lenghts_0],
                                  "locations_history": locations_0}

        return monte_carlo_parameters

    def locations_sample_to_tensor(self, locations_mcmc):
        locations_now = []
        for locations_series in locations_mcmc:
            locations_now.append(torch.tensor(locations_series[-1]).unsqueeze(0))
        return torch.cat(locations_now, dim=0)

    #@timeit
    def gibbs_locations(self, data_loader, location_index, monte_carlo_values):
        locations_sample_now = monte_carlo_values["locations_history"]
        K = monte_carlo_values["K"]
        kernel = monte_carlo_values["kernel"]

        locations = self.locations_sample_to_tensor(locations_sample_now)
        location_now = locations_sample_now[location_index][-1]
        location_proposal = self.locations_prior.sample().detach().numpy()

        new_location, ll = elliptical_slice(initial_theta=location_now,
                                            prior=location_proposal,
                                            lnpdf=self.birth_log_likelihood_realizations,
                                            pdf_params=(locations, location_index, data_loader, monte_carlo_values))

        K_new = new_kernel(locations, location_index,
                           torch.Tensor(new_location).unsqueeze(0),
                           self.total_assets_in_history.item(),K, kernel)

        monte_carlo_values["locations_history"][location_index].append(new_location)
        monte_carlo_values["K"] = K_new

        return monte_carlo_values

    #@timeit
    def birth_log_likelihood_realizations(self,new_location, locations, location_index, data_loader, monte_carlo_values):
        """
        We calculate the log likelihood at each steps making sure that we get the right assets at
        each time step
        """
        K = monte_carlo_values["K"]
        kernel = monte_carlo_values["kernel"]

        interest_rate = data_loader["expected_returns_history"]
        log_returns = data_loader["log_returns"]

        if isinstance(locations, np.ndarray):
            locations = torch.Tensor(locations)
        if isinstance(new_location, np.ndarray):
            new_location = torch.Tensor(new_location)

        index_left = list(range(mbpc.total_assets_in_history.item()))
        index_left.remove(location_index)

        new_location = torch.Tensor(new_location).unsqueeze(0)
        K_new = new_kernel(locations, location_index, new_location, mbpc.total_assets_in_history.item(), K, kernel)

        current_log_probability = 0.
        number_of_realizations = log_returns.shape[0]
        for time_index in range(number_of_realizations):
            current_assets_in_the_market = self.assets_in_the_market[time_index]
            current_interest_rate = interest_rate[None, :current_assets_in_the_market]
            current_diffusion_covariance = K_new[:current_assets_in_the_market, :current_assets_in_the_market]
            current_diffusion_distribution = MultivariateNormal(current_interest_rate,
                                                                torch.Tensor(current_diffusion_covariance))
            current_log_returns = log_returns[time_index, :current_assets_in_the_market]

            current_log_probability += current_diffusion_distribution.log_prob(current_log_returns[None, :])
        return current_log_probability
    

if __name__ == "__main__":
    from bdp import data_path
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
    # MERTON BIRTHS
    #===================================================================
    """
    data_dir = "C:/Users/cesar/Desktop/Projects/General/deep_random_fields/data/raw/merton_birth_covariance/"
    my_data_path = os.path.join(data_dir, "merton_birth_simulation.tr")

    model_param = MertonBirthPoissonCovariance.get_parameters()
    inference_param = MertonBirthPoissonCovariance.get_inference_parameters()

    inference_param.update({"nmc": 300,
                            "burning": 150,
                            "metrics_logs": 50})
    inference_param.update({"train_locations": True})

    mbpc_s = MertonBirthPoissonCovariance(None, None, None, **model_param)

    data_loader = mbpc_s.sample()
    data_ = {"data_loader":data_loader,"model_param":model_param}
    torch.save(data_,my_data_path)

    data_ = torch.load(my_data_path)
    data_loader = data_["data_loader"]
    model_param = data_["model_param"]

    mbpc = MertonBirthPoissonCovariance(None, data_loader, None, **model_param)
    monte_carlo_parameters = mbpc.inference(data_loader,**inference_param)
    """