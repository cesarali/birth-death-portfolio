import os
import torch
from torch import nn

import numpy as np
import pandas as pd
from pprint import pprint
from numpy.linalg import svd

from torch.distributions import Gamma,Normal
from deep_fields import project_path, data_path
from deep_fields.models.abstract_models import DeepBayesianModel
from deep_fields.data.crypto.dataloaders import CryptoDataLoader, ADataLoader
from deep_fields.models.crypto.predictors import PredictorsModelFactory as predictor_factory
from deep_fields.models.deep_architectures.tcn import TemporalConvNet


from gpytorch.kernels import RBFKernel, ScaleKernel
from deep_fields.models.gaussian_processes.gaussian_processes import MV_Normal
from deep_fields.models.gaussian_processes.gaussian_processes import multivariate_normal, white_noise_kernel

from bovy_mcmc import elliptical_slice
from deep_fields.models.crypto.portfolio_objectives import  excess_return
from deep_fields.models.deep_architectures.deep_nets import MLP

predictor_factory = predictor_factory()

class NonparametricStochasticPortfolio(DeepBayesianModel):

    def __init__(self, model_dir=None, data_loader: ADataLoader = None, model_name=None, **kwargs):
        if model_name is None:
            model_name = "nonparametric_stochastic_portfolio"

        DeepBayesianModel.__init__(self,
                                   model_name,
                                   model_dir=model_dir,
                                   data_loader=data_loader,
                                   **kwargs)

    @classmethod
    def get_parameters(self):
        kwargs = {"dropout": .4,
                  "predictor":{"name":"crypto_seq2seq",
                               "model_dir":os.path.join(project_path,
                                                        'results',
                                                        'crypto_seq2seq',
                                                        '1637576663')},
                  "portfolio":"nonparametric",
                  "dimension":3,
                  "steps_ahead":14,
                  "number_of_assets":10,
                  "kernel_size": 3,  # TCN values
                  "number_of_levels": 10,
                  "time_encoding": 8,
                  "objective":"excess_returns",
                  "model_path": os.path.join(project_path, 'results')}

        return kwargs

    def update_parameters(self, dataloader, **kwargs):
        kwargs.update({"number_of_assets":dataloader.batch_size})
        return kwargs

    def set_parameters(self,**kwargs):
        self.predictor_parameters = kwargs.get("predictor")

        self.performance = kwargs.get("performance")
        self.portfolio_type = kwargs.get("portfolio")
        self.number_of_assets = kwargs.get("number_of_assets")
        self.steps_ahead = kwargs.get("steps_ahead")
        self.dimension = kwargs.get("dimension")

        self.dropout = kwargs.get("dropout")
        #==============================================
        # ENCODER
        #==============================================
        # TCN parameters
        self.portfolio_dimension = self.dimension*self.number_of_assets
        self.time_encoding = kwargs.get("time_encoding")

        self.number_of_levels = kwargs.get("number_of_levels")
        self.kernel_size = kwargs.get("kernel_size")
        self.num_channels = [self.time_encoding] * self.number_of_levels
        self.receptive_field = 2 ** len(self.num_channels)

    def define_deep_models(self):
        predictors_name = self.predictor_parameters.get("name")
        # poisson process
        self.predictor = predictor_factory.create(predictors_name,**self.predictor_parameters)
        self.dimension = self.predictor.dimension
        try:
            self.covariates_dimension = self.predictor.covariates_dimension
        except:
            self.covariates_dimension = 0

        #==============================================
        # ENCODE PORTFOLIO
        #==============================================
        self.TCN = TemporalConvNet(self.portfolio_dimension,
                                   self.num_channels,
                                   self.kernel_size,
                                   dropout=self.dropout)

        #nn.Linear(self.time_encoding,self.number_of_assets)
        self.encode_to_hidden = MLP(input_dim=self.time_encoding,
                                    layers_dim=[10,10],
                                    output_dim=self.number_of_assets,
                                    output_transformation=None,
                                    dropout=self.dropout)

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()
        return inference_parameters

    def encoder(self,databatch,use_case="train"):
        if use_case == "train":
            series = databatch.pmv
            batch_size = series.shape[0]
            sequence_lenght = series.shape[1]
            dimension = series.shape[2]

            sequence_lenght_ = sequence_lenght - self.steps_ahead + 1
            portfolio_series = databatch.pmv[:,:-(self.steps_ahead-1),:]
            portfolio_series = portfolio_series.permute(0,2,1)
            portfolio_series = portfolio_series.reshape(batch_size * dimension, -1).unsqueeze(0)

            hidden_state = self.TCN(portfolio_series).permute(0,2,1) #[batch_size,sequence_lenght_,time_encoding]
            hidden_state = self.encode_to_hidden(hidden_state)#[batch_size*sequence_lenght_,decoder_hidden_state*dimension]

            return hidden_state

        elif use_case == "prediction":
            series = databatch.pmv
            batch_size = series.shape[0]
            sequence_lenght = series.shape[1]
            sequence_lenght_ = sequence_lenght - self.steps_ahead + 1
            pmv = databatch.pmv[:,:,:]
            pmv = pmv.permute(0,2,1)
            encoding = self.TCN(pmv).permute(0,2,1) #[batch_size,sequence_lenght_,time_encoding]
            encoder_ = encoding.reshape(batch_size * sequence_lenght, -1)#[batch_size*sequence_lenght_,time_encoding]
            hidden_state = self.encode_to_hidden(encoder_)#[batch_size*sequence_lenght_,decoder_hidden_state*dimension]
            return hidden_state

    def loss(self, databatch, forward_results, dataloader, epoch):
        policy, unfolded_series = forward_results
        prices = unfolded_series[:, :, :, 0]
        prices_below = prices[:, :, 0]
        prices_ahead = prices[:, :, -1]
        loss = excess_return(policy, prices_ahead, prices_below).mean()

        print(loss)
        return {"loss":loss}

    def forward(self,databatch,use_case="train"):
        if use_case == "train":
            series = databatch.pmv
            batch_size = series.shape[0]
            sequence_lenght = series.shape[1]
            dimension = series.shape[2]

            steps_ahead = self.steps_ahead
            sequence_lenght_ = sequence_lenght - steps_ahead + 1

            # we unfold the series so at each step on time we predict (steps_ahead)
            unfolded_series = series.unfold(dimension=1, size=steps_ahead, step=1).contiguous().permute(0,1,3,2)

            hidden_state = self.encoder(databatch) #[batch_size*sequence_lenght_,time_encoding]
            policy = torch.softmax(hidden_state, dim=-1).squeeze().permute(1,0)

            return policy, unfolded_series

        elif use_case == "prediction":
            series = databatch.pmv
            batch_size = series.shape[0]
            sequence_lenght = series.shape[1]

            # we unfold the series so at each step on time we predict (steps_ahead)
            hidden_state = self.encoder(databatch,use_case="prediction") #[batch_size*sequence_lenght,time_encoding]
            output,hidden_state = self.decoder(hidden_state,series[:, :, :],batch_size,
                                               sequence_lenght_=sequence_lenght,use_case="prediction")
            return output, series

    def initialize_inference(self,data_loader: ADataLoader, parameters=None, **inference_parameters):
        super().initialize_inference(data_loader=data_loader, parameters=parameters, **inference_parameters)
        date = inference_parameters.get("date")
        span = inference_parameters.get("span")

class GaussianProcessPortfolio(NonparametricStochasticPortfolio):

    def __init__(self,model_dir=None, data_loader: ADataLoader = None, model_name=None, **kwargs):
        if model_name is None:
            model_name = "gp_stochastic_portfolio"
        kwargs.update({"model_name":model_name})
        NonparametricStochasticPortfolio.__init__(self,**kwargs)

    @classmethod
    def get_parameters(self):
        kwargs = super().get_parameters()
        kwargs.update({"portfolio":"bayesian_parametric"})
        kwargs.update({"gp_grid_size":10})
        kwargs.update({"gamma_a":7.0})
        kwargs.update({"gamma_b":0.5})
        #kwargs.update({"portfolio": "bayesian_nonparametric"})
        return kwargs

    def set_parameters(self,**kwargs):
        super().set_parameters(**kwargs)
        self.portfolio_type = kwargs.get("portfolio")
        self.gp_grid_size = kwargs.get("gp_grid_size")
        self.mean_0 = torch.zeros(self.gp_grid_size)
        self.gamma_a = kwargs.get("gamma_a")
        self.gamma_b = kwargs.get("gamma_b")

    @classmethod
    def get_inference_parameters(cls):
        inference_parameters = super().get_inference_parameters()

        inference_parameters.update({"date": "2021-06-14"})
        inference_parameters.update({"span": "full"})

        inference_parameters.update({"regularizers": {"nll": None}})
        inference_parameters.update({"nmc": 1000})
        inference_parameters.update({"start": 0})
        inference_parameters.update({"end": None})
        inference_parameters.update({"steps_ahead":1})
        inference_parameters.update({"performance_metric":"excess_return"})

        inference_parameters.update({"prior_x_mean":0.})
        inference_parameters.update({"prior_x_std":1.})

        inference_parameters.update({"prior_sigma_mean":0.})
        inference_parameters.update({"prior_sigma_std":1.})

        inference_parameters.update({"prior_length_mean":0.})
        inference_parameters.update({"prior_length_std":1.})

        return inference_parameters

    def define_deep_models(self):
        super().define_deep_models()
        self.grid_boundaries = torch.linspace(0., 1., self.gp_grid_size-1) # bucket for prices / market cap / volume
        self.grid =  torch.linspace(0., 1., self.gp_grid_size)
        self.gamma_likelihood = Gamma(7., 0.5)

    def initialize_inference(self,data_loader: ADataLoader, parameters=None, **inference_parameters):
        super().initialize_inference(data_loader, parameters, **inference_parameters)
        self.nmc = inference_parameters.get("nmc")
        self.start = inference_parameters.get("start")
        self.end = inference_parameters.get("end")
        self.steps_ahead = inference_parameters.get("steps_ahead")
        self.performance_metric_name = inference_parameters.get("performance_metric")

        self.gp_dimension = self.gp_grid_size**self.dimension
        self.x_proposal_std = torch.ones(self.gp_dimension)

        #PERFORMANCE
        if self.performance_metric_name == "excess_return":
            self.performance_metric = excess_return_daily
            self.performance_metric_param = {"dataloader":data_loader,
                                             "start":self.start,
                                             "end":self.end,
                                             "steps_ahead":self.steps_ahead}
        #KERNEL
        self.prior_sigma_mean = inference_parameters.get("prior_sigma_mean")
        self.prior_sigma_std = inference_parameters.get("prior_sigma_std")

        self.prior_length_mean = inference_parameters.get("prior_length_mean")
        self.prior_length_std = inference_parameters.get("prior_length_std")

        self.sigma_prior = Normal(torch.full((self.dimension,),self.prior_sigma_mean),
                                  torch.full((self.dimension,),self.prior_sigma_std))

        self.lenght_prior = Normal(torch.full((self.dimension,),self.prior_length_mean),
                                   torch.full((self.dimension,),self.prior_length_std))

        sigma0 = self.sigma_prior.sample()
        lenght0 = self.lenght_prior.sample()

        monte_carlo_values = {"kernel_sigma_s":[sigma0],
                              "kernel_lenght_scales_s":[lenght0],
                              "kernel_sigma_s0":[sigma0],
                              "kernel_lenght_scales_s0": [lenght0]}

        # GAUSSIAN PROCESS
        self.prior_x_mean = inference_parameters.get("prior_x_mean")
        self.prior_x_std = inference_parameters.get("prior_x_std")

        self.x_prior = Normal(torch.full((self.gp_dimension,),self.prior_x_mean),
                              torch.full((self.gp_dimension,),self.prior_x_std))

        monte_carlo_values["LS"] = [[]]*self.dimension
        monte_carlo_values["KS"] = [[]]*self.dimension

        X = self.x_prior.sample()
        monte_carlo_values = self.set_kernel(monte_carlo_values)
        monte_carlo_values["X0"] = [X]
        monte_carlo_values["X"] = [X]

        if self.end is None:
            # in reference policies the returns are taken as reference
            self.portfolio_pmv = data_loader.portfolio_pmv[:, self.start:-1, :].clone()
        else:
            # in reference policies the returns are taken as reference
            self.portfolio_pmv = data_loader.portfolio_pmv[:, self.start:self.end, :].clone()

        self.portfolio_mask = self.portfolio_pmv != self.portfolio_pmv

        return monte_carlo_values

    def set_kernel(self,monte_carlo_values):
        kernel_sigma_s = monte_carlo_values["kernel_sigma_s0"][-1]
        kernel_lenght_scales_s = monte_carlo_values["kernel_lenght_scales_s0"][-1]

        hyperparameter_index = 0
        for kernel_sigma,kernel_lenght_scale in zip(kernel_sigma_s,kernel_lenght_scales_s):
            kernel_parameters = {"kernel_sigma": np.exp(kernel_sigma),
                                 "kernel_lenght_scales": [np.exp(kernel_lenght_scale)]}

            kernel = ScaleKernel(RBFKernel(ard_num_dims=self.dimension, requires_grad=True),
                                 requires_grad=True) + white_noise_kernel()

            kernel_hypers = {"raw_outputscale": torch.tensor(kernel_parameters.get("kernel_sigma")),
                             "base_kernel.raw_lengthscale": torch.tensor(kernel_parameters.get("kernel_lenght_scales"))}

            kernel.kernels[0].initialize(**kernel_hypers)
            kernel_eval = lambda locations: kernel(locations, locations).evaluate().float()

            # initialize gp
            K = kernel_eval(self.grid)
            K_ = K.detach().numpy()
            U = svd(K_)[0]
            D = svd(K_)[1]

            L = np.matmul(U,np.diag(D)**.5)
            monte_carlo_values["LS"][hyperparameter_index] = [L]
            monte_carlo_values["KS"][hyperparameter_index] = [K_]
            hyperparameter_index+= 1

        return monte_carlo_values

    def sample_f(self,monte_carlo_values):
        X = monte_carlo_values["X0"][-1]
        L = monte_carlo_values["LS"][0][-1]
        for i in range(1, self.dimension):
            L = np.kron(L, monte_carlo_values["LS"][i][-1])
        log_f = np.matmul(L,X)
        return log_f

    def log_likelihood_X(self,X,monte_carlo_values):
        monte_carlo_values["X0"] = [X]
        policy = portfolio.portfolio(data_loader,self.start,self.end,self.steps_ahead,monte_carlo_values)
        ER = excess_return_daily(policy, data_loader, start, end, steps_ahead)
        log_likelihood = self.gamma_likelihood.log_prob(ER)
        return log_likelihood.item()

    def gibbs_X(self,monte_carlo_values):
        X0 = monte_carlo_values["X"][-1]
        X = self.x_prior.sample()
        X, ll = elliptical_slice.elliptical_slice(initial_theta=X0,
                                                  prior=X,
                                                  lnpdf=self.log_likelihood_X,
                                                  pdf_params=(monte_carlo_values,))
        monte_carlo_values["X"].append(X)
        return monte_carlo_values

    def log_likelihood_hyperparameters(self,X,monte_carlo_values):
        monte_carlo_values["X0"] = [X]
        policy = portfolio.portfolio(data_loader,self.start,self.end,self.steps_ahead,monte_carlo_values)
        ER = excess_return_daily(policy, data_loader, start, end, steps_ahead)
        log_likelihood = self.gamma_likelihood.log_prob(ER)
        return log_likelihood.item()

    def gibbs_hyperparameters(self,monte_carlo_values):
        sigma = self.sigma_prior.sample()
        lenght = self.lenght_prior.sample()

        monte_carlo_values["kernel_sigma_s0"] = [sigma]
        monte_carlo_values["kernel_lenght_scales_s0"] = [lenght]

        X0 = monte_carlo_values["X"][-1]
        X = self.x_prior.sample()
        X, ll = elliptical_slice.elliptical_slice(initial_theta=X0,
                                                  prior=X,
                                                  lnpdf=self.log_likelihood_X,
                                                  pdf_params=(monte_carlo_values,))
        monte_carlo_values["X"].append(X)

        return monte_carlo_values

    def inference(self,data_loader: ADataLoader, parameters=None, **inference_parameters):
        monte_carlo_values = self.initialize_inference(data_loader,None, **inference_parameters)
        print("#      ---------------- ")
        print("#      Start of MCMC    ")
        print("#      ---------------- ")
        for mcmc_index in range(self.nmc):
            monte_carlo_values = self.gibbs_X(monte_carlo_values)
            monte_carlo_values = self.gibbs_hyperparameters(monte_carlo_values)
            break

    def portfolio(self,data_loader,start=0,end=None,steps_ahead=1,monte_carlo_values=None,train=True):
        if train:
            log_f = self.sample_f(monte_carlo_values)
        else:
            print("Estimation Of f from MCMC not coded")
            raise Exception

        if not train:
            if end is None:
                # in reference policies the returns are taken as reference
                portfolio_pmv = data_loader.portfolio_pmv[:,start:-1,:].clone()
            else:
                # in reference policies the returns are taken as reference
                portfolio_pmv = data_loader.portfolio_pmv[:,start:end,:].clone()

            portfolio_mask = portfolio_pmv != portfolio_pmv
        else:
            portfolio_pmv = self.portfolio_pmv
            portfolio_mask = self.portfolio_mask

        portfolio_pmv[portfolio_pmv != portfolio_pmv] = 0.

        max_ = torch.max(portfolio_pmv, dim=1)
        max_ = max_.values[:, None, :]
        portfolio_pmv = portfolio_pmv / (max_) #include nans when the price is 0
        grid_index = torch.bucketize(portfolio_pmv, self.grid_boundaries)
        grid_dim_index = self.dimension ** np.arange(0, self.dimension, 1)
        grid_index = grid_index * grid_dim_index[None, None, :]
        grid_index = grid_index.sum(axis=-1)
        policy = torch.tensor(log_f[grid_index])
        policy = torch.softmax(policy, axis=0)
        policy[portfolio_mask[:,:,0]] = np.nan

        if steps_ahead > 1:
            policy = policy[:, :-(steps_ahead - 1)]

        return policy

if __name__=="__main__":
    from deep_fields.models.crypto.reference_portfolios import excess_return_daily
    from deep_fields.models.crypto.reference_portfolios import market_portfolio

    #================================================
    # DATA
    #================================================
    steps_ahead = 14

    date_string = "2021-06-14"
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, date_string)

    kwargs = {"path_to_data":data_folder,
              "batch_size": 10,
              "steps_ahead":steps_ahead,
              "date_string": date_string,
              "clean":"interpol",
              "span":"full"}

    data_loader = CryptoDataLoader('cpu', **kwargs)
    data_batch = next(data_loader.train.__iter__())

    #==============================================
    # nonparametric portfolio model
    model_param = NonparametricStochasticPortfolio.get_parameters()
    inference_param = NonparametricStochasticPortfolio.get_inference_parameters()

    portfolio = NonparametricStochasticPortfolio(**model_param,data_loader=data_loader)
    policy, unfolded_series = portfolio(data_batch)
    pprint(inference_param)

    portfolio.inference(data_loader=data_loader, **inference_param)

    # gaussian processes portfolio model
    #model_param = GaussianProcessPortfolio.get_parameters()
    #inference_param = GaussianProcessPortfolio.get_inference_parameters()

    #portfolio = GaussianProcessPortfolio(**model_param)
    #portfolio.inference(data_loader,parameters=None, **inference_param)

    # reference portfolio
    # data_loader.set_portfolio_assets(date="2021-06-14", span="full", predictor=None, top=10)
    # policy = market_portfolio(data_loader, start, end,"price",steps_ahead)
    #excess_return_daily(policy, data_loader, start, end,steps_ahead)


