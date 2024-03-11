import numpy as np
import torch
from deep_fields.models.abstract_models import DeepBayesianModel
from deep_fields.models.basic_utils import generate_training_message
from deep_fields.models.gaussian_processes.gaussian_processes import multivariate_normal, white_noise_kernel
from gpytorch.kernels import RBFKernel, ScaleKernel
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.optim import Adam
from tqdm import tqdm


class heterocedastic_gaussian_process(DeepBayesianModel, nn.Module):
    """
    Here we follow:
    Variational Heteroscedastic Gaussian Process Regression
    """

    def __init__(self, data_loader=None, **kwargs):
        DeepBayesianModel.__init__(self, "HGP", **kwargs)
        nn.Module.__init__(self)
        # file handling
        self.number_of_inducing_points = kwargs.get("number_of_inducing_points")
        self.define_deep_models(self.number_of_inducing_points)

    @classmethod
    def get_parameters(cls):
        parameters = {"number_of_inducing_points": 10,
                      "model_path": "C:/Users/cesar/Desktop/Projects/DeepDynamicTopicModels/Results/"}
        return parameters

    @classmethod
    def get_inference_parameters(cls):
        inference_parametes = {"number_of_epochs": 100,
                               "burning_time": 10,
                               "bbt": 3,
                               "train_percentage": .8,
                               "batch_size": 4,
                               "learning_rate": .001,
                               "debug": False}
        return inference_parametes

    def sample(self):
        """
        returns
        -------
        (y,f,g)
        """
        train_input = torch.tensor(np.random.uniform(0., 10., self.number_of_inducing_points))
        f_variance = self.f_kernel(train_input, train_input).evaluate().float()
        f_mean = torch.zeros((1, self.number_of_inducing_points))  # [number_of_batches,mean_dimension]
        f_distribution = multivariate_normal(f_mean, f_variance)
        f = f_distribution.rsample().float()

        g_variance = self.g_kernel(train_input, train_input).evaluate().float()
        g_mean = torch.zeros((1, self.number_of_inducing_points))  # [number_of_batches,mean_dimension]
        g_distribution = multivariate_normal(g_mean, g_variance)
        g = g_distribution.rsample().float()
        r = torch.exp(g)

        noise_distribution = Normal(torch.zeros_like(r), r)
        epsilon = noise_distribution.sample()
        y = f + epsilon

        return (train_input, y, f, g)

    def forward(self):
        return None

    def evaluate_mv_normal_approximation(self, g_prior_distribution):
        K_gg = g_prior_distribution.covariance_matrix
        L1 = torch.diag((self.Lambda - .5 * torch.ones_like(self.Lambda)))
        mu = torch.matmul(torch.matmul(K_gg, L1), torch.ones_like(self.Lambda))

        Sigma_inverse = g_prior_distribution.varinv() + torch.diag(self.Lambda)
        Sigma = multivariate_normal(torch.zeros_like(mu), Sigma_inverse).varinv()
        return mu, Sigma

    def evaluate_covariances(self, train_input):
        K_gg = self.g_kernel(train_input, train_input).evaluate().float()
        K_ff = self.f_kernel(train_input, train_input).evaluate().float()
        g_prior_distribution = multivariate_normal(torch.ones_like(train_input), K_gg)
        return g_prior_distribution, K_ff

    def initialize_inference(self, data_loader, **inference_parameters):
        train_input, train_ouput = data_loader
        g_prior_distribution, K_ff = self.evaluate_covariances(train_input)
        mu, Sigma = self.evaluate_mv_normal_approximation(g_prior_distribution)

        inference_variables = {"g_prior": g_prior_distribution, "K_ff": K_ff, "mu": mu, "Sigma": Sigma}
        return inference_variables

    def loss(self, databatch, Likelihood, Posterior, Prior):
        input, output = databatch
        KL = torch.distributions.kl.kl_divergence(Posterior, Prior)
        Trace = .25 * torch.trace(Likelihood.covariance_matrix)
        likelihood = Likelihood.log_prob(output)
        loss = likelihood - Trace - KL
        return {"loss": loss, "nll": likelihood, "Trace": Trace, "KL": KL}

    def inference_step(self, databatch, inference_variables, **inference_parameters):
        input, output = databatch
        g_prior = inference_variables["g_prior"]
        mu = inference_variables["mu"]
        Sigma = inference_variables["Sigma"]
        K_ff = inference_variables["K_ff"]

        R = torch.exp(mu - .5 * torch.diagonal(Sigma))
        Likelihood = MultivariateNormal(torch.zeros_like(mu), K_ff + torch.diag(R))
        Posterior = MultivariateNormal(mu.double(), Sigma.double())
        Prior = MultivariateNormal(g_prior.par_1.double(), g_prior.covariance_matrix.double())
        LOSS = self.loss(databatch, Likelihood, Posterior, Prior)

        return None

    def inference(self, databatch, **inference_parameters):
        generate_training_message()
        learning_rate = inference_parameters.get("learning_rate", None)
        number_of_epochs = inference_parameters.get("number_of_epochs", 10)
        inference_variables = self.initialize_inference(databatch, **inference_parameters)

        self.number_of_iterations = 0
        optimizer = Adam(self.parameters(), lr=learning_rate)
        for epoch in tqdm(range(1, number_of_epochs + 1)):
            self.inference_step(databatch, inference_variables, **inference_parameters)
            break

    def define_deep_models(self, number_of_data_points):
        windows_size = 1
        self.f_kernel = ScaleKernel(RBFKernel(ard_num_dims=windows_size, requires_grad=True),
                                    requires_grad=True) + white_noise_kernel()
        self.g_kernel = ScaleKernel(RBFKernel(ard_num_dims=windows_size, requires_grad=True),
                                    requires_grad=True) + white_noise_kernel()
        self.Lambda = nn.Parameter(torch.ones(number_of_data_points))

        f_hypers = {"raw_outputscale": torch.tensor(1.),
                    "base_kernel.raw_lengthscale": torch.tensor(np.repeat(1., windows_size))}

        g_hypers = {"raw_outputscale": torch.tensor(1.),
                    "base_kernel.raw_lengthscale": torch.tensor(np.repeat(1., windows_size))}

        self.f_kernel.kernels[0].initialize(**f_hypers)
        self.g_kernel.kernels[0].initialize(**g_hypers)

    def init_parameters(self):
        # self.topic_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.topic_decoder.bias.data.fill_(0)
        return None
