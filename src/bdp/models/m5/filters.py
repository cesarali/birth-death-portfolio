import torch
from tqdm import tqdm
from torch.optim import Adam
from deep_fields.utils.loss_utils import kl_gaussian_diagonal

import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from torch import nn
from deep_fields.models.abstract_models import DeepBayesianModel
from deep_fields.models.basic_utils import set_debugging_scheme, generate_training_message


class M5_neural_volatility(DeepBayesianModel, nn.Module):
    """
    Here we follow:  https://arxiv.org/pdf/1605.06432.pdf

    {"wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap",
    "price"}
    """

    def __init__(self, data_loader, **kwargs):
        DeepBayesianModel.__init__(self, "neural_volatility", **kwargs)
        nn.Module.__init__(self)
        self.device = torch.device("cpu")

        self.observable_dim = kwargs.get("observable_dim")
        self.observable_embedding_dim = kwargs.get("observable_embedding_dim")
        self.hidden_observable_transition_dim = kwargs.get("hidden_observable_transition_dim")

        self.hidden_state_dim = kwargs.get("hidden_state_dim")
        self.hidden_state_transition_dim = kwargs.get("hidden_state_transition_dim")
        self.control_variable_dim = kwargs.get("control_variable_dim")
        self.number_of_steps = kwargs.get("number_of_steps")

        # ONLY NEEDED FOR
        self.F = kwargs.get("transition_matrix")
        self.H = kwargs.get("emission_matrix")
        self.Q = kwargs.get("transition_noise")
        self.R = kwargs.get("emission_noise")

        self.define_deep_models()

    def define_deep_models(self):
        # self.topics_vocab_size = self.number_of_topics + 2 # we assume that only pads and eos is used
        # self.topic_embeddings = nn.Embedding(self.topics_vocab_size, self.embedding_dimension)
        # self.topic_lstm = nn.LSTM(self.embedding_dimension, self.hidden_state_size)
        # self.topic_decoder = nn.Linear(self.hidden_state_size,self.topics_vocab_size)
        # self.criterion = nn.CrossEntropyLoss()

        # EMBEDDINGS

        self.observable_embedding = nn.Sequential(nn.Linear(self.observable_dim, self.observable_embedding_dim))

        # RECOGNITION MODEL
        self.bidirectional_lstm = nn.LSTM(self.observable_embedding_dim,
                                          self.hidden_state_transition_dim,
                                          bidirectional=True)
        self.recognition_lstm = nn.LSTM(self.observable_embedding_dim + self.hidden_state_transition_dim * 2,
                                        self.hidden_state_transition_dim)
        self.recognition_mean = nn.Linear(self.hidden_state_transition_dim,
                                          self.hidden_state_dim)
        self.recognition_logvar = nn.Linear(self.hidden_state_transition_dim,
                                            self.hidden_state_dim)

        # PRIOR
        self.prior_lstm = nn.LSTM(self.hidden_state_dim,
                                  self.hidden_state_transition_dim)
        self.prior_mean = nn.Linear(self.hidden_state_transition_dim,
                                    self.hidden_state_dim)
        self.prior_logvar = nn.Linear(self.hidden_state_transition_dim,
                                      self.hidden_state_dim)

        # LIKELIHOOD MODEL
        self.likelihood_lstm = nn.LSTM(self.observable_embedding_dim + self.hidden_state_dim,
                                       self.hidden_observable_transition_dim)
        self.likelihood_mean = nn.Linear(self.hidden_observable_transition_dim,
                                         self.observable_dim)
        self.likelihood_logvar = nn.Linear(self.hidden_observable_transition_dim,
                                           self.observable_dim)

    @classmethod
    def get_parameters(cls):
        observable_size = 2
        observable_embedding_dim = 2
        hidden_observable_dim = 2
        hidden_state_size_transition = 3
        hidden_state_size = 5
        control_variable_size = 4

        h = 5
        M = 4
        T = 25

        # topic dynamics as a state space model from simulation
        F = torch.tensor(np.random.random((hidden_state_size, hidden_state_size))).type(torch.float32)
        B = torch.tensor(np.random.random((hidden_state_size, control_variable_size))).type(torch.float32)
        H = torch.tensor(np.random.random((observable_size, hidden_state_size))).type(torch.float32)

        Q = torch.Tensor(np.diag(np.repeat(3., hidden_state_size)))
        R = torch.Tensor(np.diag(np.repeat(3., observable_size)))

        parameters_sample = {"observable_dim": observable_size,  # x
                             "observable_embedding_dim": observable_embedding_dim,
                             "hidden_observable_transition_dim": hidden_observable_dim,
                             "hidden_state_dim": hidden_state_size,  # z
                             "hidden_state_transition_dim": hidden_state_size_transition,  # h (neural volatility)
                             "number_of_steps": T,
                             "transition_matrix": F,
                             "control_matrix": B,
                             "emission_matrix": H,
                             "transition_noise": Q,
                             "emission_noise": R,
                             "model_path": "C:/Users/cesar/Desktop/Projects/DeepDynamicTopicModels/Results/"}

        return parameters_sample

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
        # STATE SPACE TRANSITION --------------------------------------------------------------------------------
        transition_noise_distribution = MultivariateNormal(torch.zeros(self.hidden_state_dim), self.Q)
        emission_noise_distribution = MultivariateNormal(torch.zeros(self.observable_dim), self.R)

        z_0 = MultivariateNormal(torch.zeros(self.hidden_state_dim), self.Q).sample()

        dynamics = []
        for t in range(self.number_of_steps):
            # transition
            z_t = torch.matmul(self.F, z_0)
            w_t = transition_noise_distribution.sample()

            z_t = z_t + w_t
            # emission
            x_t = torch.matmul(self.H, z_t)
            y_t = emission_noise_distribution.sample()
            z_0 = z_t
            dynamics.append(x_t.unsqueeze(0))
        dynamics = torch.cat(dynamics, dim=0)

        return dynamics

    def metrics(self):
        return None

    def loss(self, target, likelihood_parameters, prior_parameters, recognition_parameters):
        likelihood_mean, likelihood_sigma = likelihood_parameters
        prior_mean, prior_sigma = prior_parameters
        recognition_mean, recognition_sigma = recognition_parameters

        batch_size, target_size = target.size()
        likelihood = Normal(likelihood_mean, likelihood_sigma)
        log_probability = likelihood.log_prob(target)
        KL = kl_gaussian_diagonal(prior_mean, prior_sigma, recognition_mean, recognition_sigma)
        loss = log_probability.sum() + KL.sum()
        loss = loss.sum()

        return None

    def forward(self):
        return None

    def inference_step(self, optimizer, observables, epoch, **inference_parameters):
        # observables batch_size,sequence_lenght,observable_dim
        BPTT_CHUNKS = torch.chunk(observables, 3, dim=1)

        batch_size, sequence_lenght, observables_dim = observables.shape

        # embeddings
        embedded_observable = self.observable_embedding(observables.view(batch_size * sequence_lenght, -1))
        embedded_observable = embedded_observable.view(batch_size, sequence_lenght, -1)

        # we define recognition for all times
        embedded_observable = embedded_observable.transpose(1, 0)  # sequence_lenght, batch_size, observable_embedded
        observables = observables.transpose(1, 0)

        hidden_likelihood, hidden_prior, hidden_bi, hidden_recognition = self.inititalize_hidden_rnn_state(batch_size)
        output_bi, hidden_bi = self.bidirectional_lstm(embedded_observable,
                                                       hidden_bi)
        recognition_input = torch.cat([embedded_observable, output_bi], dim=2)
        # prior first sample
        for t in range(self.number_of_steps - 1):
            recognition_step_input = recognition_input[t].unsqueeze(0)
            output_forward, hidden_forward = self.recognition_lstm(recognition_step_input,
                                                                   hidden_recognition)
            # output_forward # 1, batch_size, hidden_state_transition_dim
            output_forward = output_forward.squeeze()
            # reparametrization
            z_mean = self.recognition_mean(output_forward)
            z_sigma = torch.exp(.5 * self.recognition_logvar(output_forward))
            epsilon = torch.randn(z_mean.shape).to(self.device)

            z = (z_mean + epsilon * z_sigma).unsqueeze(0)  # batch_size

            # calculate prior
            prior_output, hidden_prior = self.prior_lstm(z, hidden_prior)
            prior_mean = self.prior_mean(prior_output)
            prior_std = torch.exp(.5 * self.prior_logvar(prior_output))

            # calculate nll
            likelihood_input = torch.cat([embedded_observable[t].unsqueeze(0), z], dim=2)

            # observables #batch_size, sequence_lenght, observable_dim
            likelihood_ouput, hidden_likelihood = self.likelihood_lstm(likelihood_input, hidden_likelihood)
            likelihood_mean = self.likelihood_mean(likelihood_ouput.squeeze())
            likelihood_std = torch.exp(.5 * self.likelihood_logvar(likelihood_ouput.squeeze()))

            likelihood_parameters = (likelihood_mean, likelihood_std)
            prior_parameters = (prior_mean, prior_std)
            recognition_parameters = (z_mean, z_sigma)

            self.loss(observables[t + 1], likelihood_parameters, prior_parameters, recognition_parameters)
            break

    def inference(self, data_loader, **inference_parameters):
        generate_training_message()
        learning_rate = inference_parameters.get("learning_rate", None)
        number_of_epochs = inference_parameters.get("number_of_epochs", 10)
        self.number_of_iterations = 0
        optimizer = Adam(self.parameters(), lr=learning_rate)
        for epoch in tqdm(range(1, number_of_epochs + 1)):
            self.inference_step(optimizer, data_loader, epoch, **inference_parameters)

    def validation_step(self):
        return None

    def initialize_inference(self):
        return None

    def inititalize_hidden_rnn_state(self, batch_size):
        hidden_recognition = (torch.randn(1, batch_size, self.hidden_state_transition_dim),
                              torch.randn(1, batch_size, self.hidden_state_transition_dim))

        hidden_bi = (torch.randn(2, batch_size, self.hidden_state_transition_dim),
                     torch.randn(2, batch_size, self.hidden_state_transition_dim))

        hidden_likelihood = (torch.randn(1, batch_size, self.hidden_observable_transition_dim),
                             torch.randn(1, batch_size, self.hidden_observable_transition_dim))

        hidden_prior = (torch.randn(1, batch_size, self.hidden_state_transition_dim),
                        torch.randn(1, batch_size, self.hidden_state_transition_dim))

        return hidden_likelihood, hidden_prior, hidden_bi, hidden_recognition
