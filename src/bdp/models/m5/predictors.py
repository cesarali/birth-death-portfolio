import os
import sys
import torch
import numpy as np
from torch import nn

from tqdm import tqdm
from torch.optim import Adam

from deep_fields.models.basic_utils import set_debugging_scheme, generate_training_message

from deep_fields.data.m5.dataloaders import covariates_info
from deep_fields.models.m5.embeddings import create_embeddings
from deep_fields.models.abstract_models import DeepBayesianModel
from deep_fields.models.deep_architectures.tcn import TemporalConvNet

from deep_fields.data.m5.dataloaders import M5_BLOCK
from torch.utils.data import DataLoader

from deep_fields.models.deep_architectures.deep_nets import MLP
from torch.distributions import Poisson, Normal
from deep_fields.models.m5 import embeddings

non_sequential_covariates, sequential, basic_covariates_final_maps, _ = covariates_info()


# ==================================================

class M5_RNN(DeepBayesianModel):
    """
    """

    def __init__(self, data_loader=None, **kwargs):
        DeepBayesianModel.__init__(self, "M5_RNN", **kwargs)
        device = kwargs.get("device", "cpu")
        self.device = torch.device(device)

        # prediction
        self.rnn_param = kwargs.get("rnn")
        # encoder
        # decoder

        # set up
        self.define_deep_models()
        self.init_parameters()

    @classmethod
    def get_parameters(cls):
        parameters = {"input_dimension": 1,
                      "rnn": {"name": "lstm",
                              "args": {"hidden_dim": 10},
                              "decoder": {"name": "basic", "args": {"dim": None}},
                              "output_dimension": 1,
                              "device": "cpu",
                              "model_path": "C:/Users/cesar/Desktop/Projects/DeepDynamicTopicModels/Results/"}}
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

    def define_deep_models(self):
        self.hidden_state_size = self.rnn_param.get("args").get("hidden_dim")
        self.count_embedding = nn.Linear(1., self.count_embedding_dimension)
        self.counts_rnn = nn.LSTM(self.count_embedding_dimension, self.hidden_state_size)
        self.covariates_embeddings = create_embeddings()

    def sample(self):
        print("Not a generative model!")
        return None

    def loss(self):
        return None

    def forward(self, databatch):
        return None

    def inference_step(self, optimizer, data_loader, epoch, **inference_parameters):
        batch_size = inference_parameters.get("batch_size")

        # for batch in dataloader.train_and_validation():
        databatch = data_loader  # no dataloader loop (yet)

        optimizer.zero_grad()

        # encoder decoder

        training_loss = self.loss()

        training_loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #    logits, hidden = self(validation_input, hidden)
        #    validation_loss = self.loss(logits, validation_ouput)

        metrics = {"training_loss": training_loss.item()}

        self.update_writer(metrics)
        self.number_of_iterations += 1

    def inference(self, data_loader, **inference_parameters):
        generate_training_message()
        learning_rate = inference_parameters.get("learning_rate", None)
        number_of_epochs = inference_parameters.get("number_of_epochs", 10)
        self.Train()
        self.number_of_iterations = 0
        optimizer = Adam(self.parameters(), lr=learning_rate)
        for epoch in tqdm(range(1, number_of_epochs + 1)):
            self.inference_step(optimizer, data_loader, epoch, **inference_parameters)

    def validation_step(self):
        return None

    def initialize_inference(self, data_loader, **inference_parameters):
        return None

    def inititalize_hidden_rnn_state(self, batch_size):
        hidden = (torch.randn(1, batch_size, self.hidden_state_size),
                  torch.randn(1, batch_size, self.hidden_state_size))
        return hidden

    def inititalize_parameters(self):
        print("Init Param Not Implemented")
        return None


# -----------------------------------------------------------------------------------------------------------------------
# TCN

class M5_TCN(DeepBayesianModel):
    """
    # Sequential stuff

    ['count','price','day','wday','month','year',
     'event_name_1','event_type_1','event_name_2',
     'event_type_2','snap']

     #Non sequential
     ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    """

    def __init__(self, model_dir=None, data_loader=None, **kwargs):
        nn.Module.__init__(self)
        DeepBayesianModel.__init__(self, "m5_tcn", model_dir=model_dir, data_loader=data_loader, **kwargs)

    def define_deep_models(self):
        self.embeddings_dict = embeddings.create_embeddings({**self.sequential_covariates_parameters,
                                                             **self.non_sequential_covariates_parameters})

        self.TCN = TemporalConvNet(2, self.num_channels,
                                   self.kernel_size,
                                   dropout=self.dropout)

        self.encode_covariates = MLP(input_dim=self.total_embeddings + self.time_encoding,
                                     layers_dim=self.covariates_layers,
                                     output_dim=self.covariates_encoding,
                                     ouput_transformation=None,
                                     dropout=self.dropout)

        self.decoder_lstm = nn.LSTM(self.time_encoding + self.covariates_encoding + 1,
                                    self.decoder_hidden_state,
                                    dropout=self.dropout)

        self.decode_output = nn.Linear(self.decoder_hidden_state, 1)

    @classmethod
    def get_parameters(cls):
        sequential_covariates_parameters = {"price": {"embedding": 8}, 'day': {"embedding": 8}, 'wday': {"embedding": 8},
                                            'month': {"embedding": 8}, 'year': {"embedding": 8},
                                            'event_name_1': {"embedding": 8}, 'event_type_1': {"embedding": 8},
                                            'event_name_2': {"embedding": 8}, 'event_type_2': {"embedding": 8},
                                            'snap': {"embedding": 8}}

        non_sequential_covariates_parameters = {'item_id': {"embedding": 8},
                                                'dept_id': {"embedding": 8},
                                                'cat_id': {"embedding": 8},
                                                'store_id': {"embedding": 8},
                                                'state_id': {"embedding": 8}}

        parameters = {"sequential_covariates_parameters": sequential_covariates_parameters,
                      "non_sequential_covariates_parameters": non_sequential_covariates_parameters,
                      "covariates_layers": [100, 100, 100],
                      "dropout": .4,
                      "decoder_rnn": "lstm",
                      "covariates_encoding": 20,
                      "kernel_size": 3,  # TCN values
                      "number_of_levels": 10,
                      "time_encoding": 20,
                      "decoder_hidden_state": 20,
                      "model_path": "C:/Users/cesar/Desktop/Projects/GeneralResults/"}

        return parameters

    def set_parameters(self, **kwargs):
        self.sequential_covariates_parameters = kwargs.get("sequential_covariates_parameters")
        self.non_sequential_covariates_parameters = kwargs.get("non_sequential_covariates_parameters")
        self.covariates_layers = kwargs.get("covariates_layers")
        self.covariates_encoding = kwargs.get("covariates_encoding")
        self.time_encoding = kwargs.get("time_encoding")
        self.decoder_hidden_state = kwargs.get("decoder_hidden_state")
        self.dropout = kwargs.get("dropout")

        self.number_of_levels = kwargs.get("number_of_levels")
        self.kernel_size = kwargs.get("kernel_size")
        self.num_channels = [self.time_encoding] * self.number_of_levels
        self.receptive_field = 2 ** len(self.num_channels)

        self.total_embeddings = 0
        for k, v in self.sequential_covariates_parameters.items():
            self.total_embeddings += v["embedding"]
        for k, v in self.non_sequential_covariates_parameters.items():
            self.total_embeddings += v["embedding"]
        self.parameters = kwargs

    def update_parameters(self, dataloader, **kwargs):
        # kwargs.get("v_dynamic_recognition_parameters").update({"observable_dim": dataloader.vocabulary_dim})
        # json.dump(kwargs, open(self.parameter_path, "w"))
        return None

    def loss(self, databatch, forward_results, dataloader, epoch):
        return {"loss": 0}

    def metrics(self, databatch, forward_results, epoch, mode="evaluation", dataloader=None):
        return {}

    def forward(self, databatch):
        return None

    def init_parameters(self):
        return None

    def inititalize_hidden_rnn_state(self, batch_size):
        hidden_decoder = (torch.randn(1, batch_size, self.decoder_hidden_state),
                          torch.randn(1, batch_size, self.decoder_hidden_state))
        #
        hidden_decoder = (hidden_decoder[0].to(self.device),
                          hidden_decoder[1].to(self.device))

        return hidden_decoder


if __name__ == "__main__":
    final_dir = "C:/Users/cesar/Desktop/Projects/NeuralProcessesUncertainty/data/preprocessed/"
    # dataset = M5_BLOCK(data_dir=final_dir)
    # dataloader = DataLoader(dataset, batch_size=4,shuffle=True)
    m5_predictor = M5_TCN()

# -----------------------------------------------------------------------------------------------------------------------
# TRAFORMERS XL
# https://arxiv.org/pdf/1901.02860.pdf]
