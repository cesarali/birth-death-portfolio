import os
import json
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torch.nn.utils.rnn import pack_padded_sequence
from bdp.models.crypto.prediction.past_encoders import LSTMModel
from bdp.data.crypto.coingecko.dataloaders import TimeSeriesDataLoader
from bdp.models.crypto.prediction.prediction_models import SummaryPredictionModel
from bdp.models.crypto.prediction.prediction_experiment import SummaryPredictionExperiment
from bdp.models.crypto.prediction.configuration_classes.prediction_classes import SummaryPredictionConfig
from bdp.models.crypto.prediction.metrics.metrics_utils import log_metrics

import numpy as np
from tqdm import tqdm
from typing import List,Union,Tuple
from abc import ABC,abstractmethod
from dataclasses import dataclass,field,asdict
from bdp.models.crypto.abstract_trainers import Trainer

class PredictionTrainer(Trainer):
    """
    """
    pack_sentences:bool = True

    def __init__(self,config:SummaryPredictionConfig):
        super().__init__()
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device(self.config.TrainingParameters.device)
        else:
            self.device = torch.device("cpu")

    def initialize(self,experiment:SummaryPredictionExperiment):
        """
        Initializes the training process.
        To be implemented by subclasses.
        """        
        model = experiment.prediction_model
        model.to(self.device)

        if isinstance(model.past_encoder,LSTMModel):
            self.pack_sentences = True
        # Loss function and optimizer
        self.criterion = experiment.prediction_model.loss_criterion
        self.optimizer = optim.Adam(model.parameters(), lr=experiment.config.TrainingParameters.learning_rate)
        self.experiment_class = SummaryPredictionExperiment
        

    def preprocess_data(self, databatch):
        """
        Preprocesses the data batch.
        To be implemented by subclasses.

        {0: 'indexes',
         1: 'covariates',
         2: 'past_added_sequences',
         3: 'lengths_past',
         4: 'prediction_summary'}
        """
        past_padded = databatch[2]
        lengths = databatch[3]
        y = databatch[4]
        if self.pack_sentences:
            x = pack_padded_sequence(past_padded, lengths, batch_first=True, enforce_sorted=False)
        x = x.to(self.device)
        y = y.to(self.device)
        return x.float(),y.float()

    def get_model(self):
        pass

    def train_step(self, model, databatch,number_of_training_step,epoch):
        """
        Defines a single training step.
        To be implemented by subclasses.
        """
        self.optimizer.zero_grad()  # Clear gradients for the next training iteration
        x,y = databatch

        output = model(x)  # Forward pass: compute the output
        loss = self.criterion(output, y)  # Compute the loss
        
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters

        if self.config.TrainingParameters.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.TrainingParameters.clip_max_norm)
        
        #if self.do_ema:
        #    self.generative_model.forward_rate.update_ema()

        #if self.config.trainer.lr_decay:
        #    self.scheduler.step()
            
        self.optimizer.step()  # Perform a single optimization step (parameter update)
        self.writer.add_scalar('training loss', loss.item(), number_of_training_step)

        return loss

    def test_step(self, model, databatch, number_of_test_step,epoch):
        """
        Defines a single test step.
        To be implemented by subclasses.
        """
        x,y = databatch
        output = model(x)  # Forward pass: compute the output
        loss = self.criterion(output, y)  # Compute the loss
        self.writer.add_scalar('test loss', loss.item(), number_of_test_step)

        return loss