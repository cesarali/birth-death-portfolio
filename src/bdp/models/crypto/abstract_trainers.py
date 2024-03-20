import json
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from bdp.data.crypto.coingecko.dataloaders import TimeSeriesDataLoader
from bdp.models.crypto.prediction.prediction_models import SummaryPredictionModel
from bdp.models.crypto.prediction.prediction_experiment import SummaryPredictionExperiment
from bdp.models.crypto.prediction.configuration_classes.prediction_classes import SummaryPredictionConfig
from bdp.models.crypto.prediction.metrics.metrics_utils import log_metrics
from bdp.models.crypto.prediction.configuration_classes.experiment_classes import ExperimentFiles

import numpy as np
from typing import List,Union,Tuple
from abc import ABC,abstractmethod
from dataclasses import dataclass,field,asdict
from tqdm import tqdm

@dataclass
class TrainerState:
    experiment: Union[SummaryPredictionExperiment]
    best_loss : float = np.inf

    average_train_loss : float = 0.
    average_test_loss : float = 0.

    test_loss: List[float] = field(default_factory=lambda:[])
    train_loss: List[float] = field(default_factory=lambda:[])

    number_of_test_step:int = 0
    number_of_training_steps:int = 0

    def set_average_test_loss(self):
        if len(self.test_loss) > 0:
            self.average_test_loss = np.asarray(self.test_loss).mean()

    def set_average_train_loss(self):
        self.average_train_loss = np.asarray(self.train_loss).mean()

    def finish_epoch(self):
        self.test_loss = []
        self.train_loss = []

    def update_training_batch(self,loss):
        self.train_loss.append(loss)
        self.number_of_training_steps += 1

    def update_test_batch(self,loss):
        self.number_of_test_step += 1
        self.test_loss.append(loss)

# Assuming CTDDConfig, CTDD, and other necessary classes are defined elsewhere
class Trainer(ABC):
    """
    This trainer is intended to obtain a backward process of a markov jump via
    a ratio estimator with a stein estimator

    """
    config:Union[SummaryPredictionConfig] = None

    def parameters_info(self,experiment:SummaryPredictionExperiment):
        print("# ==================================================")
        print("# START OF TRAINING ")
        print("# ==================================================")

        print("# Current Model ************************************")

        print(experiment.experiment_files.experiment_type)
        print(experiment.experiment_files.experiment_name)
        print(experiment.experiment_files.experiment_indentifier)

        print("# ==================================================")
        print("# Number of Epochs {0}".format(experiment.config.TrainingParameters.num_epochs))
        print("# ==================================================")

    @abstractmethod
    def initialize(self,experiment):
        """
        Initializes the training process.
        To be implemented by subclasses.
        """
        pass

    def initialize_(self,experiment:SummaryPredictionExperiment):
        self.initialize(experiment)
        self.parameters_info(experiment)
        self.writer = SummaryWriter(experiment.experiment_files.tensorboard_path)
        self.tqdm_object = tqdm(range(self.config.TrainingParameters.num_epochs))
        self.best_metric = np.inf
        self.number_of_epochs = experiment.config.TrainingParameters.num_epochs
        
    @abstractmethod
    def train_step(self, current_model, databatch, number_of_training_step,epoch):
        """
        Defines a single training step.
        To be implemented by subclasses.
        """
        pass

    def global_training(self,training_state,all_metrics,epoch):
        return {},all_metrics

    @abstractmethod
    def test_step(self,current_model, databatch, number_of_test_step,epoch):
        """
        Defines a single test step.
        To be implemented by subclasses.
        """
        pass

    def global_test(self,training_state,all_metrics,epoch):
        return {},all_metrics

    @abstractmethod
    def preprocess_data(self, databatch):
        """
        Preprocesses the data batch.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_model(self):
        pass

    def train(self,experiment:SummaryPredictionExperiment):
        """
        FORWARD  means sampling from p_0 (data) -> p_1 (target)

        :return:
        """
        model = experiment.prediction_model
        dataloader = experiment.dataloader

        #CREATE FOLDER REQUIRED TO STORE EVERYTHING
        experiment.experiment_files.create_directories()
        with open(experiment.experiment_files.config_path, 'w') as file:
            json.dump(asdict(experiment.config), file, indent=4)

        # INITIATE LOSS
        self.initialize_(experiment)
        all_metrics = {}
        results_ = {}
        self.saved = False

        training_state = TrainerState(experiment)
        for epoch in self.tqdm_object:
            #TRAINING
            for step, databatch in enumerate(dataloader.train()):
                databatch = self.preprocess_data(databatch)
                # DATA
                loss = self.train_step(model,databatch,training_state.number_of_training_steps,epoch)
                loss_ = loss.item() if isinstance(loss, torch.Tensor) else loss
                training_state.update_training_batch(loss_)
                self.tqdm_object.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
                self.tqdm_object.refresh()  # to show immediately the update
                if self.config.TrainingParameters.debug:
                    break
            training_state.set_average_train_loss()
            results_,all_metrics = self.global_training(training_state,all_metrics,epoch)

            #EVALUATES VALIDATION LOSS
            if not self.config.TrainingParameters.save_model_metrics_stopping:
                for step, databatch in enumerate(dataloader.test()):
                    databatch = self.preprocess_data(databatch)
                    # DATA
                    loss = self.test_step(model,databatch,training_state.number_of_test_step,epoch)
                    loss_ = loss.item() if isinstance(loss, torch.Tensor) else loss
                    training_state.update_test_batch(loss_)
                    if self.config.TrainingParameters.debug:
                        break
            # EVALUATES METRICS TO STOP
            else:
                if epoch > self.config.TrainingParameters.save_model_metrics_warming:
                    all_metrics = log_metrics(model, all_metrics=all_metrics, epoch="best",writer=self.writer)

            training_state.set_average_test_loss()
            results_,all_metrics = self.global_test(training_state,all_metrics,epoch)

            # STORING MODEL CHECKPOINTS
            if (epoch + 1) % self.config.TrainingParameters.save_model_epochs == 0:
                results_ = self.save_results(training_state,epoch+1,checkpoint=True)

            # SAVE RESULTS IF LOSS DECREASES IN VALIDATION NOT BY IMPROVED METRICS
            if not self.config.TrainingParameters.save_model_metrics_stopping:
                current_average = training_state.average_test_loss if self.config.TrainingParameters.save_model_test_stopping else training_state.average_train_loss
                if current_average < training_state.best_loss:
                    if self.config.TrainingParameters.warm_up_best_model_epoch < epoch or epoch == self.number_of_epochs - 1:
                        results_ = self.save_results(training_state,epoch + 1,checkpoint=False)
                    training_state.best_loss = training_state.average_test_loss

            #SAVE RESULTS IF IT INCREASES METRICS
            else:
                if epoch > self.config.TrainingParameters.save_model_metrics_warming:
                    if all_metrics[self.config.TrainingParameters.metric_to_save] < self.best_metric:
                        results_ = self.save_results(training_state, epoch + 1, checkpoint=False)
                        self.best_metric = all_metrics[self.config.TrainingParameters.metric_to_save]
            training_state.finish_epoch()
        
        #=====================================================
        # BEST MODEL IS READ AND METRICS ARE STORED
        #=====================================================
        #experiment_dir = experiment.experiment_files.experiment_dir
        #if self.saved:
        #    self.model = self.generative_model_class(experiment_dir=experiment_dir)
        #all_metrics = log_metrics(self.model, all_metrics=all_metrics, epoch="best", writer=self.writer)
        #self.writer.close()

        return results_,all_metrics

    def save_results(self,
                     training_state:TrainerState,
                     epoch:int,
                     checkpoint:bool=True):
        experiment_files = training_state.experiment.experiment_files
        RESULTS = {
            "model": training_state.experiment.prediction_model,
            "best_loss": training_state.best_loss,
            "training_loss":training_state.average_train_loss,
            "test_loss":training_state.average_test_loss,
        }
        if checkpoint:
            best_model_path_checkpoint = experiment_files.best_model_path_checkpoint.format(epoch)
            torch.save(RESULTS,best_model_path_checkpoint)
        else:
            torch.save(RESULTS, experiment_files.best_model_path)
        self.saved = True
        return RESULTS

