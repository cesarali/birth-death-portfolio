import os
import sys
import yaml
import torch
from pathlib import Path

from bdp.models.crypto.prediction.prediction_experiment import SummaryPredictionExperiment
from bdp.models.crypto.prediction.trainers import PredictionTrainer
import pytest

from bdp.data.crypto.coingecko.dataloaders import TimeSeriesTorchForTraining
from bdp import config_path

if __name__=="__main__":

    config_file = config_path / "crypto" / "prediction" / "basic_summary_prediction_config.yaml"
    prediction_experiment = SummaryPredictionExperiment(config_path=config_file)
    prediction_trainer = PredictionTrainer(prediction_experiment.config)

    databatch = next(prediction_experiment.dataloader.train().__iter__())
    x,y = prediction_trainer.preprocess_data(databatch)
    
    output = prediction_experiment.prediction_model(x)
    print(output.shape)