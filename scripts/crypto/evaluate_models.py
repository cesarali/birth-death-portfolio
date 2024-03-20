import os
import sys
import yaml
import torch
from pathlib import Path

from bdp.models.crypto.prediction.prediction_experiment import SummaryPredictionExperiment
from bdp.models.crypto.prediction.prediction_models import SummaryPredictionModel

from bdp.data.crypto.coingecko.dataloaders import (
    TimeSeriesTorchForTraining
)

from bdp.models.crypto.prediction.trainers import PredictionTrainer

if __name__=="__main__":
    from bdp import config_path
    from bdp.models.crypto.prediction.prediction_experiment import SummaryPredictionExperiment

    #experiment_dir = r"C:\Users\cesar\Desktop\Projects\BirthDeathPortafolioChoice\Codes\birth-death-portfolio\results\SummaryPrediction\Basic\1710931925"
    experiment_dir = r"C:\Users\cesar\Desktop\Projects\BirthDeathPortafolioChoice\Codes\birth-death-portfolio\results\SummaryPrediction\Basic\1710958468"
    prediction_experiment = SummaryPredictionExperiment(experiment_dir=experiment_dir)
    print(prediction_experiment.prediction_model)


    