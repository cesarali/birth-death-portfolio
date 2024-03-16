import os
import sys
import yaml
import torch
from pathlib import Path

from bdp.models.crypto.prediction.prediction_experiment import SummaryPredictionExperiment

from bdp.data.crypto.coingecko.dataloaders import (
    TimeSeriesTorchForTraining
)

from bdp.models.crypto.prediction.trainers import PredictionTrainer

if __name__=="__main__":
    from bdp import config_path
    from bdp.models.crypto.prediction.prediction_experiment import SummaryPredictionExperiment

    config_file = config_path / "crypto" / "prediction" / "basic_summary_prediction_config.yaml"
    prediction_experiment = SummaryPredictionExperiment(config_path=config_file)
    trainer = PredictionTrainer(prediction_experiment.config)
    trainer.train(prediction_experiment)
    