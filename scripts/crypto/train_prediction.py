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
    from bdp.utils.config_file_operations import dynamic_load_config_from_yaml

    config_file = config_path / "crypto" / "prediction" / "basic_summary_prediction_config.yaml"
    config = dynamic_load_config_from_yaml(config_file)
    config.TrainingParameters.num_epochs = 3

    prediction_experiment = SummaryPredictionExperiment(config=config)
    trainer = PredictionTrainer(prediction_experiment.config)
    results_,all_metrics = trainer.train(prediction_experiment)
    print(all_metrics)
    