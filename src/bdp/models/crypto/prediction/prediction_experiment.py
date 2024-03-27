import yaml
import json
import torch
from pathlib import Path
from dataclasses import asdict

from bdp.models.crypto.prediction.prediction_models import (
    SummaryPredictionModel
)

from bdp.data.crypto.coingecko.dataloaders import (
    TimeSeriesDataLoader
)

from bdp.utils.config_file_operations import dynamic_load_config_from_yaml
from bdp.models.utils.experiment_files_classes import ExperimentFiles
from bdp.models.crypto.prediction.configuration_classes.prediction_classes import SummaryPredictionConfig,load_dataclass_from_dict

class SummaryPredictionExperiment:
    """
    Contains all the elements for an experiment
    """
    prediction_model:SummaryPredictionModel = None
    dataloader:TimeSeriesDataLoader = None
    config: SummaryPredictionConfig = None
    experiment_files: ExperimentFiles = None

    def __init__(self,config:SummaryPredictionConfig=None,config_path:Path|str=None,experiment_dir:Path|str=None):
        if experiment_dir is not None:
            self.get_from_experiment(experiment_dir)
        else:
            if config is not None:
                self.get_from_config(config)
            elif config_path is not None:
                config = dynamic_load_config_from_yaml(config_path)
                self.get_from_config(config)
            else:
                print("No Config or Experiment")

    def get_from_config(self,config:SummaryPredictionConfig):
        """
        reads yaml file
        """
        if isinstance(config,SummaryPredictionConfig):
            self.config = config
        else:
            raise Exception("Wrong Config Class for Summary Prediction Experiment")
        
        self.experiment_files = ExperimentFiles(
            experiment_name=self.config.ExperimentMetaData.experiment_name,
            experiment_type=self.config.ExperimentMetaData.experiment_type,
            experiment_indentifier=self.config.ExperimentMetaData.experiment_indentifier
        )
        self.prediction_model = SummaryPredictionModel(self.config)
        self.dataloader = TimeSeriesDataLoader(self.config)
        
    def get_from_experiment(self,experiment_dir:Path|str=None):

        """
        reads results in folder
        """
        self.experiment_files = ExperimentFiles(
            experiment_dir=experiment_dir
        )
        with open(self.experiment_files.config_path, 'r') as file:
            config_dict = json.load(file)

        self.config = load_dataclass_from_dict(SummaryPredictionConfig,config_dict)

        results = self.experiment_files.load_results()
        self.prediction_model = results["model"]
        self.dataloader = TimeSeriesDataLoader(self.config)