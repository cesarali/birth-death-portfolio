import torch
from pathlib import Path

from bdp.models.crypto.prediction.prediction_experiment import SummaryPredictionExperiment

from bdp.data.crypto.coingecko.dataloaders import (
    TimeSeriesTorchForTraining
)

from bdp.models.crypto.prediction.trainers import PredictionTrainer

import os
import optuna
import numpy as np
import datetime
from bdp import results_path

from optuna.visualization import (
    plot_optimization_history, 
    plot_slice, 
    plot_contour, 
    plot_parallel_coordinate, 
    plot_param_importances
)

from bdp.utils.config_file_operations import dynamic_load_config_from_yaml

def single_run_prediction(config,
                          epochs,
                          batch_size,
                          learning_rate,
                          hidden_dim,
                          num_layers,
                          dropout):
    
    config.DataLoaderParameters.batch_size = batch_size
    config.TrainingParameters.num_epochs = epochs
    config.TrainingParameters.learning_rate = learning_rate

    config.PredictionModel.PastEncoder.hidden_dim = hidden_dim
    config.PredictionModel.PastEncoder.layer_num = num_layers

    prediction_experiment = SummaryPredictionExperiment(config_path=config_file)
    trainer = PredictionTrainer(prediction_experiment.config)
    results_,all_metrics = trainer.train(prediction_experiment)
    test_loss_value = all_metrics['test_loss_value']
    return test_loss_value


class SummaryPredictionScanOptuna:

    def __init__(self, 
                 basic_config_file,
                 device="cuda:0",
                 n_trials=100,
                 epochs=500,
                 batch_size=(5, 50),
                 learning_rate=(1e-5, 1e-2), 
                 hidden_dim=(50,500),
                 num_layers=2, 
                 dropout=None):
        
        #...scan experiments
        self.basic_config_file = basic_config_file
        self.config = dynamic_load_config_from_yaml(self.basic_config_file)
        if not (self.config.ExperimentMetaData.results_dir is None):
            results_path = Path(self.config.ExperimentMetaData.results_dir)

        self.workdir = results_path / self.config.ExperimentMetaData.experiment_name / self.config.ExperimentMetaData.experiment_type
        self.device = device
    
        #...params        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        #...scan
        self.iteration, self.metric = 0, np.inf
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective, n_trials=n_trials)

        print(self.study.best_params)
        

    def def_param(self, trial, name, param, type):
        if type == "int":
            return trial.suggest_int(name, param[0], param[1]) if isinstance(param, tuple) else param
        if type == "cat":
            return trial.suggest_categorical(name, param) if isinstance(param, list) else param
        if type == "float":
            if not isinstance(param, tuple): return param
            elif param[0] / param[1] > 0.05: return trial.suggest_float(name, param[0], param[1])
            else: return trial.suggest_float(name, param[0], param[1], log=True)
       

    def objective(self, trial):
        self.iteration += 1
        #exp_id = self.experiment_indentifier + "_" + str(self.iteration)

        #...scaning params:
        epochs = self.def_param(trial, 'epochs', self.epochs, type="int")
        batch_size = self.def_param(trial, 'bach_size', self.batch_size, type="int")
        learning_rate = self.def_param(trial, 'lr', self.learning_rate, type="float")

        hidden_dim = self.def_param(trial, 'dim_hid', self.hidden_dim, type="int") if self.hidden_dim is not None else None
        num_layers = self.def_param(trial, 'num_layers', self.num_layers, type="int") if self.num_layers is not None else None
        dropout = self.def_param(trial, 'dropout', self.dropout, type="float") if self.dropout is not None else None

        #...run single experiment:
        metric = single_run_prediction(self.config,
                                       epochs,
                                       batch_size,
                                       learning_rate,
                                       hidden_dim,
                                       num_layers,
                                       dropout)

        # mse_histograms = metrics["mse_marginal_histograms"]
        #if self.nist_metric < self.metric: self.metric = self.nist_metric
        #else: os.system("rm -rf {}/{}".format(self.workdir, exp_id))
        return metric


if __name__ == "__main__":
   ############################
    #  SUMMARY PREDICTION
    ############################
    from bdp import config_path

    config_file = config_path / "crypto" / "prediction" / "basic_summary_prediction_config_optuna.yaml"

    scan = SummaryPredictionScanOptuna(basic_config_file=config_file,
                                       device="cuda:0",
                                       n_trials=3,
                                       epochs=3,
                                       batch_size=64,
                                       learning_rate=(1e-6, 1e-2), 
                                       hidden_dim=10,
                                       num_layers=1)

    df = scan.study.trials_dataframe()
    df.to_csv(scan.workdir / 'trials.tsv', sep='\t', index=False)

    # Save Optimization History
    fig = plot_optimization_history(scan.study)
    fig.write_image(scan.workdir / "optimization_history.png")

