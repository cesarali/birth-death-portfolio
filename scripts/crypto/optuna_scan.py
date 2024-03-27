
from bdp.models.crypto.prediction.optuna_scans import SummaryPredictionScanOptuna
from optuna.visualization import (
    plot_optimization_history, 
    plot_slice, 
    plot_contour, 
    plot_parallel_coordinate, 
    plot_param_importances
)


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

