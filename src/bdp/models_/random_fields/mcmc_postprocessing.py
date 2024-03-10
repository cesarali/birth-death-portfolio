import os
import json
import torch
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def obtain_pandas_from_numeric_metrics(metrics_timeseries):
    """
    returns
    -------
    pandas data frame
    """
    metrics_keys = metrics_timeseries[0].keys()
    numeric_keys = []
    for key_string in metrics_keys:
        if isinstance(metrics_timeseries[0][key_string],(float,int)):
            numeric_keys.append(key_string)
    dataframe_dict = {key_string:[] for key_string in numeric_keys}
    for metrics_entry in metrics_timeseries:
        for key_string in numeric_keys:
            value = metrics_entry[key_string]
            dataframe_dict[key_string].append(value)
    return pd.DataFrame(dataframe_dict)

def obtain_time_series_from_metric_timeseries(metrics_timeseries, metric_key_string):
    """
    Parameters
    ----------

    Returns
    -------
    """
    mcmc_index = []
    metric_values = []
    for metrics_entry in metrics_timeseries:
        value = metrics_entry[metric_key_string]
        index = metrics_entry["montecarlo_index"]
        metric_values.append(value)
        mcmc_index.append(index)
    return mcmc_index, metric_values

def read_results(results_dir_model):
    """
    Read the results as obtained from a results file

    returns
    -------

    parameters,metrics_timeseries,monte_carlo_results
    """
    # INFERENCE RESULTS
    file = open(os.path.join(results_dir_model, "inference_results.json"), "r")
    metrics_timeseries = []
    for line in file:
        result = json.loads(line)
        metrics_timeseries.append(result)

    monte_carlo_results = torch.load(os.path.join(results_dir_model, 'best_model.p'))
    parameters = json.loads(open(os.path.join(results_dir_model, "parameters.json"), "r").readline())
    return parameters, metrics_timeseries, monte_carlo_results

def define_experiments_from_lists(experiments):
    """
    experiments:
        {'jump_arrival_alpha': [0.1, 10.0, 50.0],
         'jump_arrival_beta': [0.1, 10.0, 50.0]}
    """

    # one per realization on the lists
    experiments_set = [{}]
    experiment_index = 0
    for key in experiments.keys():
        experiments_set_0 = []
        for event in experiments_set:
            for key_value in experiments[key]:
                new_event = event.copy()
                new_event[key] = key_value
                experiments_set_0.append(new_event)
        experiments_set = experiments_set_0

    # return with an index
    experiments_set_ = {}
    for experiment_index, experiment in enumerate(experiments_set):
        experiments_set_[experiment_index] = experiment

    return experiments_set_

if __name__=="__main__":
    results_dir = "C:/Users/cesar/Desktop/Projects/General/deep_random_fields/results/merton_jumps_poisson_covariance/"
    results_dir_model = os.path.join(results_dir, "1654344050")
    parameters, metrics_timeseries, monte_carlo_results = read_results(results_dir_model)
    print(obtain_pandas_from_numeric_metrics(metrics_timeseries).head())