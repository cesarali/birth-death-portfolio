import os
from bdp import results_path
from bdp.models.finance.preprocess_data import ml_estimates_of_geometric_brownian_motion_mertonjump

def estimate_mcmc_dataloader_and_parameters_mertonjump(corrected_prices_data,show_plot=True):
    """
    Here we use maximum likelihood approximations of univariate geometric brownian motion

    as well as mcmc of univariate geometric with jumps to obtain close estimates for the multivariate
    monte carlo

    :param crypto_data_loader: this data loader was designed for conditional neural sdes estimates for
                               stochastic portfolio

    :return: data_loaders
    """
    # expected returns
    corrected_prices,initial_prices,final_prices = corrected_prices_data
    DX, sigma_square_ml, mu_ml = ml_estimates_of_geometric_brownian_motion_mertonjump(corrected_prices,initial_prices,final_prices)

    if show_plot:
        #number_of_steps = corrected_prices.shape[1]
        #S = simulate_geometric_brownian(initial_prices, mu_ml, sigma_square_ml, number_of_steps)
        #plot_simulation_vs_real_mertonjump(corrected_prices, S, show=True)
        pass

    number_of_processes = sigma_square_ml.shape[0]
    number_of_realizations = corrected_prices.shape[1]

    # KERNEL PARAMETERS ESTIMATES
    print("Sigma")
    print(sigma_square_ml.std())

    returns_mean_a = mu_ml.mean().item()
    returns_mean_b = (mu_ml.std()/sigma_square_ml.mean()).item()
    kernel_sigma = sigma_square_ml.mean().item()
    kernel_lenght_scales = [1., 2.]

    locations_dimension = 2
    kernel_parameters = {"kernel_sigma": kernel_sigma,
                         "kernel_lenght_scales": kernel_lenght_scales}

    data_loader = {"arrivals_intensity": None,
                   "arrivals_indicator": None,
                   "jump_mean": None,
                   "jump_covariance": None,
                   "jump_size": None,
                   "diffusive_log_returns": DX.T,
                   "log_returns": DX.T,
                   "locations": None,
                   "kernel_sigma": kernel_sigma,
                   "kernel_lenght_scales": kernel_lenght_scales,
                   "diffusion_covariance": None,
                   "expected_returns": mu_ml,
                   "kernel": None}

    kwargs = {"locations_dimension": locations_dimension,
              "jump_size_scale_prior": 1.,
              "jump_size_a": 0.5,
              "jump_size_b": 1.,
              "jump_arrival_alpha": 1.,
              "jump_arrival_beta": 1.,
              "returns_mean_a": returns_mean_a,
              "returns_mean_b": returns_mean_b,
              "kernel_parameters": kernel_parameters,
              "number_of_processes": number_of_processes,
              "number_of_realizations": number_of_realizations,
              "model_path": results_path}

    return kwargs,data_loader
