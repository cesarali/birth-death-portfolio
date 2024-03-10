import os
import torch
import numpy as np
from deep_fields import project_path
from pprint import pprint
from scipy.interpolate import interp1d

from deep_fields.data.crypto.dataloaders import CryptoDataLoader, ADataLoader
from deep_fields.models.random_fields.sde_utils import  simulate_geometric_brownian
from deep_fields.models.random_fields.plots import plot_simulation_vs_real, plot_simulation_vs_real_mertonjump

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def select_prices_in_biggest_window(portfolio_pmv, portfolio_birth_indexes, portfolio_death_index):
    """
    Here we select the biggest window of time possible where all assets are already born
    until the minimum asset death

    Intended for (Merton Jump Process No Birth)
    """
    birth_index = portfolio_birth_indexes.max()
    last_index = portfolio_death_index.min()

    all_prices = portfolio_pmv[:, birth_index:last_index, 0]
    initial_prices = all_prices[:, 0]
    final_prices = all_prices[:, -1]

    assert torch.where(initial_prices != initial_prices)[0].sum() == 0.
    assert torch.where(final_prices != final_prices)[0].sum() == 0.

    assert torch.where(initial_prices == 0.)[0].sum() == 0.
    assert torch.where(final_prices == 0.)[0].sum() == 0.

    return all_prices, initial_prices, final_prices

def interpolate_and_complete_a_tensor(price, interpolate_value=0., kind="linear"):
    """
    Performs simple interpolation, by defining the tensor indexing as the input (support)
    it assumes the places to be interpolated are the zeros of the vector
    """
    number_of_realizations = price.shape[-1]
    indexes_full = np.asarray(list(range(0, number_of_realizations)))
    indexes_not_zero = indexes_full[np.where(price != interpolate_value)[0]]
    price_to_interpolate = torch.clone(price[np.where(price != interpolate_value)[0]])

    f2 = interp1d(indexes_not_zero, price_to_interpolate, kind='linear')
    min_index_not_nan = min(indexes_not_zero)
    max_index_not_nan = max(indexes_not_zero)

    indexes_full_ = indexes_full[np.where(indexes_full >= min_index_not_nan)]
    indexes_full_ = indexes_full_[np.where(indexes_full_ <= max_index_not_nan)]
    price_corrected = f2(indexes_full_)
    price[indexes_full_] = torch.Tensor(price_corrected)

    assert torch.where(price == 0.)[0].sum() == 0., "Interpolation Failed in Assets Time Series"

    return price

def preprocess_cryptoportfolio_for_mertonjump(crypto_data_loader):
    """
    Select biggest non nan block and interpolates

    :returns
    corrected_prices, initial_prices, final_prices
    """
    all_prices, initial_prices, final_prices = select_prices_in_biggest_window(crypto_data_loader.portfolio_pmv,
                                                                               crypto_data_loader.portfolio_birth_indexes,
                                                                               crypto_data_loader.portfolio_death_index)
    corrected_prices = []
    number_of_realizations = all_prices.shape[-1]
    number_of_assets = all_prices.shape[0]
    for coin_index in range(number_of_assets):
        price = all_prices[coin_index, :]
        corrected_price = interpolate_and_complete_a_tensor(price, interpolate_value=0., kind="linear")
        corrected_prices.append(corrected_price)
    return torch.stack(corrected_prices), initial_prices, final_prices

def ml_estimates_of_geometric_brownian_motion_mertonjump(corrected_prices,initial_prices,final_prices):
    """
    Assumes data comes from a merton jump process (No births, all survival times are equal)
    :param data_loader:
    :param skip_end:
    :return:
    """
    # we remove last 4 days in order to compensate for errors in the gathering of the data
    # (many points are missing at the end due to download speed)
    portfolio_survival = corrected_prices.shape[1]
    prices = corrected_prices
    X = torch.log(corrected_prices)

    log_initial_prices = torch.log(initial_prices)
    log_final_prices = torch.log(final_prices)

    dX = log_final_prices - log_initial_prices
    DX = X[:, 1:] - X[:, :-1]
    DX = DX ** 2.
    DX[DX != DX] = 0.
    DX[DX == np.inf] = 0.
    DX[DX == -np.inf] = 0.

    sigma_square_ml = DX.sum(axis=1) / portfolio_survival - (dX ** 2) / portfolio_survival ** 2
    mu_ml = DX.sum(axis=1) / portfolio_survival + 0.5 * sigma_square_ml

    return DX,sigma_square_ml,mu_ml

def ml_estimates_of_geometric_brownian_motion(data_loader,skip_end=10):
    # we remove last 4 days in order to compensate for errors in the gathering of the data
    # (many points are missing at the end due to download speed)
    portfolio_survival = data_loader.portfolio_survival - skip_end
    prices = data_loader.portfolio_pmv[:,:,0]
    X = torch.log(prices)

    log_initial_prices = torch.log(data_loader.portfolio_initial_prices)
    log_final_prices = torch.log(data_loader.portfolio_final_prices)

    dX = log_final_prices - log_initial_prices
    DX = X[:,1:] - X[:,:-1]
    DX = DX**2.
    DX[DX != DX] = 0.
    DX[DX == np.inf] = 0.
    DX[DX == -np.inf] = 0.

    sigma_square_ml = DX.sum(axis=1)/portfolio_survival - (dX**2)/portfolio_survival**2
    mu_ml = DX.sum(axis=1)/portfolio_survival + 0.5*sigma_square_ml

    return DX,sigma_square_ml,mu_ml

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
        number_of_steps = corrected_prices.shape[1]
        S = simulate_geometric_brownian(initial_prices, mu_ml, sigma_square_ml, number_of_steps)
        plot_simulation_vs_real_mertonjump(corrected_prices, S, show=True)

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
              "model_path": os.path.join(project_path, 'results')}

    return kwargs,data_loader

if __name__=="__main__":
    from deep_fields import data_path
    from datetime import datetime
    date_string = "2021-06-14"
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, date_string)

    kwargs = {"path_to_data":data_folder,
              "batch_size": 29,
              "steps_ahead":14,
              "date_string": date_string,
              "clean":"interpol",
              "span":"full"}

    date0 = datetime(2018, 1, 1)
    datef = datetime(2019, 1, 1)

    crypto_data_loader = CryptoDataLoader('cpu', **kwargs)
    crypto_data_loader.set_portfolio_assets("2021-06-14",
                                            "full",
                                            predictor=None,
                                            top=4,
                                            date0=None,
                                            datef=None,
                                            max_size=4)

    data_batch = next(crypto_data_loader.train.__iter__())
    corrected_prices_data = preprocess_cryptoportfolio_for_mertonjump(crypto_data_loader)
    kwargs,data_loader = estimate_mcmc_dataloader_and_parameters_mertonjump(corrected_prices_data)
    #pprint(kwargs)