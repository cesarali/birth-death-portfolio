import numpy as np
import torch


def ml_estimates_of_geometric_brownian_motion_mertonjump(prices,initial_prices,final_prices):
    """
    Assumes data comes from a merton jump process (No births, all survival times are equal)
    :param data_loader:
    :param skip_end:
    :return:
    """
    # we remove last 4 days in order to compensate for errors in the gathering of the data
    # (many points are missing at the end due to download speed)
    portfolio_survival = prices.shape[1]
    prices = prices
    X = torch.log(prices)

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