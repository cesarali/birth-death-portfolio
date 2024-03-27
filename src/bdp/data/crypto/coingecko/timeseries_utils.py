from datetime import datetime
import numpy as np

# https://parsiad.ca/blog/2020/maximum_likelihood_estimation_of_geometric_brownian_motion_parameters/

def geometric_brownian_motion_ml_estimate(price,minimum_number_of_samples=50):
    """
    Obtained from 

    https://parsiad.ca/blog/2020/maximum_likelihood_estimation_of_geometric_brownian_motion_parameters/

    HANDLES NANs
    parameters
    ----------
    price: np.array of prices 

    return
    ------
    dirft, vol
    """
    log_price = np.log(price)  # X
    last_log = log_price[np.where(~np.isnan(log_price))][-1]
    start_log = log_price[np.where(~np.isnan(log_price))][0]
    delta = log_price[1:] - log_price[:-1]  # ΔX
    delta = delta[np.where(~np.isnan(delta))]

    n_samples = delta.size  # N

    if n_samples > minimum_number_of_samples:
        n_years = 1.0  # δt

        total_change = last_log - start_log  # δX

        vol2 = (-(total_change**2) / n_samples + np.sum(delta**2)) / n_years
        # Equivalent but slower: `vol2 = np.var(delta) * delta.size / n_years`
        vol = np.sqrt(vol2)
        drift = total_change / n_years + 0.5 * vol2

        return drift, vol
    else:
        return np.nan,np.nan