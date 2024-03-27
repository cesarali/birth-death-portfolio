import os
import torch
import pickle
import numpy as np
import pandas as pd
from abc import ABC
from pprint import pprint
from dataclasses import dataclass,asdict
from bdp.models.utils.experiment_files_classes import ExperimentFiles
from pathlib import Path
from typing import Dict

from bdp.models.finance.preprocess_data import ml_estimates_of_geometric_brownian_motion_mertonjump
from bdp.models.finance.sde_samplers import simulate_geometric_brownian

def get_portfolio_timeseries():
    return None


class UnivariateMertonPortfolio:
    """
    We expect a dict of pandas timeseries for the price of the assets
    """
    weights:torch.Tensor=None
    total_capital:float = 1.
    number_of_assets:int = None
    number_of_mcmc:int = 100
    number_of_steps:int = 100
    mu_ml:torch.Tensor=None
    sigma_square_ml:torch.Tensor=None
    portfolio_ts:Dict[str,pd.Series] = None

    def __init__(self,portfolio_file_path):
        with open(portfolio_file_path,"rb") as f:
            self.portfolio_ts = pickle.load(f)
        self.set_timeseries()

    def set_weights(self,w):
        w.shape[0] 
        self.weights = w

    def set_timeseries(self):
        initial_prices = []
        real_prices = []
        final_prices = []
        for coin_id,ts in self.portfolio_ts.items():
            prices = ts.values
            valid_prices = prices[np.where(~np.isnan(prices))]
            initial_price = valid_prices[0]
            final_price = valid_prices[-1]
            initial_prices.append(initial_price)
            final_prices.append(final_price)
            real_prices.append(torch.Tensor(prices))
        self.real_prices = torch.vstack(real_prices)
        self.initial_prices = self.real_prices[:,0]
        self.final_prices = self.real_prices[:,-1]

        self.number_of_assets = self.real_prices[0]
        _,mu_ml,sigma_square_ml = ml_estimates_of_geometric_brownian_motion_mertonjump(self.real_prices,
                                                                                       self.initial_prices,
                                                                                       self.final_prices)
        self.mu_ml = mu_ml
        self.sigma_square_ml= sigma_square_ml

    def sample_assets(self)->torch.Tensor:
        repeated_mu = self.mu_ml.repeat_interleave(self.number_of_mcmc)
        repeated_volatility = self.sigma_square_ml.repeat_interleave(self.number_of_mcmc)
        repeated_initial_prices = self.initial_prices.repeat_interleave(self.number_of_mcmc)

        number_of_assets = self.mu_ml.shape[0]
        mcmc_simulation = simulate_geometric_brownian(repeated_initial_prices,repeated_mu,repeated_volatility,self.number_of_steps)
        mcmc_simulation = mcmc_simulation.reshape(number_of_assets,self.number_of_mcmc,self.number_of_steps)
        return mcmc_simulation
    
if __name__=="__main__":
    portfolio_name = "selected_portfolio_1.pck"
    data_dir = r"C:\Users\cesar\Desktop\Projects\BirthDeathPortafolioChoice\Codes\birth-death-portfolio\data\raw\uniswap\2024-03-19"
    data_dir = Path(data_dir)
    portfolio_file_path = data_dir / portfolio_name

    portfolio = UnivariateMertonPortfolio(portfolio_file_path)

    mcmc_simulation = portfolio.sample_assets()
    print(mcmc_simulation.shape)