import os
import sys
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bdp.models_.crypto.predictors import CryptoSeq2Seq
from bdp.models_.crypto.portfolio_objectives import excess_return
from bdp.data.crypto.dataloaders import CryptoDataLoader,PortfolioDataLoader
from bdp import data_path

def ml_estimates_black_scholes_from_predictors(output):
    batch_size = output.shape[0]
    seq_lenght = output.shape[1]
    steps_ahead = output.shape[2]
    dimension = output.shape[3]
    prices = output.reshape(batch_size*seq_lenght,steps_ahead,dimension)
    prices = prices[:,:,0]

    X = torch.log(prices)
    log_initial_prices = torch.log(prices[:,0]).detach()
    log_final_prices = torch.log(prices[:,-1]).detach()

    dX = log_final_prices - log_initial_prices
    DX = X[:,1:] - X[:,:-1]
    DX = DX**2.
    DX[DX != DX] = 0.
    DX[DX == np.inf] = 0.
    DX[DX == -np.inf] = 0.
    DX = DX.sum(axis=1)

    sigma_square_ml = DX/steps_ahead - (dX**2)/steps_ahead**2
    mu_ml = DX/steps_ahead + 0.5*sigma_square_ml

    mu_ml = mu_ml.reshape(batch_size,seq_lenght)
    sigma_square_ml = sigma_square_ml.reshape(batch_size,seq_lenght)
    return mu_ml, sigma_square_ml

def birth_and_death_indices(prices_below):
    portfolio_size = prices_below.shape[0]

    where_not_zero = (prices_below != 0.).float()
    column_index = torch.arange(0, prices_below.shape[1], 1).long().unsqueeze(0)
    column_index = column_index.repeat(portfolio_size, 1)
    birth_index = where_not_zero * column_index + (1. - where_not_zero) * (prices_below.shape[1] + 1)
    birth_index = birth_index.long()
    birth_index = birth_index.min(axis=1).values

    death_index = where_not_zero * column_index
    death_index = death_index.long()
    death_index = death_index.max(axis=1).values

    return birth_index, death_index

def equally_weighted_portfolio(prices_below,prices_ahead):
    """
    :param data_loader:
    :return: policy [portfolio_size,number_of_steps,]
    """
    portfolio_size = prices_below.shape[0]
    pi_N = 1. / portfolio_size
    policy = torch.full_like(prices_below, pi_N)
    return policy


if __name__=="__main__":
    #===================================================
    # SELECT DATALOADER
    #===================================================
    date_string = "2021-06-14"
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, date_string)

    kwargs = {"path_to_data":data_folder,
              "batch_size": 29,
              "steps_ahead":14,
              "date_string": date_string,
              "clean":"interpol",
              "span":"full"}

    data_loader = PortfolioDataLoader('cpu', **kwargs)
    train_data_batch = next(data_loader.train.__iter__())

    #===================================================
    # SELECT PREDICTOR
    #===================================================

    average_returns_stats = []
    std_returns_stats = []
    max_returns_stats = []
    prediction_returns_stats = []

    model_dir="C:/Users/cesar/Desktop/Projects/General/deep_random_fields/results/crypto_seq2seq/1637576663/"
    cs2s = CryptoSeq2Seq(model_dir=model_dir)

    for j, train_data_batch in enumerate(data_loader.train):
        print("{0} out of {1}".format(j, data_loader.n_train_batches))

        output_train, unfolded_series_train = cs2s(train_data_batch, use_case="train")

        start = 0
        end = None
        market_type = "price"
        if market_type == "price":
            series_index = 0
        elif market_type == "market_cap":
            series_index = 1
            policy = series / series.sum(axis=0)
        elif market_type == "volume":
            series_index = 2

        series = unfolded_series_train[:, :, 0, series_index]
        prices_below = unfolded_series_train[:, :, 0, 0]
        prices_ahead_prediction = output_train[:, :, -1, 0]
        prices_ahead_real = unfolded_series_train[:, :, -1, 0]

        policy_market = series / series.sum(axis=0)
        policy_ew = equally_weighted_portfolio(prices_below, prices_ahead_prediction)

        er_prediction_ew = excess_return(policy_ew, prices_ahead_prediction, prices_below)
        er_real_ew = excess_return(policy_ew, prices_ahead_real, prices_below)

        er_prediction_market = excess_return(policy_market, prices_ahead_prediction, prices_below)
        er_real_market = excess_return(policy_market, prices_ahead_real, prices_below)

        max_return = er_real_market.max().item()
        av_return = er_real_market.mean().item()
        av_return_prediction = er_prediction_market.mean().item()
        std_return = er_real_market.std().item()

        print(av_return)
        print(std_return)
        print(max_return)

        average_returns_stats.append(av_return)
        std_returns_stats.append(std_return)
        max_returns_stats.append(max_return)
        prediction_returns_stats.append(av_return_prediction)

    print("Av Max")
    print((np.asarray(max_returns_stats).mean()))
    print("Av Mean")
    print(np.asarray(average_returns_stats).mean())
    print("Av Mean Prediction")
    print(np.asarray(prediction_returns_stats).mean())
    print("Av Std")
    print(np.asarray(std_returns_stats).mean())

    stuff = plt.hist(average_returns_stats)
    plt.show()
    stuff = plt.hist(std_returns_stats)
    plt.show()
    stuff = plt.hist(max_return)
    plt.show()
