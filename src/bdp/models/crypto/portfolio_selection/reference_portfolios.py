import os
import torch
import pandas as pd
import numpy as np

def equally_weighted_portfolio(data_loader,start=0,end=None,steps_ahead=1):
    """
    :param data_loader:
    :return: policy [portfolio_size,number_of_steps,]
    """
    portfolio_returns = data_loader.portfolio_returns
    pi_N = 1. / data_loader.portfolio_size
    policy = torch.full_like(portfolio_returns[:,start:end], pi_N)
    if steps_ahead > 1:
        policy = policy[:,:-(steps_ahead-1)]
    return policy

def market_portfolio(data_loader,start=0,end=None,market_type="price",steps_ahead=1):
    portfolio_pmv = data_loader.portfolio_pmv
    if market_type == "price":
        series = portfolio_pmv[:, start:end, 0][:,:-1]
        policy = series/series.sum(axis=0)
    elif market_type == "market_cap":
        series = portfolio_pmv[:, start:end, 1][:,:-1]
        policy = series/series.sum(axis=0)
    elif market_type == "volume":
        series = portfolio_pmv[:, start:end, 2][:,:-1]
        policy = series/series.sum(axis=0)
    if steps_ahead > 1:
        policy = policy[:,:-(steps_ahead-1)]
    return policy

def diversity_weighted_portfolio(data_loader,p=0.,start=0,end=None,market_type="price",steps_ahead=1):
    """
    Equation 14 of Stochastic Portfolio Theory a Machine Learning Perspective
    """
    mu = market_portfolio(data_loader,start,end,market_type)
    mu_p = torch.pow(mu, p)
    policy = torch.softmax(mu_p, axis=0)
    if steps_ahead > 1:
        policy = policy[:,:-(steps_ahead-1)]
    return policy

def excess_return_daily(policy_evaluated, data_loader, start=0, end=None, steps_ahead=1):
    portfolio_pmv = data_loader.portfolio_pmv[:,start:end,:]
    prices = portfolio_pmv[:, :-steps_ahead, 0]

    unfolded_portfolio_pmv = portfolio_pmv.unfold(dimension=1,
                                                  size=steps_ahead + 1,
                                                  step=1).contiguous()
    unfolded_prices = unfolded_portfolio_pmv[:, :, 0, :]
    prices_ahead = unfolded_prices[:, :, -1]
    portfolio_returns = (prices_ahead - prices) / prices

    assert policy_evaluated.shape == portfolio_returns.shape

    policy_evaluated = policy_evaluated * portfolio_returns
    policy_evaluated[policy_evaluated != policy_evaluated] = 0.

    excess_return = 1. + policy_evaluated.sum(axis=0)
    excess_return = excess_return[torch.arange(0,excess_return.shape[0],steps_ahead).long()]
    excess_return_ = torch.prod(excess_return)

    return excess_return_

if __name__=="__main__":
    from deep_fields import data_path
    from matplotlib import pyplot as plt
    from deep_fields.data.crypto.dataloaders import CryptoDataLoader
    from deep_fields.models.crypto.grid_search_references import optimal_p

    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, "2021-06-02")
    kwargs = {"path_to_data": data_folder,
              "batch_size": 29,
              "steps_ahead": 10,
              "span": "month"}

    data_loader = CryptoDataLoader('cpu', **kwargs)
    data_loader.set_portfolio_assets(date="2021-06-14",
                                     span="full",
                                     predictor=None,
                                     top=10,
                                     date0=None,
                                     datef=None,
                                     max_size=10)

    steps_ahead = 1
    start = 0
    end = None

    policy = equally_weighted_portfolio(data_loader, start, end,steps_ahead)
    print(excess_return_daily(policy, data_loader, start, end,steps_ahead))

    policy = market_portfolio(data_loader, start, end,"price",steps_ahead)
    print(excess_return_daily(policy, data_loader, start, end,steps_ahead))

    policy = diversity_weighted_portfolio(data_loader,0.,start,end,"price",steps_ahead)
    print(excess_return_daily(policy, data_loader, start, end,steps_ahead))

    """
    optimal_p_,max_excess = optimal_p(data_loader,-8.,8.,1000,0,None,48)
    print(optimal_p_)
    print(max_excess)
    ME = []
    for hours_ahead in range(12,24*7,6):
        optimal_p_,max_excess = optimal_p(data_loader,-8.,8.,1000,0,None,hours_ahead)
        ME.append(max_excess)
        print(hours_ahead)

    plt.plot( range(12,24*7,6),ME,"o-")
    plt.show()
    #policy = diversity_weighted_portfolio(data_loader, optimal_p, -2, -1)
    #ids = data_loader.portfolio_ids
    #policy = policy.view(-1).detach().numpy()

    #print(dict(zip(ids,policy)))
    #print(optimal_p)
    #print(max_excess)
    """