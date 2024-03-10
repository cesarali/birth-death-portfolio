import torch
import numpy as np

from deep_fields.models.crypto.reference_portfolios import *

def optimal_p(data_loader,p0=-8.,pf=8.,grid_size=1000,start=0,end=None,steps_ahead=1):
    ps = torch.linspace(p0, pf, grid_size).detach().numpy()
    diversity_excess_returns = []
    for p in ps:
        policy = diversity_weighted_portfolio(data_loader, p, start, end)
        ER = excess_return_daily(policy, data_loader, start, end,steps_ahead).item()
        diversity_excess_returns.append(ER)
    diversity_excess_returns = np.array(diversity_excess_returns)

    max_excess = np.max(diversity_excess_returns)
    optimal_p = np.argmax(diversity_excess_returns)
    optimal_p = ps[optimal_p]
    return optimal_p,max_excess

def optimal_window(p0=-8.,pf=8.,grid_size=1000,start=0,end=None):
    start = 0
    end = None
    for end in [500,1000,1500,1700,2000,None]:
        optimal_p,max_excess = optimal_p(-8., 8.,1000, start, end)
        #print(max(diversity_excess_returns))





"""
end = None
size_earnings = []
for portfolio_size in range(10, 200, 10):
    print(portfolio_size)
    crypto_folder = os.path.join(data_path, "raw", "crypto")
    data_folder = os.path.join(crypto_folder, "2021-06-02")
    kwargs = {"path_to_data": data_folder,
              "batch_size": 29,
              "steps_ahead": 10,
              "span": "month"}

    data_loader = CryptoDataLoader('cpu', **kwargs)
    data_loader.set_portfolio_assets(None, portfolio_size)

    ps = torch.linspace(-8., 8., 1000).detach().numpy()
    diversity_excess_returns = []
    for p in ps:
        policy = diversity_weighted_portfolio(data_loader, p, start, end)
        ER = excess_return_daily(policy, data_loader, start, end).item()
        diversity_excess_returns.append(ER)
    diversity_excess_returns = np.array(diversity_excess_returns)
    max_excess = np.max(diversity_excess_returns)
    size_earnings.append(max_excess)
    optimal_p = np.argmax(diversity_excess_returns)
    optimal_p = ps[optimal_p]

    print("Alice")
"""