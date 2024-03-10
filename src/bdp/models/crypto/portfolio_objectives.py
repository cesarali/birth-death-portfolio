import torch
import numpy as np
from matplotlib import pyplot as plt


def excess_return(policy, prices_ahead, prices_below):
    assert prices_ahead.shape == prices_below.shape
    assert policy.shape == prices_ahead.shape

    portfolio_returns = (prices_ahead - prices_below) / prices_below
    policy_evaluated = portfolio_returns * policy
    prices_zero = prices_below == 0.

    policy_evaluated[torch.where(policy_evaluated != policy_evaluated)] = 0.
    policy_evaluated[torch.where(prices_zero)] = 0.

    excess_return = 1. + policy_evaluated.sum(axis=0)
    return excess_return