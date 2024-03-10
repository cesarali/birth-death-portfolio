import torch
from matplotlib import pyplot as plt
import numpy as np

def plot_simulation_vs_real_mertonjump(prices, S, show=True):
    simulated_returns = (S[:, 1:] - S[:, :-1]) / (S[:, :-1])
    real_returns = (prices[:, 1:] - prices[:, :-1]) / (prices[:, :-1])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax[0].plot(torch.log(S[0, :]).detach().numpy(), "b-", label="simulated")
    ax[0].plot(torch.log(prices[0,:]).detach().numpy(), "r-", label="real")
    ax[0].legend(loc="best")
    ax[0].set_title("Log Asset Price")
    ax[1].plot(simulated_returns[0, :].detach().numpy(), alpha=0.4, label="simulated")
    ax[1].plot(real_returns[0, :].detach().numpy(), alpha=0.4, color="red", label="real")
    ax[1].legend(loc="best")
    ax[1].set_title("Log Returns")
    if show:
        plt.show()

def plot_simulation_vs_real(data_loader,S,show=True):
    simulated_returns = (S[:, 1:] - S[:, :-1]) / (S[:, :-1])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax[0].plot(torch.log(S[0, :]).detach().numpy(), "b-", label="simulated")
    ax[0].plot(torch.log(data_loader.portfolio_pmv[0, :, 0]).detach().numpy(), "r-", label="real")
    ax[0].legend(loc="best")
    ax[0].set_title("Log Asset Price")
    ax[1].plot(simulated_returns[0, :].detach().numpy(), alpha=0.4, label="simulated")
    ax[1].plot(data_loader.portfolio_returns[0, :].detach().numpy(), alpha=0.4, color="red", label="real")
    ax[1].legend(loc="best")
    ax[1].set_title("Log Returns")
    if show:
        plt.show()

def arrivals_indicator_plot(monte_carlo_parameters, data_loader, simulated=True):
    if simulated:
        arrivals_intensity = data_loader["arrivals_intensity"].item()
    else:
        arrivals_intensity = np.asarray(monte_carlo_parameters["arrivals_intensity"]).mean()

    arrivals_indicator = np.vstack(monte_carlo_parameters["arrivals_indicator"])

    montecarlo_steps = arrivals_indicator.shape[0]
    number_of_realizations = float(arrivals_indicator.shape[1])

    number_of_arrivals = arrivals_indicator.sum(axis=1) / number_of_realizations

    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(number_of_arrivals)
    ax.axhline(arrivals_intensity, color="red", linestyle="--")
    plt.show()