import torch
from torch.distributions import Normal 

def simulate_geometric_brownian(initial_prices,mu_ml,sigma_square_ml,number_of_steps):
    """
    we generate the equation

    \begin{equation}
        S_t = S_0 \exp\left\{(\mu-\frac{\sigma^2}{2})t + \sigma W_t\right\}
    \end{equation}

    :param initial_prices:
    :param mu_ml:
    :param sigma_square_ml:
    :param number_of_steps:
    :return:
    """
    number_of_assets = mu_ml.shape[0]
    sigma_ml = torch.sqrt(sigma_square_ml)
    dW = Normal(0, 1.).sample(sample_shape=(number_of_assets, number_of_steps))
    W = dW.cumsum(axis=1)
    t = torch.arange(0, number_of_steps)[None, :].repeat(number_of_assets, 1)

    S = initial_prices[:, None] * torch.exp((mu_ml - sigma_square_ml * .5)[:, None] * t + sigma_ml[:, None] * W)
    return S