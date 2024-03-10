import torch
from torch import matmul as m


def wasserstein_gaussians(mu_a,sigma_a,mu_b,sigma_b):
    sigma_a_sq = torch.pow(sigma_a,.5)
    B = m(sigma_a_sq,m(sigma_b,sigma_a_sq))
    B = sigma_a + sigma_b + 2.*pow(B,.5)
    B = torch.einsum("kii",B) #trace
    norm_2 = torch.norm(mu_a-mu_b,dim=1)**2
    W = norm_2 + B
    return W