import torch

def polya_gamma_mean(b,c):
    return (b/(2*c))*torch.tanh(c/2)
