import torch

def symmetric_function(x, x_plus, x_minus, k):
    """
    """
    x_minus = x_minus[torch.where(x_minus >= 0)[0]]
    where_big = torch.where(x > 0)

    # Upper Values
    f_sym_plus = torch.sqrt(k) * torch.sqrt(x_plus)

    # Lower values
    f_sym_minus = torch.sqrt(k) * torch.sqrt(x[where_big].float())

    return f_sym_plus, f_sym_minus