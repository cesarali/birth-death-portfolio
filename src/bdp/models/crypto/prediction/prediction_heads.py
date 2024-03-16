import torch
from torch import nn
from torch.nn import functional as F

class MLPRegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,**kwargs):
        super(MLPRegressionHead, self).__init__()
        # Create MLP layers
        self.layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dim))
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

