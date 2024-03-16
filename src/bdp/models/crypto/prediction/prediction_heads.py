import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_

class MLPRegressionHead(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim,**kwargs):
        super(MLPRegressionHead, self).__init__()
        # Create MLP layers
        self.layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layer = nn.Linear(input_dim, hidden_dim)
            else:
                layer = nn.Linear(hidden_dims[i-1], hidden_dim)
            # Apply Xavier initialization to the layer weights
            xavier_uniform_(layer.weight)
            self.layers.append(layer)
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        xavier_uniform_(self.output_layer.weight)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

