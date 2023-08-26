### Set Up ###

# global imports
import torch
import torch.nn as nn
from dlearn.layers.linear import LinearLayers, LinearParams


### Classes ###

class MLP(nn.Module):
    def __init__(self, linear_params: LinearParams):
        self.linear_layers = LinearLayers(linear_params)

    def forward(self, X: torch.tensor) -> torch.tensor:
        y = self.linear_layers(X)
        return y