### Set Up ###

# global imports
import torch
import torch.nn as nn
from dlearn.params.linear import LinearParams


### Classes ###

class LinearLayers(nn.Module):
    def __init__(self, linear_params: LinearParams):
        # initialize superclass
        super(LinearLayers, self).__init__()

        # get parameters
        dropout, hidden_sizes, input_size, output_size, probabilities =\
            linear_params.get_params()

        # initialize linear values
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        n_layers = len(layer_sizes) - 1

        # create linear layers
        for i in range(n_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # add activation function
            if i < n_layers:
                layers.append(nn.ReLU())

            # add dropout regularization
            if (dropout > 0) and i < n_layers:
                layers.append(nn.Dropout(p=dropout))

        # add softmax to normalize for probabilities
        if probabilities == True:
            layers.append(nn.Softmax())

        # create neural net object
        self.linear_layers = nn.Sequential(*layers)

    def forward(self, X: torch.tensor) -> torch.tensor:
        y = self.linear_layers(X)
        return y
    