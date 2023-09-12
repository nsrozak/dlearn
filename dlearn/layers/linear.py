### Set Up ###

# global imports
import torch
import torch.nn as nn
from dlearn.utils import Params

### Classes ###

class LinearParams(Params):
    # set default variables
    input_size: int=None
    output_size: int=None
    hidden_sizes: list=[]
    probabilities: bool=False
    dropout: float=0

    def __init__(self, **kwargs):
        # initialize super class
        super(LinearParams, self).__init__()

        # specify allowed member variables
        self.member_variables = {'input_size': int, 
                                 'output_size': int, 
                                 'hidden_sizes': list, 
                                 'probabilities': bool, 
                                 'dropout': float
                                 }

        # set member vairables
        self.set_member_variables(**kwargs)

class LinearLayers(nn.Module):
    def __init__(self, linear_params: LinearParams):
        # initialize superclass
        super(LinearLayers, self).__init__()

        # get parameters
        dropout, hidden_sizes, input_size, output_size, probabilities =\
            linear_params.get_params()

        # initialize linear values
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.linear_layers = nn.ModuleList()
        n_layers = len(layer_sizes) - 1

        # create linear layers
        for i in range(n_layers):
            self.linear_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # add activation function
            if i < n_layers:
                self.linear_layers.append(nn.ReLU())

            # add dropout regularization
            if (dropout > 0) and i < n_layers:
                self.linear_layers.append(nn.Dropout(p=dropout))

        # add softmax to normalize for probabilities
        if probabilities == True:
            self.linear_layers.append(nn.Softmax())

    def forward(self, X: torch.tensor) -> torch.tensor:
        y = self.linear_layers(X)
        return y
    