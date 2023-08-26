### Set Up ###

# global imports
import torch
import torch.nn as nn
from typing import Tuple
from dlearn.models.layers.linear import LinearLayers, LinearParams

### Classes ###

class RecurrentParams():
    # set default variables
    input_size: int=None
    hidden_size: int=None
    num_layers: int=1
    nonlinearity: str='tanh'
    batch_first: bool=True
    dropout: float=0

    def __init__(self, **kwargs):
        # specify allowed member variables
        self.member_variables = {'input_size': int,
                                 'hidden_size': int,
                                 'num_layers': int,
                                 'nonlinearity': str,
                                 'dropout': float
                                 }

        # set member vairables
        self.set_member_variables(**kwargs)

    def set_member_variables(self, **kwargs):
        # iterate over kwargs
        for key, value in kwargs.items():
            # set attr if key is a member variable and value is correct type
            if (key in self.member_variables) and\
                (type(value) == self.member_variables[key]):
                setattr(self, key, value)

class RNN(nn.Module):
    def __init__(self, recurrent_params: RecurrentParams, linear_params: LinearParams):
       # initialize superclass
        super(RNN, self).__init__()

        # get parameters
        input_size = recurrent_params.input_size
        hidden_size = recurrent_params.hidden_size
        num_layers = recurrent_params.num_layers
        nonlinearity = recurrent_params.nonlinearity
        batch_first = recurrent_params.batch_first
        dropout = recurrent_params.dropout

        # create recurrent layers
        self.recurrent_layers = nn.RNN(input_size, hidden_size, 
                                       num_layers=num_layers, 
                                       nonlinearity=nonlinearity, 
                                       batch_first=batch_first, 
                                       dropout=dropout
                                      )
        
        # create linear layers
        linear_params.set_member_variables(input_size=hidden_size)
        self.linear_layers = LinearLayers(linear_params)
                                                                           
    def forward(self,  X: torch.tensor, h: torch.tensor=None) -> Tuple[torch.tensor]:
        # get predictions for recurrent layers
        if h is not None:
            X, h = self.recurrent_layers(X, h)
        else:
            X, h = self.recurrent_layers(X)

        # get predictions for hidden layers   
        X = X[:, 0, :]
        y = self.linear_layers(X)
        return y, h
    