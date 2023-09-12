### Set Up ###

# global imports
import torch
import torch.nn as nn
from typing import Tuple
from dlearn.utils import Params
from dlearn.layers.linear import LinearLayers, LinearParams

### Classes ###

class RecurrentParams(Params):
    # set default variables
    input_size: int=None
    hidden_size: int=None
    num_layers: int=1
    nonlinearity: str='tanh'
    batch_first: bool=True
    dropout: float=0

    def __init__(self, **kwargs):
        # initialize super class
        super(RecurrentParams, self).__init__()

        # specify allowed member variables
        self.member_variables = {'input_size': int,
                                 'hidden_size': int,
                                 'num_layers': int,
                                 'nonlinearity': str,
                                 'dropout': float
                                 }

        # set member vairables
        self.set_member_variables(**kwargs)


class RNN(nn.Module):
    def __init__(self, recurrent_params: RecurrentParams, linear_params: LinearParams):
       # initialize superclass
        super(RNN, self).__init__()

        # get parameters
        batch_first, dropout, input_size, hidden_size, nonlinearity, num_layers =\
            recurrent_params.get_params()

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
    