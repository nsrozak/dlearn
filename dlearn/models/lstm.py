### Set Up ###

# global imports
import torch
import torch.nn as nn
from typing import Tuple
from dlearn.models.rnn import RecurrentParams
from dlearn.layers.linear import LinearLayers, LinearParams

### Classes ###

class LSTM(nn.Module):
    def __init__(self, recurrent_params: RecurrentParams, linear_params: LinearParams):
       # initialize superclass
        super(LSTM, self).__init__()

        # get parameters
        input_size = recurrent_params.input_size
        hidden_size = recurrent_params.hidden_size
        num_layers = recurrent_params.num_layers
        batch_first = recurrent_params.batch_first
        dropout = recurrent_params.dropout

        # create recurrent layers
        self.recurrent_layers = nn.LSTM(input_size, hidden_size, 
                                        num_layers=num_layers,
                                        batch_first=batch_first, 
                                        dropout=dropout
                                       )
        
        # create linear layers
        linear_params.set_member_variables(input_size=hidden_size)
        self.linear_layers = LinearLayers(linear_params)
                                                                           
    def forward(self,  X: torch.tensor, h: Tuple[torch.tensor]=None) -> Tuple:
        # get predictions for recurrent layers
        if h is not None:
            X, h = self.recurrent_layers(X, h)
        else:
            X, h = self.recurrent_layers(X)

        # get predictions for hidden layers   
        X = X[:, 0, :]
        y = self.linear_layers(X)
        return y, h
    