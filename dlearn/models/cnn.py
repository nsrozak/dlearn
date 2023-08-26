### Set Up ###

# global imports
import torch
import torch.nn as nn
from typing import Tuple
from dlearn.models.layers.linear import LinearLayers, LinearParams
from dlearn.models.layers.conv import ConvLayers, ConvParams, conv_output_size


### Classes ###

class CNN(nn.Module):
    def __init__(self, input_dims: Tuple[int], conv_params: ConvParams, linear_params: LinearParams):
        # initialize superclass
        super(CNN, self).__init__()

        # create convolution layers
        self.conv_layers = ConvLayers(conv_params)

        # get parameters
        channels = conv_params.channels
        kernel_sizes = conv_params.kernel_sizes
        strides = conv_params.strides
        paddings = conv_params.paddings
        dilations = conv_params.dilations
        output_channels = channels[-1]

        # get input size
        input_size = conv_output_size(input_dims, 
                                      output_channels, 
                                      kernel_sizes, 
                                      strides, 
                                      paddings, 
                                      dilations
                                      )
        
        # create linear layers
        linear_params.set_member_variables(input_size=input_size)
        self.linear_layers = LinearLayers(linear_params)

    def forward(self,  X: torch.tensor) -> torch.tensor:
        X = self.conv_layers(X)
        X = torch.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        y = self.linear_layers(X)
        return y
    