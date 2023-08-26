### Set Up ###

# global imports
import torch
import torch.nn as nn

from typing import Tuple
from dlearn.models import MLP

### Functions ###

def _conv_output_dims(input_dims: Tuple[int], kernel_size: Tuple[int], 
                      stride: Tuple[int]=(1, 1), padding: Tuple[int]=(0, 0), 
                      dilation: Tuple[int]=(1, 1)
                     ) -> Tuple[int]:
    # set variables
    H_in, W_in = input_dims[0], input_dims[1]
    Kh, Kw = kernel_size[0], kernel_size[1]
    Sh, Sw = stride[0], stride[1]
    Ph, Pw = padding[0], padding[1]
    Dh, Dw = dilation[0], dilation[1]

    # compute dimensions
    H_out = torch.floor(((H_in + (2 * Ph) - (Dh * (Kh - 1)) - 1) / Sh) + 1).item()
    W_out = torch.floor(((W_in + (2 * Pw) - (Dw * (Kw - 1)) - 1) / Sw) + 1).item()
    return (H_out, W_out)


def conv_output_size(input_dims: Tuple[int], output_channels: int, 
                     kernel_sizes: list, strides: list, paddings: list, 
                     dilations: list) -> int:
    # iterate over all dimensions
    for i in range(len(kernel_sizes)):
        # get variables
        kernel_size = kernel_sizes[i]
        stride = strides[i]
        padding = paddings[i]
        dilation = dilations[i]

        # get new dimensions
        input_dims = _conv_output_dims(input_dims, kernel_size, 
                                       stride=stride, 
                                       padding=padding, 
                                       dilation=dilation
                                       )
        
    # compute the new dimension
    output_size = input_dims[0] * input_dims[1] * output_channels
    return output_size



### Classes ###

class CNN(nn.Module):
    def __init__(self, input_dims: Tuple[int], input_channel: int, channels: list, 
                 kernel_sizes: list, strides: list, paddings: list, dilations: list, 
                 is_convs: list, output_size: int, hidden_sizes: list=[], 
                 probabilities: bool=False, is_normalized: bool=True, 
                 p_dropout: float=0):
        # initialize superclass
        super(CNN, self).__init__()

        # create convolutional layers
        self.conv_layers = nn.ModuleList()

        for i in range(len(channels)):
            # get arguments
            in_channel = input_channel if i == 0 else channels[i - 1]
            out_channel = channels[i]
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = paddings[i]
            dilation = dilations[i]
            is_conv = is_convs[i]

            # add the convolution
            if is_conv == True:
                self.conv_layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, 
                                                  stride=stride, 
                                                  padding=padding, 
                                                  dilation=dilation, 
                                                  groups=in_channel
                                                  )
                                        )
                
                # add activation function
                if i < len(channels):
                    self.conv_layers.append(nn.ReLU())
                # add batch norm regularization
                if is_normalized == True:
                    self.conv_layers.append(nn.BatchNorm2d(out_channel))

            # add pooling   
            else: 
                self.conv_layers.append(nn.MaxPool2d(kernel_size, 
                                                     stride=stride, 
                                                     padding=padding, 
                                                     dilation=dilation
                                                     )
                                       )

        # create linear layers
        output_channels = channels[-1]
        input_size = conv_output_size(input_dims, 
                                      output_channels, 
                                      kernel_sizes, 
                                      strides, 
                                      paddings, 
                                      dilations
                                      )
        
        self.linear_layers = MLP(input_size, output_size, 
                                 hidden_sizes=hidden_sizes, 
                                 probabilities=probabilities, 
                                 p_dropout=p_dropout
                                )
        
    def forward(self, X: torch.tensor) -> torch.tensor:
        X = self.conv_layers(X)
        X = torch.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        y = self.linear_layers(X)
        return y
