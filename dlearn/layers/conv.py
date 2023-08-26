### Set Up ###

# global imports
import torch
import torch.nn as nn
from typing import Tuple

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

class ConvParams():
    # set default variables
    input_channel: int=None
    channels: list=[]
    kernel_sizes: list=[]
    strides: list=[]
    paddings: list=[]
    dilations: list=[]
    is_convs: list=[]
    is_normalized: bool=True

    def __init__(self, **kwargs):
        # specify allowed member variables
        self.member_variables = {'input_channel': int, 
                                 'channels': list, 
                                 'kernel_sizes': list, 
                                 'strides': list, 
                                 'paddings': list, 
                                 'dilations': list, 
                                 'is_convs': list, 
                                 'is_normalized': bool
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

class ConvLayers(nn.Module):
    def __init__(self, conv_params: ConvParams):
        # initialize superclass
        super(ConvLayers, self).__init__()

        # get params
        input_channel = conv_params.input_channel 
        channels = conv_params.channels
        kernel_sizes = conv_params.kernel_sizes 
        strides = conv_params.strides 
        paddings = conv_params.paddings 
        dilations = conv_params.dilations 
        is_convs = conv_params.is_convs 
        is_normalized = conv_params.is_normalized

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

        
    def forward(self, X: torch.tensor) -> torch.tensor:
        X = self.conv_layers(X)
        return X
