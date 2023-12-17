### Set Up ###

# global imports
import torch
from typing import Tuple
from dlearn.params.params import Params


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

class ConvParams(Params):
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
        # initialize super class
        super(ConvParams, self).__init__()

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

        # set member variables
        self.set_member_variables(**kwargs)
