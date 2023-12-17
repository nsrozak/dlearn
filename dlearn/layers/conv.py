### Set Up ###

# global imports
import torch
import torch.nn as nn
from dlearn.params.conv import ConvParams


### Classes ###

class ConvLayers(nn.Module):
    def __init__(self, conv_params: ConvParams):
        # initialize superclass
        super(ConvLayers, self).__init__()

        # get params
        channels, dilations, input_channel, is_normalized, is_convs, kernel_sizes, paddings, strides =\
            conv_params.get_params()

        # create convolutional layers
        layers = []

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
                layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, 
                                        stride=stride, 
                                        padding=padding, 
                                        dilation=dilation, 
                                        groups=in_channel
                                       )
                             )
                
                # add activation function
                layers.append(nn.ReLU())
                
                # add batch norm regularization
                if is_normalized == True:
                    layers.append(nn.BatchNorm2d(out_channel))

            # add pooling   
            else: 
                layers.append(nn.MaxPool2d(kernel_size, 
                                           stride=stride, 
                                           padding=padding, 
                                           dilation=dilation
                                          )
                             )
                
        # create neural net object
        self.conv_layers = nn.Sequential(*layers)
        
    def forward(self, X: torch.tensor) -> torch.tensor:
        X = self.conv_layers(X)
        return X
