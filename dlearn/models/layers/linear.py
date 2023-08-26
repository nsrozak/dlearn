### Set Up ###

# global imports
import torch
import torch.nn as nn

### Classes ###

class LinearParams():
    # set default variables
    input_size: int=None
    output_size: int=None
    hidden_sizes: list=[]
    probabilities: bool=False
    p_dropout: float=0

    def __init__(self, **kwargs):
        # specify allowed member variables
        self.member_variables = {'input_size': int, 
                                 'output_size': int, 
                                 'hidden_sizes': list, 
                                 'probabilities': bool, 
                                 'p_dropout': float
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

class LinearLayers(nn.Module):
    def __init__(self, linear_params: LinearParams):
        # initialize superclass
        super(LinearLayers, self).__init__()

        # get parameters
        input_size = linear_params.input_size
        output_size = linear_params.output_size 
        hidden_sizes = linear_params.hidden_sizes
        probabilities = linear_params.probabilities
        p_dropout = linear_params.p_dropout

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
            if (p_dropout > 0) and i < n_layers:
                self.linear_layers.append(nn.Dropout(p=p_dropout))

        # add softmax to normalize for probabilities
        if probabilities == True:
            self.linear_layers.append(nn.Softmax())

    def forward(self, X: torch.tensor) -> torch.tensor:
        y = self.linear_layers(X)
        return y
    