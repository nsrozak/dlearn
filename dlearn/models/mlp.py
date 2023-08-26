### Set Up ###

# global imports
import torch
import torch.nn as nn

### Classes ###

class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list=[], 
                 probabilities: bool=False, p_dropout: float=0):
        # initialize superclass
        super(MLP, self).__init__()

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
    