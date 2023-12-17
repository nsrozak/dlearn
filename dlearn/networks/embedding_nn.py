### Set Up ###

# global imports
import torch
import torch.nn as nn
from dlearn.params.embedding import EmbeddingParams
from dlearn.params.linear import LinearParams
from dlearn.layers.linear import LinearLayers

### Classes ###

class EmbeddingNN(nn.Module):
    def __init__(self, input_size: int, embedding_params: EmbeddingParams, 
                 linear_params: LinearParams):
       # initialize superclass
        super(EmbeddingNN, self).__init__()

        # get parameters
        embedding_dim, max_norm, norm_type, vocab_size =\
            embedding_params.get_params()

        # create embedding layer
        self.embedding_layers = nn.Embedding(vocab_size, embedding_dim, 
                                             max_norm=max_norm, 
                                             norm_type=norm_type
                                            )

        # create linear layer
        input_size = input_size * embedding_dim
        linear_params.set_member_variables(input_size=input_size)
        self.linear_layers = LinearLayers(linear_params)
                                                                           
    def forward(self,  X: torch.tensor) -> torch.tensor:
        X = self.embedding_layers(X)
        X = X.view((-1, 1))
        y = self.linear_layers(X)
        return y
        