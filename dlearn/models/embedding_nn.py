### Set Up ###

# global imports
import torch
import torch.nn as nn
from dlearn.layers.linear import LinearLayers, LinearParams
from dlearn.utils import Params

### Classes ###

class EmbeddingParams(Params):
    # set default variables
    vocab_size: int=None
    embedding_dim: int=None
    max_norm: float=None
    norm_type: float=2.0

    def __init__(self, **kwargs):
        # initialize superclass
        super(EmbeddingParams, self).__init__()

        # specify allowed member variables
        self.member_variables = {'num_embeddings': int,
                                 'embedding_dim': int,
                                 'max_norm': float,
                                 'norm_type': float
                                 }

        # set member vairables
        self.set_member_variables(**kwargs)


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
        