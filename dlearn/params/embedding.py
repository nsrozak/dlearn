### Set Up ###

# global imports
from dlearn.params.params import Params


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
