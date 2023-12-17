### Set Up ###

# global imports
from dlearn.params.params import Params


### Classes ###

class RecurrentParams(Params):
    # set default variables
    input_size: int=None
    hidden_size: int=None
    num_layers: int=1
    nonlinearity: str='tanh'
    batch_first: bool=True
    dropout: float=0

    def __init__(self, **kwargs):
        # initialize super class
        super(RecurrentParams, self).__init__()

        # specify allowed member variables
        self.member_variables = {'input_size': int,
                                 'hidden_size': int,
                                 'num_layers': int,
                                 'nonlinearity': str,
                                 'dropout': float
                                 }

        # set member vairables
        self.set_member_variables(**kwargs)
        