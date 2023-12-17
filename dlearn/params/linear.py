### Set Up ###

# global imports
from dlearn.params.params import Params


### Classes ###

class LinearParams(Params):
    # set default variables
    input_size: int=None
    output_size: int=None
    hidden_sizes: list=[]
    probabilities: bool=False
    dropout: float=0

    def __init__(self, **kwargs):
        # initialize super class
        super(LinearParams, self).__init__()

        # specify allowed member variables
        self.member_variables = {'input_size': int, 
                                 'output_size': int, 
                                 'hidden_sizes': list, 
                                 'probabilities': bool, 
                                 'dropout': float
                                 }

        # set member vairables
        self.set_member_variables(**kwargs)
        