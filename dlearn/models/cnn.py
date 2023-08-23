### Set Up ###

# global imports
import torch
import torch.nn as nn

### Classes ###

class CNN(nn.Module):
    def __init__(self):
        # initialize superclass
        super(CNN, self).__init__()