### Set Up ###

# global imports
import torch.nn as nn
from torch.utils.data import DataLoader
from dlearn.params import LinearParams
from dlearn.networks import MLP
from dlearn.utils import NeuralNetTrainer, predict

# param arguments
input_size = 0
output_size = 0
hidden_sizes = []
probabilities = False
dropout = 0

# trainer arguments
save_dir = ''
criterion = nn.MSELoss()  
minimize_loss = True
optimizer_type = 'SGD'
optimizer_kwargs = {}
scheduler_type = 'StepLR'
scheduler_kwargs = {}
clip = None
device = 'cuda'
num_epochs = 0
                 

### Main Program ###

# load the data
train_data_loader = DataLoader()
val_data_loader = DataLoader()
test_data_loader = DataLoader()

# create the model
linear_params = LinearParams(input_size=input_size, 
                             output_size=output_size,
                             hidden_sizes=hidden_sizes,
                             probabilities=probabilities,
                             dropout=dropout
                             )

model = MLP(linear_params)

# train the model
neural_net_trainer = NeuralNetTrainer(model, 
                                      criterion,  
                                      minimize_loss=minimize_loss, 
                                      optimizer_type=optimizer_type, 
                                      optimizer_kwargs=optimizer_kwargs, 
                                      scheduler_type=scheduler_type, 
                                      scheduler_kwargs=scheduler_kwargs,
                                      clip=clip, 
                                      device=device
                                     )

neural_net_trainer.train_model(train_data_loader, 
                               val_data_loader, 
                               test_data_loader, 
                               num_epochs
                               )

# get predictions
predictions = predict(test_data_loader, model, save_dir, device=device)
