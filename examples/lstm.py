### Set Up ###

# global imports
import torch.nn as nn
from torch.utils.data import DataLoader
from dlearn.params import RecurrentParams, LinearParams
from dlearn.networks import LSTM
from dlearn.utils import NeuralNetTrainer, predict

# param arguments
recurrent_input_size = 0
recurrent_hidden_size = 0
recurrent_num_layers = 1
recurrent_nonlinearity = 'tanh'
recurrent_batch_first = True
recurrent_dropout = 0

linear_output_size = 0
linear_hidden_sizes = []
linear_probabilities = False
linear_dropout = 0

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
recurrent_params = RecurrentParams(input_size=recurrent_input_size,
                                   hidden_size=recurrent_hidden_size,
                                   num_layers=recurrent_num_layers,
                                   nonlinearity=recurrent_nonlinearity,
                                   batch_first=recurrent_batch_first,
                                   dropout=recurrent_dropout
                                   )

linear_params = LinearParams(output_size=linear_output_size,
                             hidden_sizes=linear_hidden_sizes,
                             probabilities=linear_probabilities,
                             dropout=linear_dropout
                             )

model = LSTM(recurrent_params, linear_params)

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
