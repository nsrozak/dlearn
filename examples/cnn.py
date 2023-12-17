### Set Up ###

# global imports
import torch.nn as nn
from torch.utils.data import DataLoader
from dlearn.params import ConvParams, LinearParams
from dlearn.networks import CNN
from dlearn.utils import NeuralNetTrainer, predict

# param arguments
input_dims = [0, 0]

conv_input_channel = 0
conv_channels = []
conv_kernel_sizes = []
conv_strides = []
conv_paddings = []
conv_dilations = []
conv_is_convs = []
conv_is_normalized = True

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
conv_params = ConvParams(input_channel=conv_input_channel,
                         channels=conv_channels,
                         kernel_sizes=conv_kernel_sizes,
                         strides=conv_strides,
                         paddings=conv_paddings,
                         dilations=conv_dilations,
                         is_convs=conv_is_convs,
                         is_normalized=conv_is_normalized
                         )

linear_params = LinearParams(output_size=linear_output_size,
                             hidden_sizes=linear_hidden_sizes,
                             probabilities=linear_probabilities,
                             dropout=linear_dropout
                             )
model = CNN(input_dims, conv_params, linear_params)

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
