### Set Up ###

# global imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from typing import Union, Optional

from dlearn.networks.cnn import CNN
from dlearn.networks.embedding_nn import EmbeddingNN
from dlearn.networks.mlp import MLP
from dlearn.networks.lstm import LSTM
from dlearn.networks.rnn import RNN


### Functions ###

def predict(data_loader: DataLoader, model: Union[CNN, EmbeddingNN, MLP, LSTM, RNN], 
            save_dir: str, device: str='cuda'):
    # load model weights
    model.load_state_dict(torch.load(save_dir))
    # set up
    predictions = []
    model.eval()

    # iterate over the data
    for data, _ in data_loader:
        data = data.to(device)

        # get predictions and loss
        with torch.no_grad():
            out = model(data)
            predictions.extend(out.tolist())

    # return predictions
    return predictions


### Classes ###

class NeuralNetTrainer():
    def __init__(self, model: Union[CNN, EmbeddingNN, MLP, LSTM, RNN], save_dir: str, 
                 criterion: nn,  minimize_loss: bool=True, 
                 optimizer_type: str='SGD', optimizer_kwargs: dict={}, 
                 scheduler_type: str='StepLR', scheduler_kwargs: dict={},
                 clip: Optional[float]=None, 
                 device: str='cuda'
                 ):
        # set helper member variables
        self.save_dir = save_dir
        self.device = device

        # set model member variables
        self.model = model
        self.model.to(self.device)
        self.clip = clip

        # set criterion member variables
        self.criterion = criterion
        self.minimize_loss = minimize_loss

        # create optimizer
        if optimizer_type == 'SGD':  # lr, momentum, weight_decay, nesterov, maximize
            self.optimizer = optim.SGD(self.model.parameters(), **optimizer_kwargs)
        elif optimizer_type == 'Adam':  # lr, betas, weight_decay, maximize
            self.optimizer = optim.SGD(self.model.parameters, **optimizer_kwargs)
        else:
            raise ValueError('Unsupported optimizer_type: ', optimizer_type)

        # create scheduler
        self.scheduler_requires_loss = False
        
        if scheduler_type == 'StepLR':  # step_size, gamma
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **scheduler_kwargs)
        elif scheduler_type == 'ReduceLROnPlateau':  # factor, patience, threshold
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_kwargs)
            self.scheduler_requires_loss = True
        else: 
            raise ValueError('Unsupported scheduler_type: ', scheduler_type)
        
    def train(self, data_loader: DataLoader):
        # initialize values
        loss = 0
        num_obs = 0
        self.model.train()

        # iterate over the data
        for data, target in data_loader():
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()

            # get predictions and loss
            out = self.model(data)
            _loss = self.criterion(out, target)
            loss += _loss.item()
            num_obs += out.shape[0]

            # backprop
            _loss.backward()

            if self.clip != None:
                nn.utils.clip_grad(self.model.parameters(), self.clip)

            self.optimizer.step()

        # return loss
        loss /= num_obs
        return loss

    def evaluate(self, data_loader: DataLoader):
        # initialize values
        loss = 0
        num_obs = 0
        self.model.eval()

        # iterate over the data
        for data, target in data_loader:
            data = data.to(self.device)
            target = target.to(self.device)

            # get predictions and loss
            with torch.no_grad():
                out = self.model(data)
                _loss = self.criterion(out, target)
                loss += _loss.item()
                num_obs += out.shape[0]

        # return loss
        loss /= num_obs
        return loss

    def train_model(self, train_data_loader: DataLoader, val_data_loader: DataLoader, 
                    test_data_loader: DataLoader, num_epochs: int):
        # initialize values
        train_losses = []
        val_losses = []
        best_index = 0
        best_loss = np.Inf if self.minimize_loss == True else np.NINF

        # iterate over all epochs
        for epoch in range(num_epochs):
            # get train and validation loss
            train_loss = self.train(train_data_loader)
            val_loss = self.evaluate(val_data_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # update scheduler
            if self.scheduler_requires_loss == True:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # save weights if best loss
            if ((self.minimize_loss == True) and (val_loss < best_loss)) or\
                ((self.minimize_loss == False) and (val_loss > best_loss)):

                torch.save(self.model.state_dict(), self.save_dir)
                best_loss = val_loss
                best_index = epoch

        # get test loss
        self.model.load_state_dict(torch.load(self.save_dir))
        test_loss = self.evaluate(test_data_loader)

        # return values
        return train_losses, val_losses, test_loss, best_index
