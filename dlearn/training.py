### Set Up ###

# global imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from typing import Union, Optional

from dlearn.models.cnn import CNN
from dlearn.models.embedding_nn import EmbeddingNN
from dlearn.models.mlp import MLP
from dlearn.models.lstm import LSTM
from dlearn.models.rnn import RNN

### Classes ###

class NeuralNet():
    def __init__(self, model: Union[CNN, EmbeddingNN, MLP, LSTM, RNN], 
                 optimizer: optim, scheduler: optim.lr_scheduler, 
                 criterion: nn, save_dir: str, clip: Optional[float]=None, 
                 device: str='cuda', scheduler_requires_loss: bool=False, 
                 minimize_loss: bool=True):
        # set helper member variables
        self.save_dir = save_dir
        self.scheduler_requires_loss = scheduler_requires_loss
        self.minimize_loss = minimize_loss

        self.device = device
        self.clip = clip

        # set deep learning member variables
        self.model = model
        self.model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

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
