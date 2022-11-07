import os
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split
from sklearn import metrics
import torch
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
from torch.optim.optimizer import Optimizer
from typing import Tuple, Optional

class Folds:
    def __init__(self,
                 df: pd.DataFrame,
                 x_data: torch.Tensor,
                 y_data: torch.Tensor,
                 groups: torch.Tensor,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: int = 42):
        if n_splits < 2:
            raise ValueError("n_splits should be at least 2")
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError("x_data and y_data do not have compatible shapes " +
                             "(need to have same number of samples)")
        self.df = df
        self.x_data = x_data
        self.y_data = y_data
        self.groups = groups
        self.n_splits = n_splits
        self.shuffle = shuffle
        if self.shuffle:
            # GroupShuffleSplit doesn't guarantee that every group is in a test group
            self.random_state = random_state
            self.fold = GroupShuffleSplit(n_splits = self.n_splits,
                                          shuffle = self.shuffle,
                                          random_state = self.random_state)
        else:
            # GroupKFold guarantees that every group is in a test group once
            self.random_state = None
            self.fold = GroupKFold(n_splits = self.n_splits)
        self.fold_indices = list(self.fold.split(X = x_data, groups = groups))

    def get_splits(self,
                   fold_index: int,
                   dev_size: float = 0.33,
                   as_DataLoader = False,
                   data_loader_args: dict = {"batch_size": 1, 
                                             "shuffle": True}):
        if fold_index not in list(range(self.n_splits)):
            raise ValueError(f"There are {self.n_splits} folds, so " + 
                             f"fold_index must be in {list(range(self.n_splits))}")
        # obtain train and test indices for provided fold_index
        train_index = self.fold_indices[fold_index][0]
        test_index = self.fold_indices[fold_index][1]
        # obtain a validation set from the training set
        train_index, valid_index = train_test_split(train_index,
                                                    test_size = dev_size,
                                                    shuffle = self.shuffle,
                                                    random_state = self.random_state)
        
        x_train = self.x_data[train_index]
        y_train = self.y_data[train_index]
        x_valid = self.x_data[valid_index]
        y_valid = self.y_data[valid_index]
        x_test = self.x_data[test_index]
        y_test = self.y_data[test_index]
        
        if as_DataLoader:
            train = TensorDataset(x_train, y_train)
            valid = TensorDataset(x_valid, y_valid)
            test = TensorDataset(x_test, y_test)

            train_loader = DataLoader(dataset=train, **data_loader_args)
            valid_loader = DataLoader(dataset=valid, **data_loader_args)
            test_loader = DataLoader(dataset=test, **data_loader_args)
            
            return train_loader, valid_loader, test_loader
        else:
            return x_test, y_test, x_valid, y_valid, x_train, y_train

def set_seed(seed: int) -> None:
    """
    Helper function for reproducible behavior to set the seed in 
    ``random``, ``numpy``, ``torch`` (if installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def validation(model: nn.Module,
               valid_loader: DataLoader,
               criterion: nn.Module,
               epoch: int,
               verbose: bool = False) -> Tuple[float, float]:
    """
    Evaluates the PyTorch model to a validation set and returns
    the total loss, accuracy and F1 score
    """
    # sets the model to evaluation mode
    model.eval()
    number_of_labels = 0
    total_loss = 0
    labels = torch.empty((0))
    predicted = torch.empty((0))
    with torch.no_grad():     
        for emb_v, labels_v in valid_loader:
            # make prediction
            outputs = model(emb_v)
            _, predicted_v = torch.max(outputs.data, 1)
            number_of_labels += labels_v.size(0)
            # compute loss
            loss_v = criterion(outputs, labels_v)
            total_loss += loss_v.item()
            # save predictions and labels
            labels = torch.cat([labels, labels_v])
            predicted = torch.cat([predicted, predicted_v])
        # compute accuracy and f1 score 
        accuracy = ((predicted == labels).sum() / number_of_labels).item()
        f1_v = metrics.f1_score(labels,
                                predicted,
                                average = 'macro')
        if verbose:
            print(f"Epoch: {epoch} || " +
                  f"Loss: {total_loss / len(valid_loader)} || " +
                  f"Accuracy: {accuracy} || " +
                  f"F1-score: {f1_v}.")
        
        return total_loss / len(valid_loader), accuracy, f1_v

def training(model: nn.Module,
             train_loader: DataLoader,
             valid_loader: DataLoader,
             criterion: nn.Module,
             optimizer: Optimizer,
             num_epochs: int,
             seed: Optional[int] = 42,
             patience: Optional[int] = 3, 
             verbose: bool = False,
             verbose_epoch: int = 100,
             verbose_item: int = 1000) -> nn.Module:
    """
    Trains the PyTorch model using some training dataset and uses a validation dataset
    to determine if early stopping is used
    """
    # sets the model to training mode
    model.train()
    set_seed(seed)
    # early stopping parameters
    last_metric = 0
    trigger_times = 0
    # model train & validation per epoch
    for epoch in range(num_epochs):
        for i, (emb, labels) in enumerate(train_loader):
            # perform training by performing forward and backward passes
            optimizer.zero_grad()
            outputs = model(emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # show training progress
            if verbose:
                if (i % verbose_item == 0):
                    print(f"Epoch: {epoch+1}/{num_epochs} || " +
                          f"Item: {i}/{len(train_loader)} || " +
                          f"Loss: {loss.item()}")
        # show training progress
        if verbose:
            if (epoch % verbose_epoch == 0):
                print("-"*50)
                print(f"##### Epoch: {epoch+1}/{num_epochs} || " +
                      f"Loss: {loss.item()}")
                print("-"*50)
        # determine whether or not to stop early using validation set
        _, __, f1_v = validation(model = model,
                                 valid_loader = valid_loader,
                                 criterion = criterion,
                                 epoch = epoch,
                                 verbose = verbose)
        if f1_v < last_metric:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break
        else:
            trigger_times = 0
        last_metric = f1_v
        
    return model

def testing(model: nn.Module,
            test_loader: DataLoader) -> Tuple[torch.tensor, torch.tensor]:
    """
    Evaluates the PyTorch model to a validation set and returns the
    predicted labels and their corresponding true labels
    """
    # sets the model to evaluation mode
    model.eval()
    labels_all = torch.empty((0))
    predicted_all = torch.empty((0))
    with torch.no_grad():   
        # Iterate through test dataset
        for emb_t, labels_t in test_loader:
            # make prediction
            outputs_t = model(emb_t)
            _, predicted_t = torch.max(outputs_t.data, 1)
            # save predictions and labels
            labels_all = torch.cat([labels_all, labels_t])
            predicted_all = torch.cat([predicted_all, predicted_t])
      
    return predicted_all, labels_all

def KFold_pytorch(folds: Folds,
                  model: nn.Module,
                  criterion: nn.Module,
                  optimizer: Optimizer,
                  num_epochs: int,
                  seed: Optional[int] = 42,
                  patience: Optional[int] = 3,
                  verbose_args: dict = {
                      "verbose": True,
                      "verbose_epoch": 100,
                      "verbose_item": 10000,
                  }) -> pd.DataFrame:
    torch.save(obj = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "criterion": criterion},
               f = "starting_state.pkl")
    accuracy = []
    f1_score = []
    for fold in range(folds.n_splits):
        print("-"*50)
        print(f"Fold: {fold+1} / {folds.n_splits}")
        print("-"*50)
        train, valid, test = folds.get_splits(fold_index = fold,
                                              as_DataLoader = True)
        # reload starting state of the model, optimizer and loss
        checkpoint = torch.load(f = "starting_state.pkl")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        criterion = checkpoint["criterion"]
        # train pytorch model
        model = training(model = model,
                         train_loader = train,
                         valid_loader = valid,
                         criterion = criterion,
                         optimizer = optimizer,
                         num_epochs = num_epochs,
                         seed = seed,
                         patience = patience,
                         **verbose_args)
        # test model
        predicted, labels = testing(model = model,
                                    test_loader = test)
        # evaluate model
        accuracy.append(((predicted == labels).sum() / labels.size(0)).item())
        f1_score.append(metrics.f1_score(labels,
                                         predicted,
                                         average = 'macro'))
    # remove starting state pickle file
    os.remove("starting_state.pkl")
    return pd.DataFrame({"accuracy": accuracy,
                         "f1_score": f1_score})
        