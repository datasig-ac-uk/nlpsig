import os
from typing import Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from nlpsig.classification_utils import GroupFolds, set_seed
from nlpsig.focal_loss import FocalLoss


def validation_pytorch(
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    verbose: bool = False,
    verbose_epoch: int = 100,
) -> Tuple[float, float]:
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
        f1_v = metrics.f1_score(labels, predicted, average="macro")
        if verbose:
            if epoch % verbose_epoch == 0:
                print(
                    f"Epoch: {epoch+1} || "
                    + f"Loss: {total_loss / len(valid_loader)} || "
                    + f"Accuracy: {accuracy} || "
                    + f"F1-score: {f1_v}."
                )

        return total_loss / len(valid_loader), accuracy, f1_v


def training_pytorch(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    seed: Optional[int] = 42,
    patience: Optional[int] = 3,
    verbose: bool = False,
    verbose_epoch: int = 100,
    verbose_item: int = 1000,
) -> nn.Module:
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
    for epoch in tqdm(range(num_epochs)):
        for i, (emb, labels) in enumerate(train_loader):
            # perform training by performing forward and backward passes
            optimizer.zero_grad()
            outputs = model(emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # show training progress
            if verbose:
                if (epoch % verbose_epoch == 0) and (i % verbose_item == 0):
                    print(
                        f"Epoch: {epoch+1}/{num_epochs} || "
                        + f"Item: {i}/{len(train_loader)} || "
                        + f"Loss: {loss.item()}"
                    )
        # show training progress
        if verbose:
            if epoch % verbose_epoch == 0:
                print("-" * 50)
                print(
                    f"##### Epoch: {epoch+1}/{num_epochs} || " + f"Loss: {loss.item()}"
                )
                print("-" * 50)
        # determine whether or not to stop early using validation set
        _, __, f1_v = validation_pytorch(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            epoch=epoch,
            verbose=verbose,
            verbose_epoch=verbose_epoch,
        )
        if f1_v < last_metric:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break
        else:
            trigger_times = 0
        last_metric = f1_v

    return model


def testing_pytorch(
    model: nn.Module, test_loader: DataLoader
) -> Tuple[torch.tensor, torch.tensor]:
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


def KFold_pytorch(
    folds: GroupFolds,
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
    },
) -> pd.DataFrame:
    torch.save(
        obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "criterion": criterion,
        },
        f="starting_state.pkl",
    )
    accuracy = []
    f1_score = []
    for fold in tqdm(range(folds.n_splits)):
        print("\n" + "*" * 50)
        print(f"Fold: {fold+1} / {folds.n_splits}")
        print("*" * 50)

        # reload starting state of the model, optimizer and loss
        checkpoint = torch.load(f="starting_state.pkl")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        criterion = checkpoint["criterion"]
        if isinstance(criterion, FocalLoss):
            y_train = folds.get_splits(fold_index=fold)[5]
            criterion.set_alpha_from_y(y=torch.tensor(y_train))

        # obtain test, valid and test dataloaders
        train, valid, test = folds.get_splits(fold_index=fold, as_DataLoader=True)

        # train pytorch model
        model = training_pytorch(
            model=model,
            train_loader=train,
            valid_loader=valid,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            seed=seed,
            patience=patience,
            **verbose_args,
        )

        # test model
        predicted, labels = testing_pytorch(model=model, test_loader=test)

        # evaluate model
        accuracy.append(((predicted == labels).sum() / labels.size(0)).item())
        f1_score.append(metrics.f1_score(labels, predicted, average="macro"))

    # remove starting state pickle file
    os.remove("starting_state.pkl")
    return pd.DataFrame({"accuracy": accuracy, "f1_score": f1_score})