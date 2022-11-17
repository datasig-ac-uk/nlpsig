import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


class GroupFolds:
    def __init__(
        self,
        df: pd.DataFrame,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        groups: torch.Tensor,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int = 42,
    ):
        if n_splits < 2:
            raise ValueError("n_splits should be at least 2")
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError(
                "x_data and y_data do not have compatible shapes "
                + "(need to have same number of samples)"
            )
        self.df = df
        self.x_data = x_data
        self.y_data = y_data
        self.groups = groups
        self.n_splits = n_splits
        self.shuffle = shuffle
        if self.shuffle:
            # GroupShuffleSplit doesn't guarantee that every group is in a test group
            self.random_state = random_state
            self.fold = GroupShuffleSplit(
                n_splits=self.n_splits, random_state=self.random_state
            )
        else:
            # GroupKFold guarantees that every group is in a test group once
            self.random_state = None
            self.fold = GroupKFold(n_splits=self.n_splits)
        self.fold_indices = list(self.fold.split(X=x_data, groups=groups))

    def get_splits(
        self,
        fold_index: int,
        dev_size: float = 0.33,
        as_DataLoader=False,
        data_loader_args: dict = {"batch_size": 1, "shuffle": True},
    ):
        if fold_index not in list(range(self.n_splits)):
            raise ValueError(
                f"There are {self.n_splits} folds, so "
                + f"fold_index must be in {list(range(self.n_splits))}"
            )
        # obtain train and test indices for provided fold_index
        train_index = self.fold_indices[fold_index][0]
        test_index = self.fold_indices[fold_index][1]
        # obtain a validation set from the training set
        train_index, valid_index = train_test_split(
            train_index,
            test_size=dev_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

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
