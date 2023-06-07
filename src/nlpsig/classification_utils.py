from __future__ import annotations

import os
import random

import torch
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    train_test_split,
)
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


class DataSplits:
    """
    Class to split the data into train, validation and test sets.
    """

    def __init__(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        train_size: float = 0.8,
        valid_size: float | None = 0.33,
        shuffle: bool = False,
        random_state: int = 42,
    ):
        """
        Class to split the data into train, validation and test sets.

        Parameters
        ----------
        x_data : torch.Tensor
            Features for prediction.
        y_data : torch.Tensor
            Variable to predict.
        train_size : float, optional
            Proportion of data to use as training data, by default 0.8.
        valid_size : float | None, optional
            Proportion of training data to use as validation data, by default 0.33.
            If None, will not create a validation set.
        shuffle : bool, optional
            Whether or not to shuffle the dataset, by default False.
        random_state : int, optional
            Seed number, by default 42.
        """
        if (train_size < 0) or (train_size > 1):
            msg = "train_size must be between 0 and 1"
            raise ValueError(msg)
        if valid_size is not None and ((train_size < 0) or (train_size > 1)):
            msg = "valid_size must be between 0 and 1"
            raise ValueError(msg)

        # first split data into train set, test/valid set
        train_index, test_index = train_test_split(
            range(len(y_data)),
            test_size=(1 - train_size),
            shuffle=shuffle,
            random_state=random_state,
        )

        if valid_size is not None:
            # further split the train set into a train, valid set
            train_index, valid_index = train_test_split(
                train_index,
                test_size=valid_size,
                shuffle=shuffle,
                random_state=random_state,
            )
            self.x_valid = x_data[valid_index]
            self.y_valid = y_data[valid_index]
        else:
            self.x_valid = None
            self.y_valid = None

        self.x_train = x_data[train_index]
        self.y_train = y_data[train_index]
        self.x_test = x_data[test_index]
        self.y_test = y_data[test_index]

    def get_splits(
        self, as_DataLoader: bool = False, data_loader_args: dict | None = None
    ):
        """
        Returns train, validation and test set.

        Parameters
        ----------
        as_DataLoader : bool, optional
            Whether or not to return as `torch.utils.data.dataloader.DataLoader` objects
            ready to be passed into PyTorch model, by default False.
        data_loader_args : dict | None, optional
            Any keywords to be passed in obtaining the
            `torch.utils.data.dataloader.DataLoader` object,
            by default {"batch_size": 64, "shuffle": True}.

        Returns
        -------
        - If `as_DataLoader` is True, return tuple of
        `torch.utils.data.dataloader.DataLoader` objects:
          - First element is training dataset
          - Second element is validation dataset
          - Third element is testing dataset
        - If `as_DataLoader` is False, returns tuple of `torch.Tensors`:
          - First element is features for testing dataset
          - Second element is labels for testing dataset
          - First element is features for validation dataset
          - Second element is labels for validation dataset
          - First element is features for training dataset
          - Second element is labels for training dataset
        """
        if data_loader_args is None:
            data_loader_args = {"batch_size": 64, "shuffle": True}

        if as_DataLoader:
            if self.x_valid is not None:
                valid = TensorDataset(self.x_valid, self.y_valid)
                valid_loader = DataLoader(dataset=valid, **data_loader_args)
            else:
                valid_loader = None

            train = TensorDataset(self.x_train, self.y_train)
            test = TensorDataset(self.x_test, self.y_test)
            train_loader = DataLoader(dataset=train, **data_loader_args)
            test_loader = DataLoader(dataset=test, **data_loader_args)

            return train_loader, valid_loader, test_loader
        return (
            self.x_test,
            self.y_test,
            self.x_valid,
            self.y_valid,
            self.x_train,
            self.y_train,
        )


class Folds:
    """
    Class to split the data into different folds based on groups.
    """

    def __init__(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        groups: torch.Tensor | None = None,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int = 42,
    ):
        """
        Class to split the data into different folds based on groups

        Parameters
        ----------
        x_data : torch.Tensor
            Features for prediction.
        y_data : torch.Tensor
            Variable to predict.
        groups : torch.Tensor | None, optional
            Groups to split by, default None. If None is passed, then does standard KFold,
            otherwise implements GroupShuffleSplit (if shuffle is True),
            or GroupKFold (if shuffle is False).
        n_splits : int, optional
            Number of splits / folds, by default 5.
        shuffle : bool, optional
            Whether or not to shuffle the dataset, by default False.
        random_state : int, optional
            Seed number, by default 42.

        Raises
        ------
        ValueError
            if `n_splits` < 2.
        ValueError
            if `x_data` and `y_data` do not have the same number of records
            (number of rows in `x_data` should equal the length of `y_data`).
        ValueError
            if `x_data` and `groups` do not have the same number of records
            (number of rows in `x_data` should equal the length of `groups`).
        """
        if n_splits < 2:
            msg = "n_splits should be at least 2"
            raise ValueError(msg)
        if x_data.shape[0] != y_data.shape[0]:
            msg = (
                "x_data and y_data do not have compatible shapes "
                "(need to have same number of samples)"
            )
            raise ValueError(msg)
        if groups is not None and x_data.shape[0] != groups.shape[0]:
            msg = (
                "x_data and groups do not have compatible shapes "
                "(need to have same number of samples)"
            )
            raise ValueError(msg)

        self.x_data = x_data
        self.y_data = y_data
        self.groups = groups
        self.n_splits = n_splits
        self.shuffle = shuffle
        if self.shuffle:
            self.random_state = random_state
        else:
            self.random_state = None
        if self.groups is not None:
            if self.shuffle:
                # GroupShuffleSplit does not guarantee that every group is in a test group
                self.fold = GroupShuffleSplit(
                    n_splits=self.n_splits, random_state=self.random_state
                )
            else:
                # GroupKFold guarantees that every group is in a test group once
                self.fold = GroupKFold(n_splits=self.n_splits)
        else:
            self.fold = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        self.fold_indices = list(self.fold.split(X=x_data, groups=groups))

    def get_splits(
        self,
        fold_index: int,
        valid_size: float = 0.33,
        as_DataLoader: bool = False,
        data_loader_args: dict | None = None,
    ) -> (
        tuple[DataLoader, DataLoader, DataLoader]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """
        Obtains the data from a particular fold

        Parameters
        ----------
        fold_index : int
            Which fold to obtain data for
        valid_size : float, optional
            Proportion of training data to use as validation data, by default 0.33.
        as_DataLoader : bool, optional
            Whether or not to return as `torch.utils.data.dataloader.DataLoader` objects
            ready to be passed into PyTorch model, by default False.
        data_loader_args : dict | None, optional
            Any keywords to be passed in obtaining the
            `torch.utils.data.dataloader.DataLoader` object,
            by default {"batch_size": 64, "shuffle": True}.

        Returns
        -------
        - If `as_DataLoader` is True, return tuple of
        `torch.utils.data.dataloader.DataLoader` objects:
          - First element is training dataset
          - Second element is validation dataset
          - Third element is testing dataset
        - If `as_DataLoader` is False, returns tuple of `torch.Tensors`:
          - First element is features for testing dataset
          - Second element is labels for testing dataset
          - First element is features for validation dataset
          - Second element is labels for validation dataset
          - First element is features for training dataset
          - Second element is labels for training dataset

        Raises
        ------
        ValueError
            if the requested `fold_index` is not valid.
        """

        if data_loader_args is None:
            data_loader_args = {"batch_size": 64, "shuffle": True}

        if fold_index not in list(range(self.n_splits)):
            msg = (
                f"There are {self.n_splits} folds, so "
                f"fold_index must be in {list(range(self.n_splits))}"
            )
            raise ValueError(msg)
        # obtain train and test indices for provided fold_index
        train_index = self.fold_indices[fold_index][0]
        test_index = self.fold_indices[fold_index][1]
        # obtain a validation set from the training set
        train_index, valid_index = train_test_split(
            train_index,
            test_size=valid_size,
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
        return x_test, y_test, x_valid, y_valid, x_train, y_train


def set_seed(seed: int) -> None:
    """
    Helper function for reproducible behavior to set the seed in
    `random`, `torch`.

    Parameters
    ----------
    seed : int
        Seed number.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)  # not needed with numpy generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
