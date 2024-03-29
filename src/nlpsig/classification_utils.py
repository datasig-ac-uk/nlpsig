from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
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

    Parameters
    ----------
    x_data : np.array | torch.Tensor | dict[str, np.array | torch.Tensor]
        Features for prediction. This can be a numpy array or torch tensor
        of shape (n_samples, n_features), or a dictionary where the keys
        are the names of the features and the values are numpy arrays or
        torch tensors of shape (n_samples, n_features).
    y_data : np.array | torch.Tensor
        Variable to predict.
    groups : np.array | torch.Tensor | None, optional
        Groups to split by, default None.
        If groups are passed, then GroupShuffleSplit is used to create a dataset
        split where groups are fit entirely within a datasplit, i.e.
        groups would not span over the different splits created.
    train_size : float, optional
        Proportion of data to use as training data, by default 0.8.
    valid_size : float | None, optional
        Proportion of training data to use as validation data, by default 0.33.
        If None, will not create a validation set.
    indices : tuple[Iterable[int], Iterable[int] | None, Iterable[int]] | None, optional
        Train, validation, test indices to use. If passed, will split the data
        according to these indices rather than splitting it within the method
        using the train_size and valid_size provided.
        First item in the tuple should be the indices for the training set,
        second item should be the indices for the validaton set (this could
        be None if no validation set is required), and third item should be
        indices for the test set.
    shuffle : bool, optional
        Whether or not to shuffle the dataset, by default False.
        This is ignored if either groups are passed, or if indices are passed.
    random_state : int, optional
        Seed number, by default 42.
        This is ignored if indices are passed.
    """

    def __init__(
        self,
        x_data: np.array | torch.Tensor | dict[str, np.array | torch.Tensor],
        y_data: np.array | torch.Tensor,
        groups: np.array | torch.Tensor | None = None,
        train_size: float = 0.8,
        valid_size: float | None = 0.33,
        indices: tuple[Iterable[int], Iterable[int], Iterable[int]] | None = None,
        shuffle: bool = False,
        random_state: int = 42,
    ):
        if not isinstance(x_data, dict):
            # make x_data a dictionary with key "x_data"
            x_data = {"x_data": x_data}

        # iterate through the values and check they are of the correct shape
        for key, value in x_data.items():
            if value.shape[0] != y_data.shape[0]:
                msg = (
                    f"{key} and y_data do not have compatible shapes "
                    "(need to have same number of samples)."
                )
                raise ValueError(msg)

        # if groups are passed, check that groups is of the correct shape
        if groups is not None and y_data.shape[0] != len(groups):
            msg = (
                "x_data and groups do not have compatible shapes "
                "(need to have same number of samples)."
            )
            raise ValueError(msg)

        # check train_size and valid_size are between 0 and 1
        if (train_size < 0) or (train_size > 1):
            msg = "train_size must be between 0 and 1."
            raise ValueError(msg)
        if valid_size is not None and ((valid_size < 0) or (valid_size > 1)):
            msg = "valid_size must be between 0 and 1."
            raise ValueError(msg)

        self.x_data: dict[str, np.array | torch.Tensor] = x_data
        self.y_data: np.array | torch.Tensor = y_data
        self.groups: np.array | torch.Tensor | None = groups
        self.shuffle: bool | None = shuffle
        self.random_state: int | None = random_state

        if indices is not None:
            self.shuffle = False
            self.random_state = None

            # indices are provided, so use these to split the dataset
            msg = "if indices are provided, it must be a tuple of length 3."
            if not isinstance(indices, tuple):
                raise TypeError(msg)
            if len(indices) != 3:
                raise ValueError(msg)

            # check that the indices passed in are within range
            for i in range(len(indices)):
                if (indices[i] is not None) and not all(
                    j in list(range(len(y_data))) for j in indices[i]
                ):
                    problem_set = "train" if i == 0 else "valid" if i == 1 else "test"
                    msg = (
                        f"in the {problem_set} indices, "
                        "some of the indices will be out of range."
                    )
                    raise IndexError(msg)

            train_index = indices[0]
            valid_index = indices[1]
            test_index = indices[2]
        else:
            if self.groups is not None:
                # see https:/github.com/scikit-learn/scikit-learn/issues/9193
                self.shuffle = True

                # first split data into train set, test/valid set by group
                gss = GroupShuffleSplit(
                    n_splits=1, train_size=train_size, random_state=self.random_state
                )
                train_index, test_index = next(
                    gss.split(
                        X=next(iter(self.x_data.values())),
                        y=self.y_data,
                        groups=self.groups,
                    )
                )

                if valid_size is not None:
                    # further split the train set into a train, valid set
                    gss = GroupShuffleSplit(
                        n_splits=1, test_size=valid_size, random_state=self.random_state
                    )

                    # use GroupShuffleSplit to split the train indices further to get valid set
                    t_ind, v_ind = next(
                        gss.split(
                            X=next(iter(self.x_data.values()))[train_index],
                            y=self.y_data[train_index],
                            groups=self.groups[train_index],
                        )
                    )

                    # obtain updated train and valid set
                    valid_index = train_index[v_ind].tolist()
                    train_index = train_index[t_ind]
                else:
                    valid_index = None

                # convert test_index to list (from np.array)
                train_index = train_index.tolist()
                test_index = test_index.tolist()
            else:
                if not self.shuffle:
                    self.random_state = None

                # first split data into train set, test/valid set
                train_index, test_index = train_test_split(
                    range(len(self.y_data)),
                    test_size=(1 - train_size),
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                )

                if valid_size is not None:
                    # further split the train set into a train, valid set
                    train_index, valid_index = train_test_split(
                        train_index,
                        test_size=valid_size,
                        shuffle=self.shuffle,
                        random_state=self.random_state,
                    )
                else:
                    valid_index = None

        # store indices
        self.indices = (train_index, valid_index, test_index)

    def get_splits(
        self, as_DataLoader: bool = False, data_loader_args: dict | None = None
    ) -> (
        tuple[DataLoader, DataLoader, DataLoader]
        | tuple[np.array | torch.Tensor | dict[str, np.array | torch.Tensor]]
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
            by default ``{"batch_size": 64, "shuffle": True}``.

        Returns
        -------
        tuple[DataLoader] | tuple[np.array | torch.Tensor | dict[str, np.array | torch.Tensor]]
            If `as_DataLoader` is True, return tuple of `torch.utils.data.dataloader.DataLoader` objects:
                - First element is training dataset
                - Second element is validation dataset
                - Third element is testing dataset

            If `as_DataLoader` is False, returns tuple:
                - First element is features (which is either an array/tensor or dictionary) for training dataset
                - Second element is labels for training dataset as a array/tensor
                - Third element is features (which is either an array/tensor or dictionary) for validation dataset
                - Fourth element is labels for validation dataset as a array/tensor
                - Fifth element is features (which is either an array/tensor or dictionary) for testing dataset
                - Sixth element is labels for testing dataset as a array/tensor
        """
        if data_loader_args is None:
            data_loader_args = {"batch_size": 64, "shuffle": True}

        # obtain validation set
        if self.indices[1] is not None:
            x_valid = {
                key: value[self.indices[1]] for key, value in self.x_data.items()
            }
            y_valid = self.y_data[self.indices[1]]
        else:
            x_valid = None
            y_valid = None

        # obtain training set
        x_train = {key: value[self.indices[0]] for key, value in self.x_data.items()}
        y_train = self.y_data[self.indices[0]]

        # obtain test set
        x_test = {key: value[self.indices[2]] for key, value in self.x_data.items()}
        y_test = self.y_data[self.indices[2]]

        if as_DataLoader:
            # return datasets as DataLoader objects if requested
            if x_valid is not None:
                # make sure that each value in x_valid is a torch tensor
                for key, value in x_valid.items():
                    if isinstance(value, np.ndarray):
                        x_valid[key] = torch.from_numpy(value)
                # make sure that y_valid is a torch tensor
                if isinstance(y_valid, np.ndarray):
                    y_valid = torch.from_numpy(y_valid)

                # create validation TensorDataset from x_valid (a dictionary) and y_valid
                # extract tensors from the dictionary to get list of tensors
                # create a TensorDataset using the unpacked tensors
                valid = TensorDataset(*list(x_valid.values()), y_valid)
                valid_loader = DataLoader(dataset=valid, **data_loader_args)
            else:
                valid_loader = None

            # make sure that each value in x_train and x_test is a torch tensor
            for key, value in x_train.items():
                if isinstance(value, np.ndarray):
                    x_train[key] = torch.from_numpy(value)
            for key, value in x_test.items():
                if isinstance(value, np.ndarray):
                    x_test[key] = torch.from_numpy(value)

            # make sure that y_train and y_test are torch tensors
            if isinstance(y_train, np.ndarray):
                y_train = torch.from_numpy(y_train)
            if isinstance(y_test, np.ndarray):
                y_test = torch.from_numpy(y_test)

            # create validation TensorDataset from x_train and x_test (dictionary of tensors)
            # extract tensors from the dictionary to get list of tensors
            # create a TensorDataset using the unpacked tensors
            train = TensorDataset(*list(x_train.values()), y_train)
            test = TensorDataset(*list(x_test.values()), y_test)

            # create DataLoader objects for train and test sets
            train_loader = DataLoader(dataset=train, **data_loader_args)
            test_loader = DataLoader(dataset=test, **data_loader_args)

            return train_loader, valid_loader, test_loader

        # if only one key in x_data, then just return its only value rather than a dictionary
        if len(self.x_data) == 1:
            x_train = next(iter(x_train.values()))
            x_valid = next(iter(x_valid.values())) if x_valid is not None else None
            x_test = next(iter(x_test.values()))

        return (
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            y_test,
        )


class Folds:
    """
    Class to split the data into different folds based on groups

    Parameters
    ----------
    x_data : np.array | torch.Tensor | dict[str, np.array | torch.Tensor]
        Features for prediction. This can be a numpy array or torch tensor
        of shape (n_samples, n_features), or a dictionary where the keys
        are the names of the features and the values are numpy arrays or
        torch tensors of shape (n_samples, n_features).
    y_data : np.array | torch.Tensor
        Variable to predict.
    groups : np.array | torch.Tensor | None, optional
        Groups to split by, default None. If None is passed, then does standard KFold,
        otherwise implements GroupShuffleSplit (if shuffle is True),
        or GroupKFold (if shuffle is False).
    n_splits : int, optional
        Number of splits / folds, by default 5.
    valid_size : float | None, optional
        Proportion of training data to use as validation data, by default 0.33.
        If None, will not create a validation set.
    indices : tuple[tuple[Iterable[int], Iterable[int] | None, Iterable[int]]] | None, optional
        Tuple of length n_splits where each item is also a tuple containing the
        train, validation, test indices to use for each fold. If passed, will
        split the data according to these indices rather than splitting
        it within the method using the train_size and valid_size provided.
        For each item in the tuple, the first item in the tuple should
        be the indices for the training set, second item should be the indices
        for the validaton set (this could be None if no validation set is required),
        and third item should be indices for the test set.
    shuffle : bool, optional
        Whether or not to shuffle the dataset, by default False.
    random_state : int, optional
        Seed number, by default 42.
        This is ignored if indices are passed.

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

    def __init__(
        self,
        x_data: np.array | torch.Tensor | dict[str, np.array | torch.Tensor],
        y_data: np.array | torch.Tensor,
        groups: np.array | torch.Tensor | None = None,
        n_splits: int = 5,
        valid_size: float | None = 0.33,
        indices: tuple[tuple[Iterable[int], Iterable[int] | None, Iterable[int]]]
        | None = None,
        shuffle: bool = False,
        random_state: int = 42,
    ):
        if n_splits < 2:
            msg = "n_splits should be at least 2."
            raise ValueError(msg)

        if not isinstance(x_data, dict):
            # make x_data a dictionary with key "x_data"
            x_data = {"x_data": x_data}

        # iterate through the values and check they are of the correct shape
        for key, value in x_data.items():
            if value.shape[0] != y_data.shape[0]:
                msg = (
                    f"{key} and y_data do not have compatible shapes "
                    "(need to have same number of samples)."
                )
                raise ValueError(msg)

        # if groups are passed, check that groups is of the correct shape
        if groups is not None and y_data.shape[0] != len(groups):
            msg = (
                "x_data and groups do not have compatible shapes "
                "(need to have same number of samples)."
            )
            raise ValueError(msg)

        # check valid_size is between 0 and 1
        if valid_size is not None and ((valid_size < 0) or (valid_size > 1)):
            msg = "valid_size must be between 0 and 1."
            raise ValueError(msg)

        self.x_data: dict[str, np.array | torch.Tensor] = x_data
        self.y_data: np.array | torch.Tensor = y_data
        self.groups: np.array | torch.Tensor | None = groups
        self.n_splits: int = n_splits
        self.shuffle: bool | None = shuffle
        self.random_state: int | None = random_state

        if indices is not None:
            self.shuffle = False
            self.random_state = None

            # indices are provided, so use these to split the dataset
            # check that indices are a tuple of length k
            msg = "if indices are provided, it must be a tuple of length n_splits."
            if not isinstance(indices, tuple):
                raise TypeError(msg)
            if len(indices) != self.n_splits:
                raise ValueError(msg)

            for k in range(self.n_splits):
                # check that indices[k] is a tuple of length 3
                msg = f"each item in indices must be a tuple of length 3: fold {k} is not."
                if not isinstance(indices[k], tuple):
                    raise TypeError(msg)
                if len(indices[k]) != 3:
                    raise ValueError(msg)

                # check that the indices passed in are within range
                for i in range(len(indices[k])):
                    if (indices[k][i] is not None) and not all(
                        j in list(range(len(y_data))) for j in indices[k][i]
                    ):
                        problem_set = (
                            "train" if i == 0 else "valid" if i == 1 else "test"
                        )
                        msg = (
                            f"in the {problem_set} indices, for fold {k}, "
                            "some of the indices will be out of range."
                        )
                        raise IndexError(msg)

            # all checks have passed - store indices
            self.fold_indices = indices
        else:
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

            # obtain fold indices
            self.fold_indices = list(
                self.fold.split(X=next(iter(self.x_data.values())), groups=self.groups)
            )

            # make the validation sets within the indices
            for k in range(self.n_splits):
                train_index = self.fold_indices[k][0].tolist()
                test_index = self.fold_indices[k][1].tolist()

                if valid_size is not None:
                    # further split the train set into a train, valid set
                    train_index, valid_index = train_test_split(
                        train_index,
                        test_size=valid_size,
                        shuffle=self.shuffle,
                        random_state=self.random_state,
                    )
                else:
                    valid_index = None

                # store indices
                self.fold_indices[k] = (train_index, valid_index, test_index)

            self.fold_indices = tuple(self.fold_indices)

    def get_splits(
        self,
        fold_index: int,
        as_DataLoader: bool = False,
        data_loader_args: dict | None = None,
    ) -> (
        tuple[DataLoader, DataLoader, DataLoader]
        | tuple[
            np.array | torch.Tensor,
            np.array | torch.Tensor,
            np.array | torch.Tensor,
            np.array | torch.Tensor,
            np.array | torch.Tensor,
            np.array | torch.Tensor,
        ]
    ):
        """
        Obtains the data from a particular fold

        Parameters
        ----------
        fold_index : int
            Which fold to obtain data for
        as_DataLoader : bool, optional
            Whether or not to return as `torch.utils.data.dataloader.DataLoader` objects
            ready to be passed into PyTorch model, by default False.
        data_loader_args : dict | None, optional
            Any keywords to be passed in obtaining the
            `torch.utils.data.dataloader.DataLoader` object,
            by default ``{"batch_size": 64, "shuffle": True}``.

        Returns
        -------
        tuple[DataLoader] | tuple[np.array | torch.Tensor | dict[str, np.array | torch.Tensor]]
            If `as_DataLoader` is True, return tuple of `torch.utils.data.dataloader.DataLoader` objects:
                - First element is training dataset
                - Second element is validation dataset
                - Third element is testing dataset

            If `as_DataLoader` is False, returns tuple:
                - First element is features (which is either an array/tensor or dictionary) for training dataset
                - Second element is labels for training dataset as a array/tensor
                - Third element is features (which is either an array/tensor or dictionary) for validation dataset
                - Fourth element is labels for validation dataset as a array/tensor
                - Fifth element is features (which is either an array/tensor or dictionary) for testing dataset
                - Sixth element is labels for testing dataset as a array/tensor

        Raises
        ------
        ValueError
            if the requested `fold_index` is not valid (out of range).
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
        valid_index = self.fold_indices[fold_index][1]
        test_index = self.fold_indices[fold_index][2]

        # obtain validation set
        if valid_index is not None:
            x_valid = {key: value[valid_index] for key, value in self.x_data.items()}
            y_valid = self.y_data[valid_index]
        else:
            x_valid = None
            y_valid = None

        # obtain training set
        x_train = {key: value[train_index] for key, value in self.x_data.items()}
        y_train = self.y_data[train_index]

        # obtain test set
        x_test = {key: value[test_index] for key, value in self.x_data.items()}
        y_test = self.y_data[test_index]

        if as_DataLoader:
            # return datasets as DataLoader objects if requested
            if x_valid is not None:
                # make sure that each value in x_valid is a torch tensor
                for key, value in x_valid.items():
                    if isinstance(value, np.ndarray):
                        x_valid[key] = torch.from_numpy(value)
                # make sure that y_valid is a torch tensor
                if isinstance(y_valid, np.ndarray):
                    y_valid = torch.from_numpy(y_valid)

                # create validation TensorDataset from x_valid (a dictionary) and y_valid
                # extract tensors from the dictionary to get list of tensors
                # create a TensorDataset using the unpacked tensors
                valid = TensorDataset(*list(x_valid.values()), y_valid)
                valid_loader = DataLoader(dataset=valid, **data_loader_args)
            else:
                valid_loader = None

            # make sure that each value in x_train and x_test is a torch tensor
            for key, value in x_train.items():
                if isinstance(value, np.ndarray):
                    x_train[key] = torch.from_numpy(value)
            for key, value in x_test.items():
                if isinstance(value, np.ndarray):
                    x_test[key] = torch.from_numpy(value)

            # make sure that y_train and y_test are torch tensors
            if isinstance(y_train, np.ndarray):
                y_train = torch.from_numpy(y_train)
            if isinstance(y_test, np.ndarray):
                y_test = torch.from_numpy(y_test)

            # create validation TensorDataset from x_train and x_test (dictionary of tensors)
            # extract tensors from the dictionary to get list of tensors
            # create a TensorDataset using the unpacked tensors
            train = TensorDataset(*list(x_train.values()), y_train)
            test = TensorDataset(*list(x_test.values()), y_test)

            # create DataLoader objects for train and test sets
            train_loader = DataLoader(dataset=train, **data_loader_args)
            test_loader = DataLoader(dataset=test, **data_loader_args)

            return train_loader, valid_loader, test_loader

        # if only one key in x_data, then just return its only value rather than a dictionary
        if len(self.x_data) == 1:
            x_train = next(iter(x_train.values()))
            x_valid = next(iter(x_valid.values())) if x_valid is not None else None
            x_test = next(iter(x_test.values()))

        return (
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            y_test,
        )


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
