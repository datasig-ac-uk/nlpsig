from __future__ import annotations

import numpy as np
import pytest
import regex as re
from torch.utils.data.dataloader import DataLoader

from nlpsig.classification_utils import DataSplits


def test_datasplits_default_init(X_data, y_data):
    # test default initialisation
    # (no groups passed, no shuffling so random_state gets is None)
    ds = DataSplits(x_data=X_data, y_data=y_data)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    # check data split has been made correctly
    assert type(ds.indices) == tuple
    assert len(ds.indices) == 3
    # default no shuffling and random_state set to None
    assert ds.shuffle is False
    assert ds.random_state is None


def test_datasplits_default_shuffle(X_data, y_data):
    # test default initialisation with shuffling
    ds = DataSplits(x_data=X_data, y_data=y_data, shuffle=True, random_state=42)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    # check data split has been made correctly
    assert type(ds.indices) == tuple
    assert len(ds.indices) == 3
    # check shuffling and random_state has been set
    assert ds.shuffle is True
    assert ds.random_state == 42

    # check that the seed was set correctly by checking that the indices are the same
    ds_2 = DataSplits(x_data=X_data, y_data=y_data, shuffle=True, random_state=42)
    assert ds_2.x_data is ds.x_data
    assert ds_2.y_data is ds.y_data
    assert ds_2.groups is None
    # check indices are the same
    assert ds.indices == ds_2.indices
    # check shuffling and random_state has been set
    assert ds_2.shuffle is True
    assert ds_2.random_state == 42


def test_datasplits_no_validation(X_data, y_data):
    # test initialisation with no validation set
    ds = DataSplits(x_data=X_data, y_data=y_data, valid_size=None)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    # check data split has been made correctly
    assert type(ds.indices) == tuple
    assert len(ds.indices) == 3
    # check that the validation set (the middle item in tuple) is None
    assert ds.indices[1] is None
    # default no shuffling and random_state set to None
    assert ds.shuffle is False
    assert ds.random_state is None


def test_datasplits_no_validation_shuffle(X_data, y_data):
    # test initialisation with no validation set but with shuffling
    ds = DataSplits(
        x_data=X_data, y_data=y_data, valid_size=None, shuffle=True, random_state=42
    )
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    # check data split has been made correctly
    assert type(ds.indices) == tuple
    assert len(ds.indices) == 3
    # check that the validation set (the middle item in tuple) is None
    assert ds.indices[1] is None
    # check shuffling and random_state has been set
    assert ds.shuffle is True
    assert ds.random_state == 42

    # check that the seed was set correctly by checking that the indices are the same
    ds_2 = DataSplits(
        x_data=X_data, y_data=y_data, valid_size=None, shuffle=True, random_state=42
    )
    assert ds_2.x_data is ds.x_data
    assert ds_2.y_data is ds.y_data
    assert ds_2.groups is None
    # check indices are the same
    assert ds.indices == ds_2.indices
    assert ds_2.indices[1] is None
    # check shuffling and random_state has been set
    assert ds_2.shuffle is True
    assert ds_2.random_state == 42


def test_datasplits_incompatible_shape():
    # test error handling when the input data has incompatible shapes
    msg = "x_data and y_data do not have compatible shapes (need to have same number of samples)."

    # test case where x_data is 1D array
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        DataSplits(x_data=np.array([1, 2, 3]), y_data=np.array([1, 2]))

    # test case where x_data is 2D array
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        DataSplits(x_data=np.array([[1, 2], [3, 4], [5, 6]]), y_data=np.array([1, 2]))


def test_datasplits_train_size_incorrect_range(X_data, y_data):
    # test error handling when train_size is not in [0,1]
    msg = "train_size must be between 0 and 1."

    # train_size too large
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        DataSplits(x_data=X_data, y_data=y_data, train_size=1.1)

    # train_size too small
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        DataSplits(x_data=X_data, y_data=y_data, train_size=-0.1)


def test_datasplits_valid_size_incorrect_range(X_data, y_data):
    # test error handling when valid_size is not in [0,1]
    msg = "valid_size must be between 0 and 1."

    # valid_size too large
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        DataSplits(x_data=X_data, y_data=y_data, valid_size=1.1)

    # valid_size too small
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        DataSplits(x_data=X_data, y_data=y_data, valid_size=-0.1)


def test_datasplits_indices(X_data, y_data, indices):
    # test initialisation with indices
    ds = DataSplits(x_data=X_data, y_data=y_data, indices=indices)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    # check data split has been loaded in correctly
    assert type(ds.indices) == tuple
    assert len(ds.indices) == 3
    # check they are actually the indices passed in
    assert ds.indices == indices
    # check shuffle is set to False and random_state is None
    assert ds.shuffle is False
    assert ds.random_state is None


def test_datasplits_indices_no_validation(X_data, y_data, indices_no_validation):
    # test initialisation with indices with no validation set
    ds = DataSplits(x_data=X_data, y_data=y_data, indices=indices_no_validation)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    # check data split has been loaded in correctly
    assert type(ds.indices) == tuple
    assert len(ds.indices) == 3
    # check they are actually the indices passed in
    assert ds.indices == indices_no_validation
    # check shuffle is set to False and random_state is None
    assert ds.shuffle is False
    assert ds.random_state is None


def test_datasplits_indices_incorrect_type_or_length(X_data, y_data, indices):
    # test error handling that indices must be a tuple of length 3
    msg = "if indices are provided, it must be a tuple of length 3."

    # test error handling for incorrect type (i.e. not typle)
    with pytest.raises(
        TypeError,
        match=re.escape(msg),
    ):
        DataSplits(x_data=X_data, y_data=y_data, indices=list(indices))

    # test error handling for incorrect length (i.e tuple of length 2)
    indices_incorrect_length = (indices[0] + indices[1], indices[2])
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        DataSplits(x_data=X_data, y_data=y_data, indices=indices_incorrect_length)


def test_datasplits_indices_out_of_range(X_data, y_data, indices):
    # test error handling if any of the indices that are passed in have
    # index values that are out of range (e.g. has values greater than the number of samples)

    # train indices include index out of range
    train_index_wrong = (indices[0] + [len(y_data) + 1], indices[1], indices[2])
    with pytest.raises(
        IndexError,
        match=re.escape(
            "in the train indices, some of the indices will be out of range."
        ),
    ):
        DataSplits(x_data=X_data, y_data=y_data, indices=train_index_wrong)

    # valid indices include index out of range
    valid_index_wrong = (indices[0], indices[1] + [len(y_data) + 1], indices[2])
    with pytest.raises(
        IndexError,
        match=re.escape(
            "in the valid indices, some of the indices will be out of range."
        ),
    ):
        DataSplits(x_data=X_data, y_data=y_data, indices=valid_index_wrong)

    # test indices include index out of range
    test_index_wrong = (indices[0], indices[1], indices[2] + [len(y_data) + 1])
    with pytest.raises(
        IndexError,
        match=re.escape(
            "in the test indices, some of the indices will be out of range."
        ),
    ):
        DataSplits(x_data=X_data, y_data=y_data, indices=test_index_wrong)


def test_datasplit_groups(X_data, y_data, groups):
    # test initialisation with groups
    ds = DataSplits(x_data=X_data, y_data=y_data, groups=groups)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is groups
    # check data split has been made correctly
    assert type(ds.indices) == tuple
    assert len(ds.indices) == 3
    # if groups are passed, then we shuffle by default
    assert ds.shuffle is True
    assert ds.random_state == 42  # default seed


def test_datasplit_groups_shuffle(X_data, y_data, groups):
    # test initialisation with groups and shuffle set to False
    # if shuffle is False, it will get set to True as we use GroupShuffleSplit
    ds = DataSplits(
        x_data=X_data, y_data=y_data, groups=groups, shuffle=False, random_state=50
    )
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is groups
    # check data split has been made correctly
    assert type(ds.indices) == tuple
    assert len(ds.indices) == 3
    # if groups are passed, then we shuffle by default
    assert ds.shuffle is True
    assert ds.random_state == 50


def test_datasplits_groups_no_validation(X_data, y_data, groups):
    # test initialisation with groups with no validation set
    ds = DataSplits(x_data=X_data, y_data=y_data, groups=groups, valid_size=None)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is groups
    # check data split has been made correctly
    assert type(ds.indices) == tuple
    assert len(ds.indices) == 3
    # check that the validation set (the middle item in tuple) is None
    assert ds.indices[1] is None
    assert ds.shuffle is True
    assert ds.random_state == 42  # default seed


def test_datasplits_get_splits(X_data, y_data):
    # test get_splits functionality for default initialisation of DataSplits
    ds = DataSplits(x_data=X_data, y_data=y_data)

    # getting data splits as DataLoaders
    splits_as_dl = ds.get_splits(as_DataLoader=True)
    assert type(splits_as_dl) == tuple
    assert type(splits_as_dl[0]) == DataLoader
    assert type(splits_as_dl[1]) == DataLoader
    assert type(splits_as_dl[2]) == DataLoader

    # getting data splits as numpy arrays
    splits_as_dl = ds.get_splits(as_DataLoader=False)
    assert type(splits_as_dl) == tuple
    assert type(splits_as_dl[0]) == np.ndarray
    assert type(splits_as_dl[1]) == np.ndarray
    assert type(splits_as_dl[2]) == np.ndarray
    assert type(splits_as_dl[3]) == np.ndarray
    assert type(splits_as_dl[4]) == np.ndarray
    assert type(splits_as_dl[5]) == np.ndarray


def test_datasplits_get_splits_no_validation(X_data, y_data):
    # test get_splits functionality for default initialisation of DataSplits
    # with no validation set
    ds = DataSplits(x_data=X_data, y_data=y_data, valid_size=None)

    # getting data splits as DataLoaders
    splits_as_dl = ds.get_splits(as_DataLoader=True)
    assert type(splits_as_dl) == tuple
    assert type(splits_as_dl[0]) == DataLoader
    assert splits_as_dl[1] is None
    assert type(splits_as_dl[2]) == DataLoader

    # getting data splits as numpy arrays
    splits_as_dl = ds.get_splits(as_DataLoader=False)
    assert type(splits_as_dl) == tuple
    assert type(splits_as_dl[0]) == np.ndarray
    assert type(splits_as_dl[1]) == np.ndarray
    assert splits_as_dl[2] is None
    assert splits_as_dl[3] is None
    assert type(splits_as_dl[4]) == np.ndarray
    assert type(splits_as_dl[5]) == np.ndarray


def test_datasplits_get_splits_groups(X_data, y_data, groups):
    # test get_splits functionality for default initialisation of DataSplits with groups
    ds = DataSplits(x_data=X_data, y_data=y_data, groups=groups)

    # getting data splits as DataLoaders
    splits_as_dl = ds.get_splits(as_DataLoader=True)
    assert type(splits_as_dl) == tuple
    assert type(splits_as_dl[0]) == DataLoader
    assert type(splits_as_dl[1]) == DataLoader
    assert type(splits_as_dl[2]) == DataLoader

    # getting data splits as numpy arrays
    splits_as_dl = ds.get_splits(as_DataLoader=False)
    assert type(splits_as_dl) == tuple
    assert type(splits_as_dl[0]) == np.ndarray
    assert type(splits_as_dl[1]) == np.ndarray
    assert type(splits_as_dl[2]) == np.ndarray
    assert type(splits_as_dl[3]) == np.ndarray
    assert type(splits_as_dl[4]) == np.ndarray
    assert type(splits_as_dl[5]) == np.ndarray


def test_datasplits_get_splits_groups_no_validation(X_data, y_data, groups):
    # test get_splits functionality for default initialisation of DataSplits
    # with groups and no validation set
    ds = DataSplits(x_data=X_data, y_data=y_data, groups=groups, valid_size=None)

    # getting data splits as DataLoaders
    splits_as_dl = ds.get_splits(as_DataLoader=True)
    assert type(splits_as_dl) == tuple
    assert type(splits_as_dl[0]) == DataLoader
    assert splits_as_dl[1] is None
    assert type(splits_as_dl[2]) == DataLoader

    # getting data splits as numpy arrays
    splits_as_dl = ds.get_splits(as_DataLoader=False)
    assert type(splits_as_dl) == tuple
    assert type(splits_as_dl[0]) == np.ndarray
    assert type(splits_as_dl[1]) == np.ndarray
    assert splits_as_dl[2] is None
    assert splits_as_dl[3] is None
    assert type(splits_as_dl[4]) == np.ndarray
    assert type(splits_as_dl[5]) == np.ndarray
