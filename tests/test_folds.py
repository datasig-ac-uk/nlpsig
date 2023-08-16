from __future__ import annotations

import numpy as np
import pytest
import regex as re
from torch.utils.data.dataloader import DataLoader

from nlpsig.classification_utils import Folds


def test_folds_default_init(X_data, y_data):
    # test default initialisation
    # (5 Folds by default, no groups passed, no shuffling so random_state is None)
    ds = Folds(x_data=X_data, y_data=y_data)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    assert type(ds.fold_indices) == tuple
    # check that 5 folds have been created
    assert len(ds.fold_indices) == 5
    # check that each fold is a tuple of length three
    for fold in ds.fold_indices:
        assert type(fold) == tuple
        assert len(fold) == 3
    # default no shuffling and random_state set to None
    assert ds.shuffle is False
    assert ds.random_state is None


def test_folds_default_shuffle(X_data, y_data):
    # test default initialisation with shuffling
    ds = Folds(x_data=X_data, y_data=y_data, shuffle=True, random_state=42)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    assert type(ds.fold_indices) == tuple
    # check that 5 folds have been created
    assert len(ds.fold_indices) == 5
    # check that each fold is a tuple of length three
    for fold in ds.fold_indices:
        assert type(fold) == tuple
        assert len(fold) == 3
    # check shuffling and random_state has been set
    assert ds.shuffle is True
    assert ds.random_state == 42

    # check that the seed was set correctly by checking that the indices are the same
    ds_2 = Folds(x_data=X_data, y_data=y_data, shuffle=True, random_state=42)
    assert ds_2.x_data is ds.x_data
    assert ds_2.y_data is ds.y_data
    assert ds_2.groups is None
    # check indices are the same
    assert ds.fold_indices == ds_2.fold_indices
    # check shuffling and random_state has been set
    assert ds_2.shuffle is True
    assert ds_2.random_state == 42


def test_folds_no_validation(X_data, y_data):
    # test initialisation with no validation set
    ds = Folds(x_data=X_data, y_data=y_data, valid_size=None)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    # check that 5 folds have been created
    assert len(ds.fold_indices) == 5
    # check that each fold is a tuple of length three
    for fold in ds.fold_indices:
        assert type(fold) == tuple
        assert len(fold) == 3
        # check that middle element is None
        assert fold[1] is None
    # default no shuffling and random_state set to None
    assert ds.shuffle is False
    assert ds.random_state is None


def test_folds_no_validation_shuffle(X_data, y_data):
    # test initialisation with no validation set and shuffling
    ds = Folds(
        x_data=X_data, y_data=y_data, valid_size=None, shuffle=True, random_state=42
    )
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    # check that 5 folds have been created
    assert len(ds.fold_indices) == 5
    # check that each fold is a tuple of length three
    for fold in ds.fold_indices:
        assert type(fold) == tuple
        assert len(fold) == 3
        # check that the validation set (the middle item in tuple) is None
        assert fold[1] is None
    # check shuffling and random_state has been set
    assert ds.shuffle is True
    assert ds.random_state == 42

    # check that the seed was set correctly by checking that the indices are the same
    ds_2 = Folds(
        x_data=X_data, y_data=y_data, valid_size=None, shuffle=True, random_state=42
    )
    assert ds_2.x_data is ds.x_data
    assert ds_2.y_data is ds.y_data
    assert ds_2.groups is None
    # check indices are the same
    assert ds.fold_indices == ds_2.fold_indices
    # check shuffling and random_state has been set
    assert ds_2.shuffle is True
    assert ds_2.random_state == 42


def test_folds_incompatible_shape():
    # test error handling when the input data has incompatible shapes
    msg = "x_data and y_data do not have compatible shapes (need to have same number of samples)."

    # test case where x_data is 1D array
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        Folds(x_data=np.array([1, 2, 3]), y_data=np.array([1, 2]))

    # test case where x_data is 2D array
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        Folds(x_data=np.array([[1, 2], [3, 4], [5, 6]]), y_data=np.array([1, 2]))


def test_folds_valid_size_incorrect_range(X_data, y_data):
    # test error handling when valid_size is not in [0,1]
    msg = "valid_size must be between 0 and 1."

    # valid_size too large
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        Folds(x_data=X_data, y_data=y_data, valid_size=1.1)

    # valid_size too small
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        Folds(x_data=X_data, y_data=y_data, valid_size=-0.1)


def test_folds_indices(X_data, y_data, three_folds):
    # test initialisation with indices
    ds = Folds(x_data=X_data, y_data=y_data, n_splits=3, indices=three_folds)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    # check folds has been loaded in correctly
    assert type(ds.fold_indices) == tuple
    assert len(ds.fold_indices) == 3
    # check they are actually the fold indices passed in
    assert ds.fold_indices == three_folds
    # check shuffle is set to False and random_state is None
    assert ds.shuffle is False
    assert ds.random_state is None


def test_folds_indices_no_validation(X_data, y_data, three_folds_no_validation):
    # test initialisation with indices with no validation set
    ds = Folds(
        x_data=X_data, y_data=y_data, n_splits=3, indices=three_folds_no_validation
    )
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is None
    # check folds has been loaded in correctly
    assert type(ds.fold_indices) == tuple
    assert len(ds.fold_indices) == 3
    # check they are actually the fold indices passed in
    assert ds.fold_indices == three_folds_no_validation
    # check shuffle is set to False and random_state is None
    assert ds.shuffle is False
    assert ds.random_state is None


def test_folds_indices_incorrect_num_splits(X_data, y_data, three_folds):
    # test error handling when the fold indices provided aren't a tuple of length n_splits
    msg = "if indices are provided, it must be a tuple of length n_splits."

    # require 5 splits but only 3 provided
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        Folds(x_data=X_data, y_data=y_data, n_splits=5, indices=three_folds)


def test_folds_indices_incorrect_type(X_data, y_data, three_folds):
    # test error handling when the fold indices aren't a tuple
    msg = "if indices are provided, it must be a tuple of length n_splits."

    # incorrect type (i.e. not typle)
    with pytest.raises(
        TypeError,
        match=re.escape(msg),
    ):
        Folds(x_data=X_data, y_data=y_data, indices=list(three_folds))


def test_folds_indices_incorrect_elements(X_data, y_data, three_folds):
    # test error handling when some of the items in the tuple aren't tuples of length 3
    # the error handling should say which fold is incorrect (here is the 2nd one, at index 1)
    msg = "each item in indices must be a tuple of length 3: fold 1 is not."

    # test error catching when one of the folds is a tuple but not length 3
    incorrect_three_folds_length = (
        three_folds[0],
        (three_folds[1][0], three_folds[1][1]),
        three_folds[2],
    )
    with pytest.raises(
        ValueError,
        match=re.escape(msg),
    ):
        Folds(
            x_data=X_data,
            y_data=y_data,
            n_splits=3,
            indices=incorrect_three_folds_length,
        )

    # test error catching when one of the folds is not a tuple
    incorrect_three_folds_type = (three_folds[0], list(three_folds[1]), three_folds[2])
    with pytest.raises(
        TypeError,
        match=re.escape(msg),
    ):
        Folds(
            x_data=X_data, y_data=y_data, n_splits=3, indices=incorrect_three_folds_type
        )


def test_folds_indices_out_of_range(X_data, y_data, three_folds):
    # test error handling if any of the indices in the folds that are passed in have
    # index values that are out of range (e.g. has values greater than the number of samples)
    # the error handling should say which fold is incorrect (here is the 2nd one, at index 1)

    # train indices include index out of range
    train_index_wrong = (
        three_folds[0],
        (three_folds[1][0] + [len(y_data) + 1], three_folds[1][1], three_folds[1][2]),
        three_folds[2],
    )
    with pytest.raises(
        IndexError,
        match=re.escape(
            "in the train indices, for fold 1, some of the indices will be out of range."
        ),
    ):
        Folds(x_data=X_data, y_data=y_data, n_splits=3, indices=train_index_wrong)

    # valid indices include index out of range
    valid_index_wrong = (
        three_folds[0],
        (three_folds[1][0], three_folds[1][1] + [len(y_data) + 1], three_folds[1][2]),
        three_folds[2],
    )
    with pytest.raises(
        IndexError,
        match=re.escape(
            "in the valid indices, for fold 1, some of the indices will be out of range."
        ),
    ):
        Folds(x_data=X_data, y_data=y_data, n_splits=3, indices=valid_index_wrong)

    # test indices include index out of range
    test_index_wrong = (
        three_folds[0],
        (three_folds[1][0], three_folds[1][1], three_folds[1][2] + [len(y_data) + 1]),
        three_folds[2],
    )
    with pytest.raises(
        IndexError,
        match=re.escape(
            "in the test indices, for fold 1, some of the indices will be out of range."
        ),
    ):
        Folds(x_data=X_data, y_data=y_data, n_splits=3, indices=test_index_wrong)


def test_datasplit_groups(X_data, y_data, groups):
    # test initialisation with groups
    ds = Folds(x_data=X_data, y_data=y_data, groups=groups)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is groups
    assert type(ds.fold_indices) == tuple
    # check that 5 folds have been created
    assert len(ds.fold_indices) == 5
    # check that each fold is a tuple of length three
    for fold in ds.fold_indices:
        assert type(fold) == tuple
        assert len(fold) == 3
    # default no shuffling and random_state set to None
    assert ds.shuffle is False
    assert ds.random_state is None


def test_datasplit_groups_shuffle(X_data, y_data, groups):
    # test initialisation with groups and shuffle
    ds = Folds(
        x_data=X_data, y_data=y_data, groups=groups, shuffle=True, random_state=50
    )
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is groups
    assert type(ds.fold_indices) == tuple
    # check that 5 folds have been created
    assert len(ds.fold_indices) == 5
    # check that each fold is a tuple of length three
    for fold in ds.fold_indices:
        assert type(fold) == tuple
        assert len(fold) == 3
    # check shuffling and random_state has been set
    assert ds.shuffle is True
    assert ds.random_state == 50


def test_folds_groups_no_validation(X_data, y_data, groups):
    # test initialisation with groups and no validation set
    ds = Folds(x_data=X_data, y_data=y_data, groups=groups, valid_size=None)
    assert ds.x_data is X_data
    assert ds.y_data is y_data
    assert ds.groups is groups
    # check that 5 folds have been created
    assert len(ds.fold_indices) == 5
    # check that each fold is a tuple of length three
    for fold in ds.fold_indices:
        assert type(fold) == tuple
        assert len(fold) == 3
        # check that middle element is None
        assert fold[1] is None
    # default no shuffling and random_state set to None
    assert ds.shuffle is False
    assert ds.random_state is None


def test_folds_get_splits(X_data, y_data):
    # test get_splits functionality for default initialisation of 3 folds
    ds = Folds(x_data=X_data, y_data=y_data, n_splits=3)

    # getting data splits as DataLoaders
    for k in range(3):
        splits_as_dl = ds.get_splits(as_DataLoader=True, fold_index=k)
        assert type(splits_as_dl) == tuple
        assert type(splits_as_dl[0]) == DataLoader
        assert type(splits_as_dl[1]) == DataLoader
        assert type(splits_as_dl[2]) == DataLoader

    # getting data splits as numpy arrays
    for k in range(3):
        splits_as_dl = ds.get_splits(as_DataLoader=False, fold_index=k)
        assert type(splits_as_dl) == tuple
        assert type(splits_as_dl[0]) == np.ndarray
        assert type(splits_as_dl[1]) == np.ndarray
        assert type(splits_as_dl[2]) == np.ndarray
        assert type(splits_as_dl[3]) == np.ndarray
        assert type(splits_as_dl[4]) == np.ndarray
        assert type(splits_as_dl[5]) == np.ndarray


def test_folds_get_splits_no_validation(X_data, y_data):
    # test get_splits functionality for initialisation of 3 folds with no validation set
    ds = Folds(x_data=X_data, y_data=y_data, valid_size=None, n_splits=3)

    # getting data splits as DataLoaders
    for k in range(3):
        splits_as_dl = ds.get_splits(as_DataLoader=True, fold_index=k)
        assert type(splits_as_dl) == tuple
        assert type(splits_as_dl[0]) == DataLoader
        assert splits_as_dl[1] is None
        assert type(splits_as_dl[2]) == DataLoader

    # getting data splits as numpy arrays
    for k in range(3):
        splits_as_dl = ds.get_splits(as_DataLoader=False, fold_index=k)
        assert type(splits_as_dl) == tuple
        assert type(splits_as_dl[0]) == np.ndarray
        assert type(splits_as_dl[1]) == np.ndarray
        assert splits_as_dl[2] is None
        assert splits_as_dl[3] is None
        assert type(splits_as_dl[4]) == np.ndarray
        assert type(splits_as_dl[5]) == np.ndarray


def test_folds_get_splits_groups(X_data, y_data, groups):
    # test get_splits functionality for initialisation of 3 folds and groups
    ds = Folds(x_data=X_data, y_data=y_data, groups=groups)

    # getting data splits as DataLoaders
    for k in range(3):
        splits_as_dl = ds.get_splits(as_DataLoader=True, fold_index=k)
        assert type(splits_as_dl) == tuple
        assert type(splits_as_dl[0]) == DataLoader
        assert type(splits_as_dl[1]) == DataLoader
        assert type(splits_as_dl[2]) == DataLoader

    # getting data splits as numpy arrays
    for k in range(3):
        splits_as_dl = ds.get_splits(as_DataLoader=False, fold_index=k)
        assert type(splits_as_dl) == tuple
        assert type(splits_as_dl[0]) == np.ndarray
        assert type(splits_as_dl[1]) == np.ndarray
        assert type(splits_as_dl[2]) == np.ndarray
        assert type(splits_as_dl[3]) == np.ndarray
        assert type(splits_as_dl[4]) == np.ndarray
        assert type(splits_as_dl[5]) == np.ndarray


def test_folds_get_splits_groups_no_validation(X_data, y_data, groups):
    # test get_splits functionality for initialisation of 3 folds and groups with no validation set
    ds = Folds(x_data=X_data, y_data=y_data, groups=groups, valid_size=None)

    # getting data splits as DataLoaders
    for k in range(3):
        splits_as_dl = ds.get_splits(as_DataLoader=True, fold_index=k)
        assert type(splits_as_dl) == tuple
        assert type(splits_as_dl[0]) == DataLoader
        assert splits_as_dl[1] is None
        assert type(splits_as_dl[2]) == DataLoader

    # getting data splits as numpy arrays
    for k in range(3):
        splits_as_dl = ds.get_splits(as_DataLoader=False, fold_index=k)
        assert type(splits_as_dl) == tuple
        assert type(splits_as_dl[0]) == np.ndarray
        assert type(splits_as_dl[1]) == np.ndarray
        assert splits_as_dl[2] is None
        assert splits_as_dl[3] is None
        assert type(splits_as_dl[4]) == np.ndarray
        assert type(splits_as_dl[5]) == np.ndarray
