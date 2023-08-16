from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

rng = np.random.default_rng(2022)


@pytest.fixture()
def test_df_with_datetime():
    n_entries = 1000
    list_datetimes = pd.to_datetime(
        [
            "2015-01-01 00:00:00",
            "2015-01-01 00:12:00",
            "2015-01-02 00:00:00",
            "2015-01-02 00:12:00",
            "2015-01-03 00:00:00",
            "2015-01-03 00:12:00",
            "2015-01-04 00:00:00",
            "2015-01-04 00:12:00",
            "2015-01-05 00:00:00",
            "2015-01-05 00:12:00",
            "2015-01-06 00:00:00",
            "2015-01-06 00:12:00",
        ]
    )
    return pd.DataFrame(
        {
            "text": [f"text_{i}" for i in range(n_entries)],
            "binary_var": [rng.choice([0, 1]) for i in range(n_entries)],
            "continuous_var": rng.random(n_entries),
            "id_col": [0 for i in range(100)]
            + [rng.integers(1, 5) for i in range(n_entries - 100)],
            "label_col": [rng.integers(0, 4) for i in range(n_entries)],
            "datetime": [rng.choice(list_datetimes) for i in range(n_entries)],
        }
    )


@pytest.fixture()
def test_df_no_time():
    n_entries = 1000
    return pd.DataFrame(
        {
            "text": [f"text_{i}" for i in range(n_entries)],
            "binary_var": [rng.choice([0, 1]) for i in range(n_entries)],
            "continuous_var": rng.random(n_entries),
            "id_col": [0 for i in range(100)]
            + [rng.integers(1, 5) for i in range(n_entries - 100)],
            "label_col": [rng.integers(0, 4) for i in range(n_entries)],
        }
    )


@pytest.fixture()
def test_df_to_pad():
    n_entries = 100
    return pd.DataFrame(
        {
            "text": [f"text_{i}" for i in range(n_entries)],
            "binary_var": [rng.choice([0, 1]) for i in range(n_entries)],
            "continuous_var": rng.random(n_entries),
            "id_col": 0,
            "label_col": [rng.integers(0, 4) for i in range(n_entries)],
        }
    )


@pytest.fixture()
def test_empty_df_to_pad():
    return pd.DataFrame(columns=["text", "id_col", "label_col"])


@pytest.fixture()
def vec_to_standardise():
    return pd.Series([1, 2, 3])


@pytest.fixture()
def emb():
    return rng.random((1000, 1000))


@pytest.fixture()
def emb_reduced():
    return rng.random((1000, 300))


@pytest.fixture()
def emb_pooled():
    return rng.random((5, 1000))


@pytest.fixture()
def emb_1d():
    return rng.random(1000)


@pytest.fixture()
def X_fit():
    return rng.random((300, 50))


@pytest.fixture()
def X_new():
    return rng.random((200, 50))


@pytest.fixture()
def X_data():
    return rng.random((20, 4))


@pytest.fixture()
def y_data():
    return rng.random(20)


@pytest.fixture()
def indices():
    return ([0, 1, 3, 5, 6, 15, 17], [2, 4, 7, 8, 9, 16, 18], [10, 11, 12, 13, 14, 19])


@pytest.fixture()
def indices_no_validation():
    return (
        [0, 1, 3, 5, 6, 15, 17, 2, 4, 7, 8, 9, 16, 18],
        None,
        [10, 11, 12, 13, 14, 19],
    )


@pytest.fixture()
def groups():
    return np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7])


@pytest.fixture()
def three_folds():
    return (
        ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18, 19]),
        ([6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18, 19], [0, 1, 2, 3, 4, 5]),
        ([13, 14, 15, 16, 17, 18, 19], [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12]),
    )


@pytest.fixture()
def three_folds_no_validation():
    return (
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            None,
            [13, 14, 15, 16, 17, 18, 19],
        ),
        (
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            None,
            [0, 1, 2, 3, 4, 5],
        ),
        (
            [13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5],
            None,
            [6, 7, 8, 9, 10, 11, 12],
        ),
    )
