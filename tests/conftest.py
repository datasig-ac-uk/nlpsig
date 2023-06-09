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
