import pytest

import numpy as np
import pandas as pd

rng = np.random.default_rng(2022)

@pytest.fixture
def test_df():
    
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
            "id_col": [rng.integers(1, 10) for i in range(n_entries)],
            "label_col": [rng.integers(0, 4) for i in range(n_entries)],
            "datetime": [rng.choice(list_datetimes) for i in range(n_entries)],
        }
    )


@pytest.fixture
def emb():
    return rng.random((1000, 300))
