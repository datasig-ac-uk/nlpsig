import random

import numpy as np
import pandas as pd

from nlpsig.data_preparation import PrepareData

random.seed(2022)
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
test_df = pd.DataFrame(
    {
        "text": [f"text_{i}" for i in range(n_entries)],
        "id_col": [random.randint(1, 10) for i in range(n_entries)],
        "label_col": [random.randint(0, 4) for i in range(n_entries)],
        "datetime": [random.choice(list_datetimes) for i in range(n_entries)],
    }
)
emb = np.random.rand(1000, 300)
reduced_emb = np.random.rand(1000, 10)


def test_PrepareData_init():
    # test default initialisation
    obj = PrepareData(original_df=test_df, embeddings=emb)
    assert obj.id_column == "dummy_id"
    assert obj.label_column is None
    assert obj.embeddings_reduced is None


def test_PrepareData_obtain_column_names():
    # test default initialisation
    obj = PrepareData(original_df=test_df, embeddings=emb)
    assert obj._obtain_colnames(embeddings="full") == [f"e{i+1}" for i in range(300)]
