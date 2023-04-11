from __future__ import annotations

from nlpsig.data_preparation import PrepareData


def test_PrepareData_init(test_df, emb):
    # test default initialisation
    obj = PrepareData(original_df=test_df, embeddings=emb)
    assert obj.id_column == "dummy_id"
    assert obj.label_column is None
    assert obj.embeddings_reduced is None


def test_PrepareData_obtain_column_names(test_df, emb):
    # test default initialisation
    obj = PrepareData(original_df=test_df, embeddings=emb)
    assert obj._obtain_colnames(embeddings="full") == [f"e{i+1}" for i in range(300)]
