from __future__ import annotations

import pandas as pd
import pytest

from nlpsig.data_preparation import PrepareData


def test_default_initialisation_datetime(
    test_df_with_datetime,
    emb,
):
    # default initialisation
    obj = PrepareData(original_df=test_df_with_datetime, embeddings=emb)
    pd.testing.assert_frame_equal(obj.original_df, test_df_with_datetime)
    assert obj.id_column == "dummy_id"
    assert obj.label_column is None
    assert (obj.embeddings == emb).all()
    assert obj.embeddings_reduced is None
    # check that .df is a data frame and check the shape of it
    assert type(obj.df) == pd.DataFrame
    # number of columns is:
    # original number of columns +
    # number of columns in emb +
    # 3 time features +
    # 1 dummy id column
    assert obj.df.shape == (
        len(obj.original_df.index),
        len(obj.original_df.columns) + emb.shape[1] + 3 + 1,
    )
    assert obj.pooled_embeddings is None
    assert set(obj._time_feature_choices) == {
        "time_encoding",
        "time_diff",
        "timeline_index",
    }
    assert obj.time_features_added
    assert obj.df_padded is None
    assert obj.array_padded is None
    assert obj.pad_method is None


def test_default_initialisation_no_time(
    test_df_no_time,
    emb,
):
    # default initialisation
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb)
    pd.testing.assert_frame_equal(obj.original_df, test_df_no_time)
    assert obj.id_column == "dummy_id"
    assert obj.label_column is None
    assert (obj.embeddings == emb).all()
    assert obj.embeddings_reduced is None
    # check that .df is a data frame and check the shape of it
    assert type(obj.df) == pd.DataFrame
    # number of columns is:
    # original number of columns +
    # number of columns in emb +
    # 1 time feature +
    # 1 dummy id column
    assert obj.df.shape == (
        len(obj.original_df.index),
        len(obj.original_df.columns) + emb.shape[1] + 1 + 1,
    )
    assert obj.pooled_embeddings is None
    assert obj._time_feature_choices == ["timeline_index"]
    assert obj.time_features_added
    assert obj.df_padded is None
    assert obj.array_padded is None
    assert obj.pad_method is None


def test_initialisation_with_reduced_emb_datetime(
    test_df_with_datetime,
    emb,
    emb_reduced,
):
    # initialise with reduced embeddings
    obj = PrepareData(
        original_df=test_df_with_datetime,
        embeddings=emb,
        embeddings_reduced=emb_reduced,
    )
    pd.testing.assert_frame_equal(obj.original_df, test_df_with_datetime)
    assert obj.id_column == "dummy_id"
    assert obj.label_column is None
    assert (obj.embeddings == emb).all()
    assert (obj.embeddings_reduced == emb_reduced).all()
    # check that .df is a data frame and check the shape of it
    assert type(obj.df) == pd.DataFrame
    # number of columns is:
    # original number of columns +
    # number of columns in emb +
    # number of columns in emb_reduced +
    # 3 time features +
    # 1 dummy id column
    assert obj.df.shape == (
        len(obj.original_df.index),
        len(obj.original_df.columns) + emb.shape[1] + emb_reduced.shape[1] + 3 + 1,
    )
    assert obj.pooled_embeddings is None
    assert set(obj._time_feature_choices) == {
        "time_encoding",
        "time_diff",
        "timeline_index",
    }
    assert obj.time_features_added
    assert obj.df_padded is None
    assert obj.array_padded is None
    assert obj.pad_method is None


def test_initialisation_with_reduced_emb_no_time(
    test_df_no_time,
    emb,
    emb_reduced,
):
    # initialise with reduced embeddings
    obj = PrepareData(
        original_df=test_df_no_time, embeddings=emb, embeddings_reduced=emb_reduced
    )
    pd.testing.assert_frame_equal(obj.original_df, test_df_no_time)
    assert obj.id_column == "dummy_id"
    assert obj.label_column is None
    assert (obj.embeddings == emb).all()
    assert (obj.embeddings_reduced == emb_reduced).all()
    # check that .df is a data frame and check the shape of it
    assert type(obj.df) == pd.DataFrame
    # number of columns is:
    # original number of columns +
    # number of columns in emb +
    # number of columns in emb_reduced +
    # 1 time feature +
    # 1 dummy id column
    assert obj.df.shape == (
        len(obj.original_df.index),
        len(obj.original_df.columns) + emb.shape[1] + emb_reduced.shape[1] + 1 + 1,
    )
    assert obj.pooled_embeddings is None
    assert obj._time_feature_choices == ["timeline_index"]
    assert obj.time_features_added
    assert obj.df_padded is None
    assert obj.array_padded is None
    assert obj.pad_method is None


def test_initialisation_with_pooled_emb_datetime(
    test_df_with_datetime,
    emb,
    emb_reduced,
    emb_pooled,
):
    # initialise with pooled embeddings
    # (and pass in the correct id_column)
    obj = PrepareData(
        original_df=test_df_with_datetime,
        embeddings=emb,
        embeddings_reduced=emb_reduced,
        pooled_embeddings=emb_pooled,
        id_column="id_col",
    )
    # should have an error as we haven't passed in the id column,
    # and so it expects the number of rows in emb_pooled to
    # equal the number of rows in the dataframe
    pd.testing.assert_frame_equal(obj.original_df, test_df_with_datetime)
    assert obj.id_column == "id_col"
    assert obj.label_column is None
    assert (obj.embeddings == emb).all()
    assert (obj.embeddings_reduced == emb_reduced).all()
    # check that .df is a data frame and check the shape of it
    assert type(obj.df) == pd.DataFrame
    # number of columns is:
    # original number of columns +
    # number of columns in emb +
    # number of columns in emb_reduced +
    # 3 time feature
    assert obj.df.shape == (
        len(obj.original_df.index),
        len(obj.original_df.columns) + emb.shape[1] + emb_reduced.shape[1] + 3,
    )
    assert (obj.pooled_embeddings == emb_pooled).all()
    assert set(obj._time_feature_choices) == {
        "time_encoding",
        "time_diff",
        "timeline_index",
    }
    assert obj.time_features_added
    assert obj.df_padded is None
    assert obj.array_padded is None
    assert obj.pad_method is None


def test_initialisation_with_pooled_emb_no_time(
    test_df_no_time,
    emb,
    emb_reduced,
    emb_pooled,
):
    # initialise with pooled embeddings
    # (and pass in the correct id_column)
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        embeddings_reduced=emb_reduced,
        pooled_embeddings=emb_pooled,
        id_column="id_col",
    )
    # should have an error as we haven't passed in the id column,
    # and so it expects the number of rows in emb_pooled to
    # equal the number of rows in the dataframe
    pd.testing.assert_frame_equal(obj.original_df, test_df_no_time)
    assert obj.id_column == "id_col"
    assert obj.label_column is None
    assert (obj.embeddings == emb).all()
    assert (obj.embeddings_reduced == emb_reduced).all()
    # check that .df is a data frame and check the shape of it
    assert type(obj.df) == pd.DataFrame
    # number of columns is:
    # original number of columns +
    # number of columns in emb +
    # number of columns in emb_reduced +
    # 1 time feature
    assert obj.df.shape == (
        len(obj.original_df.index),
        len(obj.original_df.columns) + emb.shape[1] + emb_reduced.shape[1] + 1,
    )
    assert (obj.pooled_embeddings == emb_pooled).all()
    assert obj._time_feature_choices == ["timeline_index"]
    assert obj.time_features_added
    assert obj.df_padded is None
    assert obj.array_padded is None
    assert obj.pad_method is None


def test_initialisation_with_pooled_emb_no_id(
    test_df_with_datetime,
    emb,
    emb_reduced,
    emb_pooled,
):
    # initialise with pooled embeddings
    # (but do not pass in the id_column)
    with pytest.raises(
        ValueError,
        match="If provided, `pooled_embeddings` should have the same number "
        "of rows as there are different ids in the id-column.",
    ):
        # should have an error as we haven't passed in the id column,
        # and so it expects the number of rows in emb_pooled to equal 1
        # (as a dummy-id is created instead with zeros, so there's only one unique id)
        PrepareData(
            original_df=test_df_with_datetime,
            embeddings=emb,
            embeddings_reduced=emb_reduced,
            pooled_embeddings=emb_pooled,
        )


def test_initialisation_with_pooled_emb_wrong_id(
    test_df_with_datetime,
    emb,
    emb_reduced,
    emb_pooled,
):
    # initialise with pooled embeddings
    # (and pass in an incorrect id_column that doesn't exist in the dataframe)
    with pytest.raises(
        ValueError,
        match="If provided, `pooled_embeddings` should have the same number "
        "of rows as there are different ids in the id-column.",
    ):
        # should have an error as we passed in an id column that doesn't exist,
        # and so it expects the number of rows in emb_pooled to equal 1
        # (as a dummy-id is created instead with zeros, so there's only one unique id)
        PrepareData(
            original_df=test_df_with_datetime,
            embeddings=emb,
            embeddings_reduced=emb_reduced,
            pooled_embeddings=emb_pooled,
            id_column="fake_id_column",
        )


def test_initialisation_with_emb_no_2d(
    test_df_with_datetime, emb_reduced, emb_pooled, emb_1d
):
    # initialise with embeddings which is not a 2d array
    with pytest.raises(
        ValueError, match="`embeddings` should be a 2-dimensional array."
    ):
        # emb_1d isn't a 2d array so error
        PrepareData(
            original_df=test_df_with_datetime,
            embeddings=emb_1d,
            embeddings_reduced=emb_reduced,
            pooled_embeddings=emb_pooled,
        )


def test_initialisation_with_emb_diff_len(
    test_df_with_datetime,
    emb_reduced,
    emb_pooled,
):
    # initialise with embeddings which does not have same number of rows as original_df
    with pytest.raises(
        ValueError,
        match="`original_df` and `embeddings` should have the same number of rows.",
    ):
        PrepareData(
            original_df=test_df_with_datetime,
            embeddings=emb_pooled,
            embeddings_reduced=emb_reduced,
            pooled_embeddings=emb_pooled,
        )


def test_initialisation_with_reduced_emb_no_2d(
    test_df_with_datetime, emb, emb_pooled, emb_1d
):
    # initialise with dimension reduced embeddings which is not a 2d array
    with pytest.raises(
        ValueError,
        match="If provided, `embeddings_reduced` should be a 2-dimensional array.",
    ):
        # emb_1d isn't a 2d array so error
        PrepareData(
            original_df=test_df_with_datetime,
            embeddings=emb,
            embeddings_reduced=emb_1d,
            pooled_embeddings=emb_pooled,
        )


def test_initialisation_with_reduced_emb_diff_len(
    test_df_with_datetime,
    emb,
    emb_pooled,
):
    # initialise with dimension reduced embeddings
    # which does not have same number of rows as original_df
    with pytest.raises(
        ValueError,
        match="`original_df`, `embeddings` and `embeddings_reduced` should have the same number of rows.",
    ):
        PrepareData(
            original_df=test_df_with_datetime,
            embeddings=emb,
            embeddings_reduced=emb_pooled,
            pooled_embeddings=emb_pooled,
        )


def test_initialisation_with_pooled_emb_no_2d(
    test_df_with_datetime, emb, emb_reduced, emb_1d
):
    # initialise with pooled embeddings which is not a 2d array
    with pytest.raises(
        ValueError,
        match="If provided, `pooled_embeddings` should be a 2-dimensional array.",
    ):
        # emb_1d isn't a 2d array so error
        PrepareData(
            original_df=test_df_with_datetime,
            embeddings=emb,
            embeddings_reduced=emb_reduced,
            pooled_embeddings=emb_1d,
        )


def test_PrepareData_obtain_colnames_emb(test_df_with_datetime, emb):
    emb_names = [f"e{i+1}" for i in range(emb.shape[1])]

    # test cases where only embeddings are passed
    obj = PrepareData(original_df=test_df_with_datetime, embeddings=emb)
    assert obj._obtain_colnames(embeddings="full") == emb_names
    assert obj._obtain_colnames(embeddings="dim_reduced") == []
    assert obj._obtain_colnames(embeddings="both") == emb_names

    with pytest.raises(
        ValueError, match="Embeddings must be either 'dim_reduced', 'full', or 'both'"
    ):
        obj._obtain_colnames(embeddings="")


def test_obtain_colnames_both(test_df_with_datetime, emb, emb_reduced):
    emb_names = [f"e{i+1}" for i in range(emb.shape[1])]
    emb_reduced_names = [f"d{i+1}" for i in range(emb_reduced.shape[1])]

    # test cases where both are passed
    obj = PrepareData(
        original_df=test_df_with_datetime,
        embeddings=emb,
        embeddings_reduced=emb_reduced,
    )
    assert obj._obtain_colnames(embeddings="full") == emb_names
    assert obj._obtain_colnames(embeddings="dim_reduced") == emb_reduced_names
    assert obj._obtain_colnames(embeddings="both") == emb_reduced_names + emb_names
