from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nlpsig.data_preparation import PrepareData


def test_pad_dataframe_zero_padding_from_below_without_label(test_df_no_time, emb):
    # should have k entries in the returned dataframe
    # should have zeros on the end the dataframe
    # initialisation without label
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb)
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_dataframe(
        df=df_to_pad,
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[: len(df_to_pad.index)].reset_index(drop=True),
        df_to_pad.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        padded_df[len(df_to_pad.index) :].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_dataframe_zero_padding_from_below_with_label(test_df_no_time, emb):
    # should have k entries in the returned dataframe
    # should have zeros on the end of the dataframe
    # the labels added should have -1
    obj = PrepareData(
        original_df=test_df_no_time, embeddings=emb, label_column="label_col"
    )
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_dataframe(
        df=df_to_pad,
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[: len(df_to_pad.index)].reset_index(drop=True),
        df_to_pad.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        padded_df[len(df_to_pad.index) :].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
                obj.label_column: -1,
            }
        ),
    )


def test_pad_dataframe_zero_padding_from_above_without_label(test_df_no_time, emb):
    # should have k entries in the returned dataframe
    # should have zeros on the end
    # initialisation without label
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb)
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_dataframe(
        df=df_to_pad,
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[pad_amount:].reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        padded_df[:pad_amount].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_dataframe_zero_padding_from_above_with_label(test_df_no_time, emb):
    # should have k entries in the returned dataframe
    # should have zeros on the top
    # the labels added should have -1
    obj = PrepareData(
        original_df=test_df_no_time, embeddings=emb, label_column="label_col"
    )
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_dataframe(
        df=df_to_pad,
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[pad_amount:].reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        padded_df[:pad_amount].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
                obj.label_column: -1,
            }
        ),
    )


def test_pad_dataframe_non_zero_padding_from_below(test_df_no_time, emb):
    # should have k entries in the returned dataframe
    # should have repeated entries on the end
    # the labels added should have -1
    obj = PrepareData(
        original_df=test_df_no_time, embeddings=emb, label_column="label_col"
    )
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_dataframe(
        df=df_to_pad,
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[: len(df_to_pad.index)].reset_index(drop=True),
        df_to_pad.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        padded_df[len(df_to_pad.index) :].reset_index(drop=True),
        pd.concat([df_to_pad.tail(1)] * pad_amount).reset_index(drop=True),
    )


def test_pad_dataframe_non_zero_padding_from_above(test_df_no_time, emb):
    # should have k entries in the returned dataframe
    # should have repeated entries on the top
    # the labels added should have -1
    obj = PrepareData(
        original_df=test_df_no_time, embeddings=emb, label_column="label_col"
    )
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_dataframe(
        df=df_to_pad,
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[pad_amount:].reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        padded_df[:pad_amount].reset_index(drop=True),
        pd.concat([df_to_pad.head(1)] * pad_amount).reset_index(drop=True),
    )


def test_pad_dataframe_k_equal_zero(test_df_no_time, test_df_to_pad, emb):
    # if k == 0, should have error
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
    )
    with pytest.raises(
        ValueError,
        match="`k` must be a positive integer",
    ):
        obj._pad_dataframe(
            df=test_df_to_pad,
            k=0,
            zero_padding=False,
            colnames=obj._obtain_colnames("full"),
            time_feature=["timeline_index"],
            id=0,
            pad_from_below=False,
        )


def test_pad_dataframe_k_negative(test_df_no_time, test_df_to_pad, emb):
    # if k < 0, should have error
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
    )
    with pytest.raises(
        ValueError,
        match="`k` must be a positive integer",
    ):
        obj._pad_dataframe(
            df=test_df_to_pad,
            k=-1,
            zero_padding=False,
            colnames=obj._obtain_colnames("full"),
            time_feature=["timeline_index"],
            id=0,
            pad_from_below=False,
        )


def test_pad_dataframe_no_pad(test_df_no_time, emb):
    # choose a value k which is equal to the number of entries in df_to_pad
    # there should be no padding added and return the dataframe as it already has k entries
    obj = PrepareData(
        original_df=test_df_no_time, embeddings=emb, label_column="label_col"
    )
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols]
    k = len(df_to_pad.index)
    padded_df = obj._pad_dataframe(
        df=df_to_pad,
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df.reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )


def test_pad_dataframe_cutoff(test_df_no_time, emb):
    # choose a value k which is less than the number of entries in df_to_pad
    # it should return the last k entries in the dataframe (no padding)
    obj = PrepareData(
        original_df=test_df_no_time, embeddings=emb, label_column="label_col"
    )
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols]
    k = len(df_to_pad.index) - 10
    padded_df = obj._pad_dataframe(
        df=df_to_pad,
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df.reset_index(drop=True), df_to_pad.tail(k).reset_index(drop=True)
    )


def test_pad_id_k_equal_zero(test_df_no_time, emb):
    # if k == 0, should have error
    obj = PrepareData(
        original_df=test_df_no_time, embeddings=emb, label_column="label_col"
    )
    with pytest.raises(
        ValueError,
        match="`k` must be a positive integer",
    ):
        obj._pad_id(
            k=-1,
            zero_padding=False,
            colnames=obj._obtain_colnames("full"),
            time_feature=["timeline_index"],
            id=0,
            pad_from_below=False,
        )


def test_pad_id_k_negative(test_df_no_time, emb):
    # if k < 0, should have error
    obj = PrepareData(
        original_df=test_df_no_time, embeddings=emb, label_column="label_col"
    )
    with pytest.raises(
        ValueError,
        match="`k` must be a positive integer",
    ):
        obj._pad_id(
            k=-1,
            zero_padding=False,
            colnames=obj._obtain_colnames("full"),
            time_feature=["timeline_index"],
            id=0,
            pad_from_below=False,
        )


def test_pad_id_zero_padding_from_below(test_df_no_time, emb):
    # padding for id==0 which we know has 100 entries by construction
    # should have k entries in the returned dataframe
    # should have zeros on the end
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0]
    k = len(test_df_no_time[test_df_no_time["id_col"] == 0].index) + pad_amount
    padded_df = obj._pad_id(
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[: len(df_to_pad.index)].reset_index(drop=True),
        df_to_pad.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        padded_df[len(df_to_pad.index) :].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_id_zero_padding_from_above(test_df_no_time, emb):
    # padding for id==0 which we know has 100 entries by construction
    # should have k entries in the returned dataframe
    # should have zeros on the top
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0]
    k = len(test_df_no_time[test_df_no_time["id_col"] == 0].index) + pad_amount
    padded_df = obj._pad_id(
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[pad_amount:].reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        padded_df[:pad_amount].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_id_non_zero_padding_from_below(test_df_no_time, emb):
    # padding for id==0 which we know has 100 entries by construction
    # should have k entries in the returned dataframe
    # should have repeated entries on the end
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0]
    k = len(test_df_no_time[test_df_no_time["id_col"] == 0].index) + pad_amount
    colnames = ["e1", "e2"]
    padded_df = obj._pad_id(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[: len(df_to_pad.index)].reset_index(drop=True),
        df_to_pad.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        padded_df[len(df_to_pad.index) :].reset_index(drop=True),
        pd.concat([df_to_pad.tail(1)] * pad_amount).reset_index(drop=True),
    )


def test_pad_id_non_zero_padding_from_above(test_df_no_time, emb):
    # padding for id==0 which we know has 100 entries by construction
    # should have k entries in the returned dataframe
    # should have repeated entries on the end
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0]
    k = len(test_df_no_time[test_df_no_time["id_col"] == 0].index) + pad_amount
    padded_df = obj._pad_id(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[pad_amount:].reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        padded_df[:pad_amount].reset_index(drop=True),
        pd.concat([df_to_pad.head(1)] * pad_amount).reset_index(drop=True),
    )


def test_pad_id_no_pad(test_df_no_time, emb):
    # choose a value k which is equal to the number of entries in df_to_pad
    # there should be no padding added and return the dataframe as it already has k entries
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0]
    k = len(test_df_no_time[test_df_no_time["id_col"] == 0].index)
    padded_df = obj._pad_id(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df.reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )


def test_pad_id_cutoff(test_df_no_time, emb):
    # choose a value k which is less than the number of entries in df_to_pad
    # it should return the last k entries in the dataframe (no padding)
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0]
    k = len(test_df_no_time[test_df_no_time["id_col"] == 0].index) - pad_amount
    padded_df = obj._pad_id(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        id=0,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df.reset_index(drop=True), df_to_pad.tail(k).reset_index(drop=True)
    )


def test_pad_history_zero_padding_no_history_from_below(test_df_no_time, emb):
    # padding by history but for an index that has no history so far (empty)
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 0
    pad_amount = 10
    k = pad_amount
    colnames = ["e1", "e2"]
    padded_df = obj._pad_history(
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=False,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df,
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(k)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_history_zero_padding_no_history_from_above(test_df_no_time, emb):
    # padding by history but for an index that has no history so far (empty)
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 0
    pad_amount = 10
    k = pad_amount
    colnames = ["e1", "e2"]
    padded_df = obj._pad_history(
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=False,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df,
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(k)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_history_zero_padding_some_history_from_below(test_df_no_time, emb):
    # padding by history but not enough history to fill so needs some padding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 10
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][:index]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=False,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[: len(df_to_pad.index)].reset_index(drop=True),
        df_to_pad.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        padded_df[len(df_to_pad.index) :].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_history_zero_padding_some_history_from_above(test_df_no_time, emb):
    # padding by history but not enough history to fill so needs some padding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 10
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][:index]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=False,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[pad_amount:].reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        padded_df[:pad_amount].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_history_non_zero_padding_no_history_from_below(test_df_no_time, emb):
    # padding by history but for an index that has no history so far (empty)
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 0
    pad_amount = 10
    k = pad_amount
    colnames = ["e1", "e2"]
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=False,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df,
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(k)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_history_non_zero_padding_no_history_from_above(test_df_no_time, emb):
    # padding by history but for an index that has no history so far (empty)
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 0
    pad_amount = 10
    k = pad_amount
    colnames = ["e1", "e2"]
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=False,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df,
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(k)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_history_non_zero_padding_some_history_from_below(test_df_no_time, emb):
    # padding by history but not enough history to fill so needs some padding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 10
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][:index]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=False,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    print(obj.df.iloc[index])
    pd.testing.assert_frame_equal(
        padded_df[: len(df_to_pad.index)].reset_index(drop=True),
        df_to_pad.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        padded_df[len(df_to_pad.index) :].reset_index(drop=True),
        pd.concat([df_to_pad.tail(1)] * pad_amount).reset_index(drop=True),
    )


def test_pad_history_non_zero_padding_some_history_from_above(test_df_no_time, emb):
    # padding by history but not enough history to fill so needs some padding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 10
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][:index]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=False,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[pad_amount:].reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        padded_df[:pad_amount].reset_index(drop=True),
        pd.concat([df_to_pad.head(1)] * pad_amount).reset_index(drop=True),
    )


def test_pad_history_just_enough_history(test_df_no_time, emb):
    # padding by history and there's just enough history to fill
    # choose a value k which is equal to the number of entries in df_to_pad
    # there should be no padding added and return the dataframe as it already has k entries
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][:-1]
    k = len(test_df_no_time[test_df_no_time["id_col"] == 0].index) - 1
    index = k
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=False,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df.reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )


def test_pad_history_many_history_cutoff(test_df_no_time, emb):
    # padding by history and there's more than enough history to fill (cut off some history)
    # choose a value k which is equal to the number of entries in df_to_pad
    # there should be no padding added and return the dataframe as it already has k entries
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][:-1]
    k = len(test_df_no_time[test_df_no_time["id_col"] == 0].index) - pad_amount
    index = k - 1 + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=False,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df.reset_index(drop=True), df_to_pad.tail(k).reset_index(drop=True)
    )


def test_pad_history_no_history_zero_padding_include_current_from_below(
    test_df_no_time, emb
):
    # padding by history but for an index that has no history so far
    # but has the current embedding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 0
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][0:1]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=True,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[: len(df_to_pad.index)].reset_index(drop=True),
        df_to_pad.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        padded_df[len(df_to_pad.index) :].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_history_no_history_zero_padding_include_current_from_above(
    test_df_no_time, emb
):
    # padding by history but for an index that has no history so far
    # but has the current embedding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 0
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][0:1]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=True,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[pad_amount:].reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        padded_df[:pad_amount].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_history_some_history_zero_padding_include_current_from_below(
    test_df_no_time, emb
):
    # padding by history but not enough history to fill so needs some padding
    # but has the current embedding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 10
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    # up to index+1 because it needs to include the emdedding in the index
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][: (index + 1)]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=True,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[: len(df_to_pad.index)].reset_index(drop=True),
        df_to_pad.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        padded_df[len(df_to_pad.index) :].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_history_some_history_zero_padding_include_current_from_above(
    test_df_no_time, emb
):
    # padding by history but not enough history to fill so needs some padding
    # but has the current embedding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 10
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    # up to index+1 because it needs to include the emdedding in the index
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][: (index + 1)]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=True,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=True,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[pad_amount:].reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        padded_df[:pad_amount].reset_index(drop=True),
        pd.DataFrame(
            {
                "timeline_index": [0 for i in range(pad_amount)],
                "e1": 0.0,
                "e2": 0.0,
                obj.id_column: 0,
            }
        ),
    )


def test_pad_history_no_history_non_zero_padding_include_current_from_below(
    test_df_no_time, emb
):
    # padding by history but for an index that has no history so far
    # but has the current embedding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 0
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][0:1]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=True,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df.reset_index(drop=True),
        pd.concat([df_to_pad] * k).reset_index(drop=True),
    )


def test_pad_history_no_history_non_zero_padding_include_current_from_above(
    test_df_no_time, emb
):
    # padding by history but for an index that has no history so far
    # but has the current embedding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 0
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][0:1]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=True,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df.reset_index(drop=True),
        pd.concat([df_to_pad] * k).reset_index(drop=True),
    )


def test_pad_history_some_history_non_zero_padding_include_current_from_below(
    test_df_no_time, emb
):
    # padding by history but not enough history to fill so needs some padding
    # but has the current embedding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 10
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    # up to index+1 because it needs to include the emdedding in the index
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][: (index + 1)]
    k = len(df_to_pad.index) + pad_amount
    colnames = ["e1", "e2"]
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=True,
        pad_from_below=True,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[: len(df_to_pad.index)].reset_index(drop=True),
        df_to_pad.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        padded_df[len(df_to_pad.index) :].reset_index(drop=True),
        pd.concat([df_to_pad.tail(1)] * pad_amount).reset_index(drop=True),
    )


def test_pad_history_some_history_non_zero_padding_include_current_from_above(
    test_df_no_time, emb
):
    # padding by history but not enough history to fill so needs some padding
    # but has the current embedding
    obj = PrepareData(original_df=test_df_no_time, embeddings=emb, id_column="id_col")
    index = 10
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column]
    # up to index+1 because it needs to include the emdedding in the index
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0][: (index + 1)]
    k = len(df_to_pad.index) + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=True,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df[pad_amount:].reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        padded_df[:pad_amount].reset_index(drop=True),
        pd.concat([df_to_pad.head(1)] * pad_amount).reset_index(drop=True),
    )


def test_pad_history_just_enough_history_include_current(test_df_no_time, emb):
    # padding by history and there's just enough history to fill
    # but has the current embedding
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0]
    k = len(test_df_no_time[test_df_no_time["id_col"] == 0].index)
    index = k - 1
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=True,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df.reset_index(drop=True), df_to_pad.reset_index(drop=True)
    )


def test_pad_history_many_history_include_current(test_df_no_time, emb):
    # but has the current embedding
    # padding by history and there's more than enough history to fill (cut off some history)
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    pad_amount = 10
    colnames = ["e1", "e2"]
    cols = ["timeline_index", *colnames] + [obj.id_column] + [obj.label_column]
    df_to_pad = obj.df[cols][obj.df["id_col"] == 0]
    k = len(test_df_no_time[test_df_no_time["id_col"] == 0].index) - pad_amount
    index = k - 1 + pad_amount
    padded_df = obj._pad_history(
        k=k,
        zero_padding=False,
        colnames=colnames,
        time_feature=["timeline_index"],
        index=index,
        include_current_embedding=True,
        pad_from_below=False,
    )
    assert len(padded_df.index) == k
    pd.testing.assert_frame_equal(
        padded_df.reset_index(drop=True), df_to_pad.tail(k).reset_index(drop=True)
    )


def test_pad_by_id_k_last(test_df_no_time, emb):
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    k = 10
    padded_array = obj.pad(
        pad_by="id",
        method="k_last",
        zero_padding=True,
        k=k,
        time_feature="timeline_index",
        standardise_method=None,
        embeddings="full",
        include_current_embedding=True,
        pad_from_below=True,
    )
    # number of columns is:
    # number of time features + number of columns in emb + id col + label col
    ncol = len(obj._time_feature_choices) + emb.shape[1] + 1 + 1
    assert type(obj.df_padded) == pd.DataFrame
    assert obj.df_padded.shape == (k * len(obj.original_df["id_col"].unique()), ncol)
    assert type(obj.array_padded) == np.ndarray
    assert np.array_equal(padded_array, obj.array_padded)
    assert obj.array_padded.shape == (len(obj.original_df["id_col"].unique()), k, ncol)


def test_pad_by_id_max(test_df_no_time, emb):
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    padded_array = obj.pad(
        pad_by="id",
        method="max",
        zero_padding=True,
        time_feature="timeline_index",
        standardise_method=None,
        embeddings="full",
        include_current_embedding=True,
        pad_from_below=True,
    )
    # number of columns is:
    # number of time features + number of columns in emb + id col + label col
    ncol = len(obj._time_feature_choices) + emb.shape[1] + 1 + 1
    assert type(obj.df_padded) == pd.DataFrame
    k = obj.original_df["id_col"].value_counts().max()
    assert obj.df_padded.shape == (k * len(obj.original_df["id_col"].unique()), ncol)
    assert type(obj.array_padded) == np.ndarray
    assert np.array_equal(padded_array, obj.array_padded)
    assert obj.array_padded.shape == (len(obj.original_df["id_col"].unique()), k, ncol)


def test_pad_by_history_k_last(test_df_no_time, emb):
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    k = 10
    padded_array = obj.pad(
        pad_by="history",
        method="k_last",
        zero_padding=True,
        k=k,
        time_feature="timeline_index",
        standardise_method=None,
        embeddings="full",
        include_current_embedding=True,
        pad_from_below=True,
    )
    # number of columns is:
    # number of time features + number of columns in emb + id col + label col
    ncol = len(obj._time_feature_choices) + emb.shape[1] + 1 + 1
    assert type(obj.df_padded) == pd.DataFrame
    assert obj.df_padded.shape == (k * len(obj.original_df.index), ncol)
    assert type(obj.array_padded) == np.ndarray
    assert np.array_equal(padded_array, obj.array_padded)
    assert obj.array_padded.shape == (len(obj.original_df.index), k, ncol)


def test_pad_by_history_max(test_df_no_time, emb):
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    padded_array = obj.pad(
        pad_by="history",
        method="max",
        zero_padding=True,
        time_feature="timeline_index",
        standardise_method=None,
        embeddings="full",
        include_current_embedding=True,
        pad_from_below=True,
    )
    # number of columns is:
    # number of time features + number of columns in emb + id col + label col
    ncol = len(obj._time_feature_choices) + emb.shape[1] + 1 + 1
    assert type(obj.df_padded) == pd.DataFrame
    k = obj.original_df["id_col"].value_counts().max()
    assert obj.df_padded.shape == (k * len(obj.original_df.index), ncol)
    assert type(obj.array_padded) == np.ndarray
    assert np.array_equal(padded_array, obj.array_padded)
    assert obj.array_padded.shape == (len(obj.original_df.index), k, ncol)


def test_pad_wrong_pad_by(test_df_no_time, emb):
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    with pytest.raises(ValueError, match="`pad_by` must be either 'id' or 'history'."):
        obj.pad(
            pad_by="fake_pad_by",
            method="max",
            zero_padding=True,
            time_feature="timeline_index",
            standardise_method=None,
            embeddings="full",
            include_current_embedding=True,
            pad_from_below=True,
        )


def test_pad_wrong_method(test_df_no_time, emb):
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    with pytest.raises(ValueError, match="`method` must be either 'k_last' or 'max'."):
        obj.pad(
            pad_by="id",
            method="fake_method",
            zero_padding=True,
            time_feature="timeline_index",
            standardise_method=None,
            embeddings="full",
            include_current_embedding=True,
            pad_from_below=True,
        )


def test_pad_by_id_k_last_standardise_standardise(test_df_no_time, emb):
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    k = 10
    padded_array = obj.pad(
        pad_by="id",
        method="k_last",
        zero_padding=True,
        k=k,
        time_feature="timeline_index",
        standardise_method="standardise",
        embeddings="full",
        include_current_embedding=True,
        pad_from_below=True,
    )
    standardise_vec = obj._standardise_pd(
        vec=obj.df["timeline_index"], method="standardise"
    )["standardised_pd"]
    pd.testing.assert_series_equal(obj.df["timeline_index"], standardise_vec)
    # number of columns is:
    # number of time features + number of columns in emb + id col + label col
    ncol = len(obj._time_feature_choices) + emb.shape[1] + 1 + 1
    assert type(obj.df_padded) == pd.DataFrame
    assert obj.df_padded.shape == (k * len(obj.original_df["id_col"].unique()), ncol)
    assert type(obj.array_padded) == np.ndarray
    assert np.array_equal(padded_array, obj.array_padded)
    assert obj.array_padded.shape == (len(obj.original_df["id_col"].unique()), k, ncol)


def test_pad_by_id_k_last_standardise_normalise(test_df_no_time, emb):
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    k = 10
    padded_array = obj.pad(
        pad_by="id",
        method="k_last",
        zero_padding=True,
        k=k,
        time_feature="timeline_index",
        standardise_method="normalise",
        embeddings="full",
        include_current_embedding=True,
        pad_from_below=True,
    )
    normalise_vec = obj._standardise_pd(
        vec=obj.df["timeline_index"], method="normalise"
    )["standardised_pd"]
    pd.testing.assert_series_equal(obj.df["timeline_index"], normalise_vec)
    # number of columns is:
    # number of time features + number of columns in emb + id col + label col
    ncol = len(obj._time_feature_choices) + emb.shape[1] + 1 + 1
    assert type(obj.df_padded) == pd.DataFrame
    assert obj.df_padded.shape == (k * len(obj.original_df["id_col"].unique()), ncol)
    assert type(obj.array_padded) == np.ndarray
    assert np.array_equal(padded_array, obj.array_padded)
    assert obj.array_padded.shape == (len(obj.original_df["id_col"].unique()), k, ncol)


def test_pad_by_id_k_last_standardise_minmax(test_df_no_time, emb):
    obj = PrepareData(
        original_df=test_df_no_time,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    k = 10
    padded_array = obj.pad(
        pad_by="id",
        method="k_last",
        zero_padding=True,
        k=k,
        time_feature="timeline_index",
        standardise_method="minmax",
        embeddings="full",
        include_current_embedding=True,
        pad_from_below=True,
    )
    minmax_vec = obj._standardise_pd(vec=obj.df["timeline_index"], method="minmax")[
        "standardised_pd"
    ]
    pd.testing.assert_series_equal(obj.df["timeline_index"], minmax_vec)
    # number of columns is:
    # number of time features + number of columns in emb + id col + label col
    ncol = len(obj._time_feature_choices) + emb.shape[1] + 1 + 1
    assert type(obj.df_padded) == pd.DataFrame
    assert obj.df_padded.shape == (k * len(obj.original_df["id_col"].unique()), ncol)
    assert type(obj.array_padded) == np.ndarray
    assert np.array_equal(padded_array, obj.array_padded)
    assert obj.array_padded.shape == (len(obj.original_df["id_col"].unique()), k, ncol)


def test_pad_by_id_k_last_standardise_multiple(test_df_with_datetime, emb):
    obj = PrepareData(
        original_df=test_df_with_datetime,
        embeddings=emb,
        id_column="id_col",
        label_column="label_col",
    )
    k = 10
    time_features = ["timeline_index", "time_encoding", "time_diff"]
    # expected standardised vectors
    standardised_vec = obj._standardise_pd(
        vec=obj.df["timeline_index"], method="standardise"
    )["standardised_pd"]
    normalised_vec = obj._standardise_pd(
        vec=obj.df["time_encoding"], method="normalise"
    )["standardised_pd"]
    none_standardisation_vec = obj.df["time_diff"]
    # pad and perform standardisation
    padded_array = obj.pad(
        pad_by="id",
        method="k_last",
        zero_padding=True,
        k=k,
        time_feature=time_features,
        standardise_method=["standardise", "normalise", None],
        embeddings="full",
        include_current_embedding=True,
        pad_from_below=True,
    )
    pd.testing.assert_series_equal(obj.df["timeline_index"], standardised_vec)
    pd.testing.assert_series_equal(obj.df["time_encoding"], normalised_vec)
    pd.testing.assert_series_equal(obj.df["time_diff"], none_standardisation_vec)
    # number of columns is:
    # number of time features + number of columns in emb + id col + label col
    ncol = len(time_features) + emb.shape[1] + 1 + 1
    assert type(obj.df_padded) == pd.DataFrame
    assert obj.df_padded.shape == (k * len(obj.original_df["id_col"].unique()), ncol)
    assert type(obj.array_padded) == np.ndarray
    assert np.array_equal(padded_array, obj.array_padded)
    assert obj.array_padded.shape == (len(obj.original_df["id_col"].unique()), k, ncol)
