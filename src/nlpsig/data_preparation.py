from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


class PrepareData:
    """
    Class to prepare dataset for computing signatures.

    Parameters
    ----------
    original_df : pd.DataFrame
        Dataset as a pandas dataframe.
    embeddings : np.array
        Embeddings for each of the items in `original_df`.
    embeddings_reduced : np.array | None, optional
        Dimension reduced embeddings, by default None.
    pooled_embeddings : np.array | None, optional
        Pooled embeddings for each unique id in `id_column`, by default None.
    id_column : str | None, optional
        Name of the column which identifies each of the text, e.g.
        - "text_id" (if each item in `original_df` is a word or sentence from a particular text),
        - "user_id" (if each item in `original_df` is a post from a particular user)
        - "timeline_id" (if each item in `original_df` is a post from a particular time)
        If None, it will create a dummy id_column named "dummy_id" and fill with zeros.
    label_column : str | None, optional
        Name of the column which are corresponds to the labels of the data.

    Raises
    ------
    ValueError
        if `original_df` and `embeddings` does not have the same number of rows.
    ValueError
        if `original_df` and `embeddings_reduced` does not have the same number of rows
        (if `embeddings_reduced` is provided).
    """

    def __init__(
        self,
        original_df: pd.DataFrame,
        embeddings: np.array,
        embeddings_reduced: np.array | None = None,
        pooled_embeddings: np.array | None = None,
        id_column: str | None = None,
        label_column: str | None = None,
        verbose: bool = True,
    ):
        self.verbose = verbose

        # perform checks that original_df have the right column names to work with
        if embeddings.ndim != 2:
            raise ValueError("`embeddings` should be a 2-dimensional array.")
        if original_df.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "`original_df` and `embeddings` should have the same number of rows."
            )
        if embeddings_reduced is not None:
            if embeddings_reduced.ndim != 2:
                raise ValueError(
                    "If provided, `embeddings_reduced` should be a 2-dimensional array."
                )
            if original_df.shape[0] != embeddings_reduced.shape[0]:
                raise ValueError(
                    "`original_df`, `embeddings` and `embeddings_reduced` "
                    "should have the same number of rows."
                )

        self.original_df: pd.DataFrame = original_df
        self.id_column: str | None = id_column
        if (label_column is not None) and (label_column not in original_df.columns):
            raise KeyError(f"{label_column} is not a column in original_df.")
        self.label_column: str | None = label_column
        # set embeddings
        self.embeddings: np.array = embeddings
        self.embeddings_reduced: np.array | None = embeddings_reduced
        # obtain modelling dataframe
        self.df: pd.DataFrame | None = None
        self.df = self._get_modeling_dataframe()

        # set pooled embeddings if provided
        if pooled_embeddings is not None:
            if pooled_embeddings.ndim != 2:
                raise ValueError(
                    "If provided, `pooled_embeddings` should be a 2-dimensional array."
                )
            if len(self.df[self.id_column].unique()) != pooled_embeddings.shape[0]:
                if self.verbose:
                    print(
                        f"[INFO] `len(self.df[self.id_column].unique())`={len(self.df[self.id_column].unique())}"
                        f" and `pooled_embeddings.shape[0]`={pooled_embeddings.shape[0]}."
                    )
                raise ValueError(
                    "If provided, `pooled_embeddings` should have the same number "
                    "of rows as there are different ids in the id-column."
                )

        self.pooled_embeddings: np.array | None = pooled_embeddings
        # obtain time features
        self._feature_list: list[str] = []
        self.time_features_added: bool = False
        self.df = self._set_time_features()
        self.df_padded: pd.DataFrame | None = None
        self.array_padded: np.array | None = None
        self.pad_method: str = None
        self.standardise_transform: dict[str, Callable] | None = None

    def _get_modeling_dataframe(self) -> pd.DataFrame:
        """
        [Private] Combines `.original_df` with the sentence
        embeddings and the dimension reduced embeddings

        Returns
        -------
        pd.DataFrame
            Original dataframe concatenated with the embeddings and
            dimension reduced embeddings (column-wise)
            - columns starting with "e" followed by a number denotes each
              dimension of the embeddings
            - columns starting with "d" followed by a number denotes each
              dimension of the dimension reduced embeddings
        """
        if self.df is not None:
            return self.df

        if self.verbose:
            print("[INFO] Concatenating the embeddings to the dataframe...")
            print("[INFO] - columns beginning with 'e' denote the full embddings.")

        embedding_df = pd.DataFrame(
            self.embeddings,
            columns=[f"e{i+1}" for i in range(self.embeddings.shape[1])],
        )

        if self.embeddings_reduced is not None:
            if self.verbose:
                print(
                    "[INFO] - columns beginning with 'd' denote the dimension reduced embeddings."
                )

            embeddings_reduced_df = pd.DataFrame(
                self.embeddings_reduced,
                columns=[f"d{i+1}" for i in range(self.embeddings_reduced.shape[1])],
            )
            df = pd.concat(
                [
                    self.original_df.reset_index(drop=True),
                    embeddings_reduced_df,
                    embedding_df,
                ],
                axis=1,
            )
        else:
            df = pd.concat(
                [self.original_df.reset_index(drop=True), embedding_df],
                axis=1,
            )

        if self.id_column is None:
            self.id_column = "dummy_id"
            if self.verbose:
                print(
                    f"[INFO] No id_column was passed, so setting id_column to '{self.id_column}'."
                )

        if self.id_column not in self.original_df.columns:
            if self.verbose:
                print(
                    f"[INFO] There is no column in `.original_df` called '{self.id_column}'. "
                    f"Adding a new column named '{self.id_column}' of zeros."
                )
            # set default value to id_column
            df[self.id_column] = 0

        return df

    @staticmethod
    def _time_fraction(x: pd.Timestamp) -> float:
        """
        [Private] Converts a date, x, as a fraction of the year.

        Parameters
        ----------
        x : pd.Timestamp
            Date.

        Returns
        -------
        float
            The date as a fraction of the year.
        """
        # compute how many seconds the date is into the year
        x_year_start = pd.Timestamp(x.year, 1, 1)
        seconds_into_cal_year = abs(x - x_year_start).total_seconds()
        # compute the time fraction into the year
        time_frac = seconds_into_cal_year / (365 * 24 * 60 * 60)
        return x.year + time_frac

    def _set_time_features(self) -> pd.DataFrame:
        """
        [Private] Updates the dataframe in `.df` to include time features:
        - `time_encoding`: the date as a fraction of the year
           (only if 'datetime' is a column in `.df` dataframe).
        - `time_diff`: the difference in time (in minutes) between successive records
           (only if 'datetime' is a column in `.df` dataframe).
        - `timeline_index`: the index of each post for each id.

        Returns
        -------
        pd.DataFrame
            Updated dataframe with time features.
        """
        if self.time_features_added:
            if self.verbose:
                print("Time features have already been added.")
            return None

        if self.verbose:
            print("[INFO] Adding time feature columns into dataframe in `.df`.")

        if "datetime" in self.df.columns:
            self._feature_list += ["time_encoding", "time_diff"]

            # checking 'datetime' column is datatime type
            self.df["datetime"] = pd.to_datetime(self.df["datetime"])

            # obtain time encoding by computing the fraction of year it is in
            if self.verbose:
                print("[INFO] Adding 'time_encoding' feature...")

            self.df["time_encoding"] = self.df["datetime"].map(
                lambda t: self._time_fraction(t)
            )
            # reset index so that we can return to the starting index afterwards
            self.df = self.df.reset_index()

            # sort by the id and the date
            self.df = self.df.sort_values(by=[self.id_column, "datetime"])

            # calculate time difference between posts
            if self.verbose:
                print("[INFO] Adding 'time_diff' feature...")

            self.df["time_diff"] = list(
                self.df.groupby(self.id_column)
                .apply(
                    lambda x: [0.0]
                    + [
                        (
                            x["datetime"].iloc[i] - x["datetime"].iloc[i - 1]
                        ).total_seconds()
                        / 60
                        for i in range(1, len(x))
                    ]
                )
                .explode()
            )
        else:
            if self.verbose:
                print(
                    "[INFO] Note 'datetime' is not a column in `.df`, "
                    "so only 'timeline_index' is added."
                )
                print(
                    "[INFO] As 'datetime' is not a column in `.df`, "
                    "we assume that the data is ordered by time with respect to the id."
                )

        # assign index for each post in each timeline
        self._feature_list += ["timeline_index"]

        if self.verbose:
            print("[INFO] Adding 'timeline_index' feature...")

        self.df["timeline_index"] = list(
            self.df.groupby(self.id_column)
            .apply(lambda x: list(range(1, len(x) + 1)))
            .explode()
        )

        if "datetime" in self.df.columns:
            # re-sort the dataframe by the index
            self.df = self.df.sort_index()

        self.time_features_added = True

        return self.df

    def _obtain_embedding_colnames(self, embeddings: str) -> list[str]:
        """
        [Private] Obtains the column names storing the embeddings.

        Parameters
        ----------
        embeddings : str
            Options are:
            - "dim_reduced": dimension reduced embeddings.
            - "full": full embeddings.
            - "both": concatenation of dimension reduced and full embeddings.

        Returns
        -------
        list[str]
            List of column names which store the embeddings.

        Raises
        ------
        ValueError
            if embeddings is not either of 'dim_reduced', 'full', or 'both'.
        """
        if embeddings not in ["dim_reduced", "full", "both"]:
            raise ValueError(
                "Embeddings must be either 'dim_reduced', 'full', or 'both'."
            )

        if embeddings == "dim_reduced":
            # obtain columns for the dimension reduced embeddings
            # these are columns which start with 'd' and have a number following it
            colnames = [col for col in self.df.columns if re.match(r"^d\w*[0-9]", col)]
        elif embeddings == "full":
            # obtain columns for the full embeddings
            # these are columns which start with 'e' and have a number following it
            colnames = [col for col in self.df.columns if re.match(r"^e\w*[0-9]", col)]
        elif embeddings == "both":
            # add columns for the embeddings
            colnames = [col for col in self.df.columns if re.match(r"^d\w*[0-9]", col)]
            colnames += [col for col in self.df.columns if re.match(r"^e\w*[0-9]", col)]

        return colnames

    def _check_feature_exists(self, feature: str) -> bool:
        """
        [Private] Checks if `feature` is a column in `self._feature_list`. If not,
        check if `self.df` dataframe and if it is, add this to `self._feature_list`.

        Parameters
        ----------
        feature : str
            Feature name.

        Returns
        -------
        bool
            True if `feature` is in `self._feature_list` or is a
            column name in `self.df`.
        """
        if (feature not in self._feature_list) and (feature in self.df.columns):
            # not in ._feature_list, but is a valid column name in self.df,
            # so add to feature list
            self._feature_list += [feature]

        return feature in self._feature_list

    def _obtain_feature_columns(
        self,
        features: list[str] | str | None,
    ) -> list[str]:
        """
        [Private] Obtains the column names storing the feature(s) requested.
        If a string or list is passed, it essentially just checks if it is an
        available feature that is stored in `_feature_list` and returns
        the feature(s) in a list.

        Parameters
        ----------
        features : list[str] | str | None
            If is a string, it must be in the list found in
            `_feature_list` attribute. If is a list,
            each item must be a string and it must be in the
            list found in `_feature_list` attribute.

        Returns
        -------
        list[str]
            List of column names which store the feature(s).

        Raises
        ------
        ValueError
            if `features` is a string, and it is not found in
            `_feature_list` attribute or if `features` is a
            list of strings, and one of the items
            is not found in `_feature_list` attribute.
        """
        if features is None:
            # no features are wanted, return an empty list
            features = []
        else:
            # convert to list of strings
            if isinstance(features, str):
                features = [features]

            if isinstance(features, list):
                # check each item in features is in self._feature_list
                # if it isn't, but is a column in self.df, it will add
                # it to self._feature_list
                for item in features:
                    if not self._check_feature_exists(feature=item):
                        raise ValueError(
                            f"{item} must be in `self.feature_list`: {self._feature_list}, "
                            "or a column in `self.df`."
                        )
            else:
                # features is neither None, a string or a list
                raise TypeError(
                    "`time_feature` must be either None, a string, or a list of strings."
                )

        return features

    def _pad_dataframe(
        self,
        df: pd.DataFrame,
        k: int,
        zero_padding: bool,
        colnames: list[str],
        features: list[str],
        id: int,
        pad_from_below: bool,
    ) -> pd.DataFrame:
        """
        [Private] If `padding_n > 0`, we pad `padding_n` number of entries
        to the dataframe (either by zeros if `zero_padding==True`, or by the last post
        in df if `zero_padding==False`). If `padding_n <= 0`, we don't need to pad
        and we simply return the last `k` entries (throws error if `k` is less than number
        of entries in `.df`).

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to pad with.
        k : int
            Number of items to keep.
        zero_padding : bool
            If True, will pad with zeros. Otherwise, pad with the latest
            text associated to the id.
        colnames : list[str]
            List of column names that we wish to keep from the dataframe.
        features : list[str]
            List of feature column names that we wish to keep from the dataframe.
        id : int
            Which id are we padding.
        pad_from_below: bool
            If True, will pad the path from below, otherwise pads the path from above.

        Returns
        -------
        pd.DataFrame
            Padded dataframe.

        Raises
        ------
        ValueError
            if k is not a positive integer.
        """
        if k <= 0:
            raise ValueError("`k` must be a positive integer.")
        columns = features + colnames + [self.id_column]
        if self.label_column is not None:
            columns += [self.label_column]

        amount_to_pad = k - len(df)
        if amount_to_pad > 0:
            # need to pad to fill up
            if zero_padding or len(df) == 0:
                # pad by having zero entries
                if self.label_column is not None:
                    # set labels to be -1 to indicate that they're padded values
                    data_dict = {
                        **dict.fromkeys(features, [0]),
                        **{c: [0] for c in colnames},
                        self.id_column: [id],
                        self.label_column: [-1],
                    }
                else:
                    # no label column to add
                    data_dict = {
                        **dict.fromkeys(features, [0]),
                        **{c: [0] for c in colnames},
                        self.id_column: [id],
                    }
                pad = pd.DataFrame(data_dict)
            else:
                if pad_from_below:
                    # pad by repeating the latest text
                    # (as we're padding from below)
                    pad = df[columns].tail(1)
                else:
                    # pad by repeating the first text
                    # (as we're padding from above)
                    pad = df[columns].head(1)

            if pad_from_below:
                df_padded = pd.concat(
                    [
                        df[columns],
                        pd.concat([pad] * amount_to_pad),
                    ]
                )
            else:
                df_padded = pd.concat(
                    [
                        pd.concat([pad] * amount_to_pad),
                        df[columns],
                    ]
                )

            return df_padded.reset_index(drop=True)

        return df[columns].tail(k).reset_index(drop=True)

    def _pad_id(
        self,
        k: int,
        zero_padding: bool,
        colnames: list[str],
        features: list[str],
        id: int,
        pad_from_below: bool,
    ) -> pd.DataFrame:
        """
        [Private] For a given id, the function slices the dataframe in .df
        by finding those with id_column == id and keeping only the columns
        found in colnames.
        The function returns a dataframe with k rows:
        - If the number of records with id_column == id is less than k, it "pads" the
        dataframe by adding in empty records (with label = -1 to indicate
        that they're padded) if `zero_padding=True`, or with either the first
        (if `pad_from_below=False`) or the last (if `pad_from_below=True`) item
        in the history if `zero_padding=False`
        - If the number of records with id_column == id is equal to k, it just returns
        the records with id_column == id.

        Parameters
        ----------
        k : int
            Number of items to keep.
        zero_padding : bool
            If True, will pad with zeros. Otherwise, pad with the latest
            text associated to the id.
        colnames : list[str]
            List of column names that we wish to keep from the dataframe.
        features : list[str]
            List of feature column names that we wish to keep from the dataframe.
        id : int
            Which id are we padding.
        pad_from_below: bool
            If True, will pad the path from below, otherwise pads the path from above.

        Returns
        -------
        pd.DataFrame
            Padded dataframe for a particular id.

        Raises
        ------
        ValueError
            if k is not a positive integer.
        """
        if k < 0:
            raise ValueError("`k` must be a positive integer.")
        history = self.df[self.df[self.id_column] == id]

        return self._pad_dataframe(
            df=history,
            k=k,
            zero_padding=zero_padding,
            colnames=colnames,
            features=features,
            id=id,
            pad_from_below=pad_from_below,
        )

    def _pad_history(
        self,
        k: int,
        zero_padding: bool,
        colnames: list[str],
        features: list[str],
        index: int,
        include_current_embedding: bool,
        pad_from_below: bool,
    ) -> pd.DataFrame:
        """
        [Private] For a particular index in .df, the function finds the history
        for that particular item by matching with its id_column.
        The function returns a dataframe with k rows:
        - If the number of records that occurred before this item is less than k,
        it "pads" the dataframe by adding in empty records (with label = -1 to indicate
        that they're padded) if `zero_padding=True`, or with either the first
        (if `pad_from_below=False`) or the last (if `pad_from_below=True`) item
        in the history if `zero_padding=False`
        - If the number of records that occurred before this item is equal to k, then
        it just returns the history

        Parameters
        ----------
        k : int
            Number of items to keep.
        zero_padding : bool
            If True, will pad with zeros. Otherwise, pad with the latest
            text associated to the id.
        colnames : list[str]
            List of column names that we wish to keep from the dataframe.
        features : list[str]
            List of features column names that we wish to keep from the dataframe.
        index : int
            Which index of the dataframe are we padding.
        pad_from_below: bool
            If True, will pad the path from below, otherwise pads the path from above.

        Returns
        -------
        pd.DataFrame
            Padded dataframe for a particular index of the dataframe by looking
            at the previous texts of a particular id.

        Raises
        ------
        ValueError
            if k is not a positive integer.
        ValueError
            if index is outside the range of indices of the dataframe ([0, 1, ..., len(.df)]).
        """
        if k < 0:
            raise ValueError("`k` must be a positive integer.")
        if index not in range(len(self.df)):
            raise ValueError("`index` is outside of [0, 1, ..., len(.df)].")

        # look at particular text at a given index
        text = self.df.iloc[index]
        id = text[self.id_column]
        timeline_index = text["timeline_index"]

        # obtain history for the piece of text
        if include_current_embedding:
            history = self.df[
                (self.df[self.id_column] == id)
                & (self.df["timeline_index"] <= timeline_index)
            ]
        else:
            history = self.df[
                (self.df[self.id_column] == id)
                & (self.df["timeline_index"] < timeline_index)
            ]

        return self._pad_dataframe(
            df=history,
            k=k,
            zero_padding=zero_padding,
            colnames=colnames,
            features=features,
            id=id,
            pad_from_below=pad_from_below,
        )

    @staticmethod
    def _standardise_pd(
        vec: pd.Series, method: str | None
    ) -> dict[str, pd.Series | Callable]:
        # standardised pandas series
        implemented = ["z_score", "sum_divide", "minmax", None]
        if method not in implemented:
            raise ValueError(f"`method`: {method} must be in {implemented}.")

        if method == "z_score":
            mean = vec.mean()
            std = vec.std()

            def transform(x):
                return (x - mean) / std

        elif method == "sum_divide":
            sum = vec.sum()

            def transform(x):
                return x / sum

        elif method == "minmax":
            minimum = vec.min()
            maximum = vec.max()

            def transform(x):
                return (x - minimum) / (maximum - minimum)

        elif method is None:

            def transform(x):
                return x

        return {"standardised_pd": transform(vec), "transform": transform}

    def pad(
        self,
        pad_by: str,
        method: str = "k_last",
        zero_padding: bool = True,
        k: int = 5,
        features: list[str] | str | None = None,
        standardise_method: list[str] | str | None = None,
        embeddings: str = "full",
        include_current_embedding: bool = True,
        pad_from_below: bool = True,
    ) -> np.array:
        """
        Creates an array which stores the path.
        We create a path for each id in id_column if `pad_by="id"`
        (by constructing a path of the embeddings of the texts associated to each id),
        or for each item in `.df` if `pad_by="history"`
        (by constructing a path of the embeddings of the previous texts).

        We can decide how long our path is by letting `method="k_last` and specifying `k`.
        Alternatively, we can set `method="max"`, which sets the length of the path
        by setting `k` to be the largest number of texts associated to an individual id.

        The function "pads" if there aren't enough texts to fill in (e.g. if requesting for
        the last 5 posts for an id, but there are less than 5 posts available),
        by adding empty records (if `zero_padding=True`)
        or by the last previous text (if `zero_padding=False`). This ensures that each
        path has the same number of data points.

        Parameters
        ----------
        pad_by : str
            How to construct the path. Options are:

            - "id": constructs a path of the embeddings of the texts associated to each id
            - "history": constructs a path by looking at the embeddings of the previous texts
              for each text

        method : str, optional
            How long the path is, default "k_last". Options are:

            - "k_last": specifying the length of the path through the choice of `k` (see below)
            - "max": the length of the path is chosen by looking at the largest number
              of texts associated to an individual id in `.id_column`

        zero_padding : bool, optional
            If True, will pad with zeros. Otherwise, pad with the latest
            text associated to the id.
        k : int, optional
            The requested length of the path, default 5. This is ignored if `method="max"`.
        features : list[str] | str | None, optional
            Which feature(s) to keep. If None, then doesn't keep any.
        standardise_method : list[str] | str | None, optional
            If not None, applies standardisation to the features, default None.
            If a list is passed, must be the same length as `features`. Options:

            - "z_score": transforms by subtracting the mean and dividing by standard deviation
            - "sum_divide": transforms by dividing by the sum
            - "minmax": transform by return (x-min(x)) / (max(x)-min(x)) where x
              is the vector to standardise

        embeddings : str, optional
            Which embeddings to keep, by default "full". Options:

            - "dim_reduced": dimension reduced embeddings
            - "full": full embeddings
            - "both": keeps both dimension reduced and full embeddings

        include_current_embedding : bool, optional
            If `pad_by="history"`, this determines whether or not the embedding for the
            text is included in it's history, by default True. If `pad_by="id"`,
            this argument is ignored.
        pad_from_below: bool, optional
            If True, will pad the path from below, otherwise pads the path from above,
            by default True.

        Returns
        -------
        np.array
            3 dimension array of the path:
                - First dimension is ids (if `pad_by="id"`) or each text (if `pad_by="history"`)
                - Second dimension is the associated texts
                - Third dimension are the features (e.g. embeddings /
                  dimension reduced embeddings, time features)

        """
        if self.verbose:
            print(
                "[INFO] Padding ids and storing in `.df_padded` and `.array_padded` attributes."
            )

        if pad_by not in ["id", "history"]:
            raise ValueError("`pad_by` must be either 'id' or 'history'.")

        self.pad_method = pad_by

        # obtain id_column counts
        id_counts = self.df[self.id_column].value_counts(sort=False)
        # determine padding length
        if method == "k_last":
            # use k that was passed in
            pass
        elif method == "max":
            # let k be the largest number of items associated to an id
            k = id_counts.max()
        else:
            raise ValueError("`method` must be either 'k_last' or 'max'.")

        # obtain feature colnames
        feature_colnames = self._obtain_feature_columns(features=features)
        if len(feature_colnames) > 0:
            if isinstance(standardise_method, str):
                standardise_method = [standardise_method] * len(feature_colnames)
            elif isinstance(standardise_method, list) and (
                len(standardise_method) != len(feature_colnames)
            ):
                raise ValueError(
                    "if `standardise_method` is a list, it must have the same length "
                    f"as the number of features requested: {len(feature_colnames)}."
                )

        if standardise_method is not None:
            # standardises the features in .df
            self.standardise_transform = {}
            for i in range(len(feature_colnames)):
                standardise = self._standardise_pd(
                    vec=self.df[feature_colnames[i]], method=standardise_method[i]
                )
                self.standardise_transform[feature_colnames[i]] = standardise[
                    "transform"
                ]
                self.df[feature_colnames[i]] = standardise["standardised_pd"]

        # obtain colnames of embeddings
        colnames = self._obtain_embedding_colnames(embeddings=embeddings)

        if pad_by == "id":
            # pad each of the ids in id_column and store them in a list
            padded_dfs = [
                self._pad_id(
                    k=k,
                    zero_padding=zero_padding,
                    colnames=colnames,
                    features=feature_colnames,
                    id=id,
                    pad_from_below=pad_from_below,
                )
                for id in tqdm(id_counts.index)
            ]
            self.df_padded = pd.concat(padded_dfs).reset_index(drop=True)
        elif pad_by == "history":
            # pad each of the ids in id_column and store them in a list
            padded_dfs = [
                self._pad_history(
                    k=k,
                    zero_padding=zero_padding,
                    colnames=colnames,
                    features=feature_colnames,
                    index=index,
                    include_current_embedding=include_current_embedding,
                    pad_from_below=pad_from_below,
                )
                for index in tqdm(range(len(self.df)))
            ]
            self.df_padded = pd.concat(padded_dfs).reset_index(drop=True)

        if pad_by == "id":
            self.array_padded = np.array(self.df_padded).reshape(
                len(id_counts), k, len(self.df_padded.columns)
            )
        elif pad_by == "history":
            self.array_padded = np.array(self.df_padded).reshape(
                len(self.df), k, len(self.df_padded.columns)
            )

        return self.array_padded

    def get_time_feature(
        self,
        time_feature: str = "timeline_index",
        standardise_method: str | None = None,
    ) -> dict[str, np.array | Callable | None]:
        """
        Returns a `np.array` object of the time_feature that is requested
        (the string passed has to be one of the strings in `._feature_list`).

        Parameters
        ----------
        time_feature : str, optional
            Which time feature to obtain `np.array` for, by default "timeline_index".
        standardise_method : str | None, optional
            If not None, applies standardisation to the time features, default None. Options:
            - "z_score": transforms by subtracting the mean and dividing by standard deviation
            - "sum_divide": transforms by dividing by the sum
            - "minmax": transform by return (x-min(x)) / (max(x)-min(x)) where x
              is the vector to standardise

        Returns
        -------
        dict[str, np.array | Callable | None]
            Dictionary where dict["time_feature"] stores the `np.array` of the time feature,
            and dict["transform"] is the function to transform new data using the standardisation
            applied (if `standardise_method` is not None), or None.

        Raises
        ------
        ValueError
            if `time_feature` is not in the possible time_features
            (can be found in `._feature_list` attribute).
        """
        if time_feature not in self._feature_list:
            raise ValueError(
                f"`time_feature` '{time_feature}' should be in {self._feature_list}."
            )

        if not self.time_features_added:
            self.set_time_features()

        if standardise_method is not None:
            # standardises the time features in .df_padded
            self.standardise_transform = {}
            standardise = self._standardise_pd(
                vec=self.df[time_feature], method=standardise_method
            )
            return {
                "time_feature": np.array(standardise["standardised_pd"]),
                "transform": standardise["transform"],
            }

        return {"time_feature": np.array(self.df[time_feature]), "transform": None}

    def get_path(self, include_features: bool = True) -> np.array:
        """
        Returns a `np.array` object of the path.
        Includes the features by default (if they are present after the padding).

        Parameters
        ----------
        include_features : bool, optional
            Whether or not to keep the features, by default True.

        Returns
        -------
        np.array
            Path.

        Raises
        ------
        ValueError
            if `self.array_padded` is `None`. In this case, need to call `.pad()` first.
        """
        if self.array_padded is None:
            raise ValueError("Need to first call to create the path `.pad()`.")

        # first strip away the id_column and label_column (if exists)
        if self.label_column is not None:
            # remove last two columns in the third dimension
            # (which store id_column and label_column)
            path = self.array_padded[:, :, :-2]
        else:
            # there are no labels, so just remove last column in third dimension
            # (which stores id_column)
            path = self.array_padded[:, :, :-1]

        if not include_features:
            # computes how many features there are currently
            # (which occur in the first n_features columns)
            n_features = len(
                [item for item in self._feature_list if item in self.df_padded]
            )
            # removes any features (if they're present)
            path = path[:, :, n_features:]

        return path.astype("float")

    def get_embeddings(self, reduced_embeddings: bool = False) -> np.array:
        """
        Returns a `np.array` object of the embeddings.

        Parameters
        ----------
        reduced_embeddings : bool, optional
            If True, returns `np.array` of dimension reduced embeddings,
            by default False.

        Returns
        -------
        np.array
            Embeddings.
        """
        if reduced_embeddings:
            if self.embeddings_reduced is None:
                raise ValueError(
                    "There were no reduced embeddings passed into the class."
                )
            colnames = [col for col in self.df.columns if re.match(r"^d\w*[0-9]", col)]
        else:
            colnames = [col for col in self.df.columns if re.match(r"^e\w*[0-9]", col)]

        return np.array(self.df[colnames].values)

    def get_torch_path_for_SWNUNetwork(
        self,
        include_features_in_path: bool,
        include_features_in_input: bool,
        include_embedding_in_input: bool,
        reduced_embeddings: bool = False,
    ) -> tuple[torch.tensor, int]:
        """
        Returns a `torch.tensor` object that can be passed into `nlpsig_networks.SWNUNetwork` model.

        Parameters
        ----------
        include_features_in_path : bool
            Whether or not to keep the additional features (e.g. time features) within the path.
        include_features_in_input : bool
            Whether or not to concatenate the additional features into the feed-forward neural
            network in the `nlpsig_networks.SWNUNetwork` model.
        include_embedding_in_input : bool
            Whether or not to concatenate the embeddings into the feed-forward neural
            network in the `nlpsig_networks.SWNUNetwork` model.
            If we created a path for each item in the dataset, we will concatenate
            the embeddings in `.embeddings` (if `reduced_embeddings=False`) or
            the embeddings in `.reduced_embeddings` (if `reduced_embeddings=True`).
            If we created a path for each id in `.id_column`, then we concatenate
            the embeddings in `.pooled_embeddings`.
        reduced_embeddings : bool, optional
            Whether or not to concatenate the dimension reduced embeddings, by default False.
            This is ignored if we created a path for each if in `.id_column`,
            i.e. `.pad_method='id'`.

        Returns
        -------
        Tuple[torch.tensor, int]
            First element is a tensor to be inputted to `nlpsig_networks.SWNUNetwork` model.
            Second element is the number of channels in the path for which
            we compute the path signature for in `nlpsig_networks.SWNUNetwork`.
        """
        if self.array_padded is None:
            raise ValueError("Need to first call to create the path `.pad()`.")

        # obtains a torch tensor which can be inputted into SWNUNetwork
        # computes how many features there are currently
        # (which occur in the first n_features columns)
        n_features = len(
            [item for item in self._feature_list if item in self.df_padded]
        )

        if include_embedding_in_input:
            # repeat the embeddings which will be concatenated to the path later
            if self.pad_method == "id":
                if self.verbose:
                    print(
                        f"[INFO] The path was created for each {self.id_column} in the dataframe, "
                        "so to include embeddings in the FFN input, we concatenate the "
                        "pooled embeddings."
                    )

                if self.pooled_embeddings is None:
                    raise ValueError(
                        "There were no pooled embeddings passed into the class."
                    )
                if self.array_padded.shape[0] != self.pooled_embeddings.shape[0]:
                    raise ValueError(
                        "If want to include the pooled embeddings in the FFN input, the path "
                        "(found in `.array_padded`) must have the same number of "
                        "samples as there are pooled embeddings, i.e `.array_padded.shape[0]` "
                        "must equal `.pooled_embeddings.shape[0]`."
                    )
                emb = torch.from_numpy(self.pooled_embeddings.astype("float")).float()
            elif self.pad_method == "history":
                if self.verbose:
                    print(
                        "[INFO] The path was created for each item in the dataframe, "
                        "by looking at its history, so to include embeddings in the FFN input, "
                        "we concatenate the embeddings for each sentence / text."
                    )

                if reduced_embeddings:
                    if self.embeddings_reduced is None:
                        raise ValueError(
                            "There were no reduced embeddings passed into the class."
                        )
                    if self.array_padded.shape[0] != self.embeddings_reduced.shape[0]:
                        raise ValueError(
                            "If want to include reduced embeddings in the FFN input, the path "
                            "(found in `.array_padded`) must have the same number of "
                            "samples as there are embeddings, i.e `.array_padded.shape[0]` "
                            "must equal `.embeddings_reduced.shape[0]`."
                        )
                    emb = torch.from_numpy(
                        self.embeddings_reduced.astype("float")
                    ).float()
                else:
                    if self.array_padded.shape[0] != self.embeddings.shape[0]:
                        raise ValueError(
                            "If want to include the full embeddings in the FFN input, the path "
                            "(found in `.array_padded`) must have the same number of "
                            "samples as there are embeddings, i.e `.array_padded.shape[0]` "
                            "must equal `.embeddings.shape[0]`."
                        )
                    else:
                        emb = torch.from_numpy(self.embeddings.astype("float")).float()
            repeat_emb = (
                emb.unsqueeze(2)
                .repeat(1, 1, self.array_padded.shape[1])
                .transpose(1, 2)
            )

        if include_features_in_path:
            # make sure path includes the features
            path = torch.from_numpy(self.get_path(include_features=True))
            input_channels = path.shape[2]
            if include_features_in_input:
                # need to repeat the feature columns
                # if there are no features, then we don't need to repeat anything
                if n_features == 1:
                    path = torch.cat([path, path[:, :, 0].unsqueeze(2)], dim=2)
                elif n_features > 1:
                    path = torch.cat([path, path[:, :, 0:n_features]], dim=2)
        else:
            if include_features_in_input:
                # path doesn't need to include the features
                # but we still want to include them in the input to the FFN for classification
                path = torch.from_numpy(self.get_path(include_features=True))
                input_channels = path.shape[2] - n_features
                # need to move features to the end of the path
                # if there are no features, then we don't need to move anything
                if n_features == 1:
                    path = torch.cat(
                        [path[:, :, n_features:], path[:, :, 0].unsqueeze(2)],
                        dim=2,
                    )
                elif n_features > 1:
                    path = torch.cat(
                        [path[:, :, n_features:], path[:, :, 0:n_features]],
                        dim=2,
                    )
            else:
                # path doesn't need to include the features
                # and don't need to include them in the input to the FFN for classification
                path = torch.from_numpy(self.get_path(include_features=False))
                input_channels = path.shape[2]

        if include_embedding_in_input:
            path = torch.cat([path, repeat_emb], dim=2)

        return path, input_channels

    def check_history_length_for_SeqSigNet(
        self, shift: int, window_size: int, n: int
    ) -> bool:
        """
        Helper function to detemine whether or not the path created (by `.pad()`)
        has history length (k) long enough to create a tensor for
        SeqSigNet network.

        In particular, for a given shift, window_size and n, we must have
        `history_length == shift * n + (window_size - shift)`.

        Parameters
        ----------
        shift : int
            Amount we are shifting the window.
        window_size : int
            Size of the window we use over the texts.
        n : int
            Number of units we wish to use in SeqSigNet.

        Returns
        -------
        bool
            Whether or not the history length in the path created by `.pad()`
            satisfies the requested configuration in SeqSigNet.

        Raises
        ------
        ValueError
            If a path hasn't been created yet using `.pad()`.
        """
        if self.array_padded is None:
            raise ValueError("Need to first call to create the path `.pad()`.")

        required_history_length = shift * n + (window_size - shift)
        if self.array_padded.shape[1] != required_history_length:
            # required history length not met
            if self.verbose:
                print(
                    f"A history length of size {required_history_length} is required, "
                    f"but we have history length size of {self.array_padded.shape[1]}"
                )
            return False

        # we have the required history length
        return True

    def get_torch_path_for_SeqSigNet(
        self,
        shift: int,
        window_size: int,
        n: int,
        include_features_in_path: bool,
        include_features_in_input: bool,
        include_embedding_in_input: bool,
        reduced_embeddings: bool = False,
    ) -> tuple[torch.tensor, int]:
        """
        Returns a `torch.tensor` object that can be passed into `nlpsig_networks.SeqSigNet` model.

        Parameters
        ----------
        shift : int
            Amount we are shifting the window.
        window_size : int
            Size of the window we use over the texts.
        n : int
            Number of units we wish to use in SeqSigNet.
        include_features_in_path : bool
            Whether or not to keep the additional features (e.g. time features) within the path.
        include_features_in_input : bool
            Whether or not to concatenate the additional features into the feed-forward neural
            network in the `nlpsig_networks.SeqSigNet` model.
        include_embedding_in_input : bool
            Whether or not to concatenate the embeddings into the feed-forward neural
            network in the `nlpsig_networks.SeqSigNet` model.
            If we created a path for each item in the dataset, we will concatenate
            the embeddings in `.embeddings` (if `reduced_embeddings=False`) or
            the embeddings in `.reduced_embeddings` (if `reduced_embeddings=True`).
            If we created a path for each id in `.id_column`, then we concatenate
            the embeddings in `.pooled_embeddings`.
        reduced_embeddings : bool, optional
            Whether or not to concatenate the dimension reduced embeddings, by default False.
            This is ignored if we created a path for each if in `.id_column`,
            i.e. `.pad_method='id'`.

        Returns
        -------
        Tuple[torch.tensor, int]
            First element is a tensor to be inputted to `nlpsig_networks.SeqSigNet` model.
            Second element is the number of channels in the path for which
            we compute the path signature for in `nlpsig_networks.SeqSigNet`.

        Raises
        ------
        ValueError
            If a path hasn't been created yet using `.pad()`.
        """
        if not self.check_history_length_for_SeqSigNet(
            shift=shift, window_size=window_size, n=n
        ):
            raise ValueError(
                "We do not have the required history length for "
                "this configuration of shift, window_size and n"
            )

        # obtain 3 dimensional tensor with dimensions [batch, history, channels]
        swnu_path, input_channels = self.get_torch_path_for_SWNUNetwork(
            include_features_in_path=include_features_in_path,
            include_features_in_input=include_features_in_input,
            include_embedding_in_input=include_embedding_in_input,
            reduced_embeddings=reduced_embeddings,
        )

        # taking windows of the path created (determined by shift, window_size, n)
        # obtain 4 dimensional tensor with dimsnions [batch, history, channels, units]
        seqsignnet_path = torch.stack(
            [
                swnu_path[:, (i * shift) : (i * shift + window_size), :]
                for i in range(n)
            ],
            dim=3,
        )

        return seqsignnet_path, input_channels
