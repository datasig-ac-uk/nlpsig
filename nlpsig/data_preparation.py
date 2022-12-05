import re
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


class PrepareData:
    """
    Class to prepare dataset for computing signatures
    """

    def __init__(
        self,
        dataset_df: pd.DataFrame,
        embeddings: np.array,
        embeddings_reduced: Optional[np.array] = None,
        id_column: Optional[str] = None,
        labels_column: Optional[str] = None,
    ):
        """
        Class to prepare dataset for computing signatures

        Parameters
        ----------
        dataset_df : pd.DataFrame
            Dataset as a pandas dataframe
        embeddings : np.array
            Embeddings for each of the items in dataset_df
        embeddings_reduced : Optional[np.array], optional
            Dimension reduced embeddings, by default None
        id_column : Optional[str]
            Name of the column which identifies each of the text, e.g.
            - "text_id" (if each item in dataset_df is a word or sentence from a particular text),
            - "user_id" (if each item in dataset_df is a post from a particular user)
            - "timeline_id" (if each item in dataset_df is a post from a particular time)
            If None, it will create a dummy id_column named "dummy_id" and fill with zeros
        labels_column : Optional[str]
            Name of the column which are corresponds to the labels of the data

        Raises
        ------
        ValueError
            if dataset_df and embeddings does not have the same number of rows
        ValueError
            if dataset_df and embeddings_reduced does not have the same number of rows
            (if embeddings_reduced is provided)
        """
        # perform checks that dataset_df have the right column names to work with
        if dataset_df.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "dataset_df, embeddings and embeddings_reduced "
                + "should have the same number of rows"
            )
        if embeddings_reduced is not None:
            if dataset_df.shape[0] != embeddings_reduced.shape[0]:
                raise ValueError(
                    "dataset_df, embeddings and embeddings_reduced "
                    + "should have the same number of rows"
                )
        self.dataset_df = dataset_df
        self.id_column = id_column
        self.label_column = labels_column
        self.embeddings = embeddings
        self.embeddings_reduced = embeddings_reduced
        # obtain modelling dataframe
        self.df = None
        self.df = self._get_modeling_dataframe()
        # obtain time features
        self._time_feature_choices = []
        self.time_features_added = False
        self.df = self._set_time_features()
        self.df_padded = None
        self.array_padded = None

    def _get_modeling_dataframe(self) -> pd.DataFrame:
        """
        [Private] Combines original dataset_df with the sentence
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
        else:
            embedding_df = pd.DataFrame(
                self.embeddings,
                columns=[f"e{i+1}" for i in range(self.embeddings.shape[1])],
            )
            if self.embeddings_reduced is not None:
                embeddings_reduced_df = pd.DataFrame(
                    self.embeddings_reduced,
                    columns=[
                        f"d{i+1}" for i in range(self.embeddings_reduced.shape[1])
                    ],
                )
                df = pd.concat(
                    [
                        self.dataset_df.reset_index(drop=True),
                        embeddings_reduced_df,
                        embedding_df,
                    ],
                    axis=1,
                )
            else:
                df = pd.concat(
                    [self.dataset_df.reset_index(drop=True), embedding_df],
                    axis=1,
                )
            if self.id_column is None:
                self.id_column = "dummy_id"
                print(
                    f"[INFO] No id_column was passed, so setting id_column to '{self.id_column}'"
                )
            if self.id_column not in self.dataset_df.columns:
                # set default value to id_column
                print(
                    f"[INFO] There is no column in .dataset_df called '{self.id_column}'. "
                    + "Adding a new column named '{self.id_column}' of zeros"
                )
                df[self.id_column] = 0
            return df

    @staticmethod
    def _time_fraction(x: pd.Timestamp) -> float:
        """
        [Private] Converts a date, x, as a fraction of the year.

        Parameters
        ----------
        x : pd.Timestamp
            Date

        Returns
        -------
        float
            The date as a fraction of the year
        """
        # compute how many seconds the date is into the year
        x_year_start = pd.Timestamp(x.year, 1, 1)
        seconds_into_cal_year = abs(x - x_year_start).total_seconds()
        # compute the time fraction into the year
        time_frac = seconds_into_cal_year / (365 * 24 * 60 * 60)
        return x.year + time_frac

    def _set_time_features(self) -> pd.DataFrame:
        """
        [Private] Updates the dataframe in .df to include time features:
        - `time_encoding`: the date as a fraction of the year
           (only if 'datetime' is a column in .df dataframe)
        - `time_diff`: the difference in time (in minutes) between successive records
           (only if 'datetime' is a column in .df dataframe)
        - `timeline_index`: the index of each post for each id

        Returns
        -------
        pd.DataFrame
            Updated dataframe with time features
        """
        if self.time_features_added:
            print("Time features have already been added")
            return
        print("[INFO] Adding time feature columns into dataframe in .df")
        if "datetime" in self.df.columns:
            # obtain time encoding by computing the fraction of year it is in
            self._time_feature_choices += ["time_encoding", "time_diff"]
            self.df["time_encoding"] = self.df["datetime"].map(
                lambda t: self._time_fraction(t)
            )
            # sort by the id and the date
            self.df = self.df.sort_values(by=[self.id_column, "datetime"]).reset_index(
                drop=True
            )
            # calculate time difference between posts
            self.df["time_diff"] = 0
            for i in range(1, len(self.df)):
                if (
                    self.df[self.id_column].iloc[i]
                    != self.df[self.id_column].iloc[i - 1]
                ):
                    diff = self.df["datetime"].iloc[i] - self.df["datetime"].iloc[i - 1]
                    diff_in_mins = diff.total_seconds() / 60
                    self.df["time_diff"][i] = diff_in_mins
        else:
            print(
                "[INFO] datetime is not a column in .df, "
                + "so only 'timeline_index' is being added"
            )
            print(
                "[INFO] as datetime is not a column in .df, "
                + "we assume that the data is ordered by time with respect to the id"
            )
        # assign index for each post in each timeline
        self._time_feature_choices += ["timeline_index"]
        self.df["timeline_index"] = 0
        first_index = 0
        for id in set(self.df[self.id_column]):
            # obtain the indices for this id
            id_len = sum(self.df[self.id_column] == id)
            last_index = first_index + id_len
            # assign indices for each post in this id from 1 to id_len
            self.df["timeline_index"][first_index:last_index] = list(
                range(1, id_len + 1)
            )
            first_index = last_index
        self.time_features_added = True

        return self.df

    def _obtain_colnames(self, embeddings: str) -> List[str]:
        """
        [Private] Obtains the column names storing the embeddings.

        Parameters
        ----------
        embeddings : str
            Options are:
            - "dim_reduced": dimension reduced embeddings
            - "full": full embeddings
            - "both": concatenation of dimension reduced and full embeddings

        Returns
        -------
        List[str]
            List of column names which store the embeddings

        Raises
        ------
        ValueError
            if embeddings is not either of 'dim_reduced', 'full', or 'both'
        """
        if embeddings not in ["dim_reduced", "full", "both"]:
            raise ValueError(
                "embeddings must be either 'dim_reduced', 'full', or 'both'"
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

    def _obtain_time_feature_columns(
        self,
        time_feature: Optional[Union[List[str], str]],
    ) -> List[str]:
        """
        [Private] Obtains the column names storing the time features requested.

        Parameters
        ----------
        time_feature : Optional[Union[List[str], str]]
            If is a string, it must be the list found in
            `_time_feature_choices` attribute. If is a list,
            each item must be a string and it must be in the
            list found in `_time_feature_choices` attribute

        Returns
        -------
        List[str]
            List of column names which store the time features

        Raises
        ------
        ValueError
            if `time_feature` is a string, and it is not found in `_time_feature_choices`
        ValueError
            if `time_feature` is a list of strings, and one of the items
            is not found in `_time_feature_choices`
        """
        if time_feature is None:
            time_feature = []
        else:
            if not self.time_features_added:
                self.set_time_features()
            if isinstance(time_feature, str):
                if time_feature not in self._time_feature_choices:
                    raise ValueError(
                        "If time_feature is a string, it must "
                        + f"be in {self._time_feature_choices}"
                    )
                else:
                    time_feature = [time_feature]
            elif isinstance(time_feature, list):
                if not all(
                    [item in self._time_feature_choices for item in time_feature]
                ):
                    raise ValueError(
                        f"Each item in time_feature should be in {self._time_feature_choices}"
                    )
            else:
                raise ValueError(
                    "time_feature must be either None, a string, or a list of strings"
                )
        return time_feature

    def _pad_id(
        self,
        k: int,
        zero_padding: bool,
        colnames: List[str],
        id_counts: pd.Series,
        id: int,
        time_feature: List[str],
    ) -> pd.DataFrame:
        """
        [Private] For a given id, the function slices the dataframe in .df
        by finding those with id_column == id and keeping only the columns
        found in colnames.
        The function returns a dataframe with k rows:
        - If the number of records with id_column == id is less than k, it "pads" the
        dataframe by adding in empty records (with label = -1 to indicate they're padded)
        - If the number of records with id_column == id is equal to k, it just returns
        the records with id_column == id

        Parameters
        ----------
        k : int
            Number of items to keep
        zero_padding : bool
            If True, will pad with zeros. Otherwise, pad with the latest
            text associated to the id
        colnames : List[str]
            List of column names that we wish to keep from the dataframe
        id_counts : pd.Series
            The number of records in associated to each id_column
        id : int
            Which id are we padding
        time_feature : List[str]
            List of time feature column names that we wish to keep from the dataframe

        Returns
        -------
        pd.DataFrame
            Padded dataframe for a particular id

        Raises
        ------
        ValueError
            if k is not a positive integer
        """
        if k < 0:
            raise ValueError("k must be a positive integer")
        padding_n = k - id_counts[id]
        columns = [self.id_column] + time_feature + colnames
        if self.label_column is not None:
            columns += [self.label_column]
        if padding_n > 0:
            # need to pad to fill up
            if zero_padding:
                # pad by having zero entries
                if self.label_column is not None:
                    data_dict = {
                        **{self.id_column: [id], self.label_column: [-1]},
                        **dict.fromkeys(time_feature, [0]),
                        **{c: [0] for c in colnames},
                    }
                else:
                    data_dict = {
                        **{self.id_column: [id]},
                        **dict.fromkeys(time_feature, [0]),
                        **{c: [0] for c in colnames},
                    }
                df_padded = pd.concat(
                    [
                        pd.concat([pd.DataFrame(data_dict)] * padding_n),
                        self.df[self.df[self.id_column] == id][columns],
                    ]
                )
            else:
                # pad by repeating the latest text
                latest_text = self.df[self.df[self.id_column] == id][columns].tail(1)
                df_padded = pd.concat(
                    [
                        self.df[self.df[self.id_column] == id][columns],
                        pd.concat([latest_text] * padding_n),
                    ]
                )
            return df_padded.reset_index(drop=True)
        else:
            return (
                self.df[self.df[self.id_column] == id][columns]
                .tail(k)
                .reset_index(drop=True)
            )

    def _pad_history(
        self,
        k: int,
        zero_padding: bool,
        colnames: List[str],
        index: int,
        time_feature: Union[List[str], None],
    ) -> pd.DataFrame:
        """
        [Private]

        Parameters
        ----------
        k : int
            Number of items to keep
        zero_padding : bool
            If True, will pad with zeros. Otherwise, pad with the latest
            text associated to the id
        colnames : List[str]
            List of column names that we wish to keep from the dataframe
        index : int
            Which index of the dataframe are we padding
        time_feature : List[str]
            List of time feature column names that we wish to keep from the dataframe

        Returns
        -------
        pd.DataFrame
            Padded dataframe for a particular index of the dataframe by looking
            at the previous texts of a particular id

        Raises
        ------
        ValueError
            if k is not a positive integer
        ValueError
            if index is outside the range of indicies of the dataframe ([0, 1, ..., len(.df)])
        """
        if k < 0:
            raise ValueError("k must be a positive integer")
        if index not in range(len(self.df)):
            raise ValueError("index is outside of [0, 1, ..., len(.df)]")
        # look at particular text at a given index
        text = self.df.iloc[index]
        id = text[self.id_column]
        timeline_index = text["timeline_index"]
        # obtain history for the piece of text
        history = self.df[
            (self.df[self.id_column] == id)
            & (self.df["timeline_index"] < timeline_index)
        ]
        padding_n = k - len(history)
        columns = [self.id_column] + time_feature + colnames
        if self.label_column is not None:
            columns += [self.label_column]
        if padding_n > 0:
            # need to pad to fill up
            if zero_padding or len(history) == 0:
                # pad by having zero entries
                if self.label_column is not None:
                    data_dict = {
                        **{self.id_column: [id], self.label_column: [-1]},
                        **dict.fromkeys(time_feature, [0]),
                        **{c: [0] for c in colnames},
                    }
                else:
                    data_dict = {
                        **{self.id_column: [id]},
                        **dict.fromkeys(time_feature, [0]),
                        **{c: [0] for c in colnames},
                    }
                df_padded = pd.concat(
                    [
                        pd.concat([pd.DataFrame(data_dict)] * padding_n),
                        history[columns],
                    ]
                )
            else:
                # pad by repeating the latest text
                latest_text = history[columns].tail(1)
                df_padded = pd.concat(
                    [
                        history[columns],
                        pd.concat([latest_text] * padding_n),
                    ]
                )
            return df_padded.reset_index(drop=True)
        else:
            return history[columns].tail(k).reset_index(drop=True)

    def pad(
        self,
        pad_by: str,
        method: str = "k_last",
        zero_padding: bool = True,
        k: int = 5,
        time_feature: Optional[Union[List[str], str]] = None,
        embeddings: str = "full",
    ) -> Tuple[pd.DataFrame, np.array]:
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
        method : str
            How long the path is, default "k_last". Options are:
            - "k_last": specifying the length of the path through the choice of `k` (see below)
            - "max": the length of the path is chosen by looking at the largest number
              of texts associated to an individual id in `.id_column`
        zero_padding : bool
            If True, will pad with zeros. Otherwise, pad with the latest
            text associated to the id
        k : int
            The requested length of the path, default 5. This is ignored if `method="max"`
        time_feature : Optional[Union[List[str], str]]
            Which time feature(s) to keep. If None, then doesn't keep any
        embeddings : str, optional
            Which embeddings to keep, by default "full". Options:
            - "dim_reduced": dimension reduced embeddings
            - "full": full embeddings
            - "both": keeps both dimension reduced and full embeddings

        Returns
        -------
        np.array
            3 dimension array of the path:
            - First dimension is ids (if `pad_by="id"`)
              or each text (if `pad_by="history"`)
            - Second dimension is the associated texts
            - Third dimension are the features (e.g. embeddings /
              dimension reduced embeddings, time features)
        """
        print(
            "[INFO] Padding ids and storing in .df_padded and .array_padded attributes"
        )
        if pad_by not in ["id", "history"]:
            raise ValueError("pad_by must be either 'id' or 'history'")
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
            raise ValueError("method must be either 'k_last' or 'max'")
        # obtain time feature colnames
        time_feature_colnames = self._obtain_time_feature_columns(
            time_feature=time_feature
        )
        # obtain colnames of embeddings
        colnames = self._obtain_colnames(embeddings=embeddings)
        if pad_by == "id":
            # pad each of the ids in id_column and store them in a list
            padded_dfs = [
                self._pad_id(
                    k=k,
                    zero_padding=zero_padding,
                    colnames=colnames,
                    id_counts=id_counts,
                    id=id,
                    time_feature=time_feature_colnames,
                )
                for id in id_counts.index
            ]
            self.df_padded = pd.concat(padded_dfs).reset_index(drop=True)
            self.array_padded = np.array(self.df_padded).reshape(
                len(id_counts), k, len(self.df_padded.columns)
            )
            return self.array_padded
        elif pad_by == "history":
            # pad each of the ids in id_column and store them in a list
            padded_dfs = [
                self._pad_history(
                    k=k,
                    zero_padding=zero_padding,
                    colnames=colnames,
                    index=index,
                    time_feature=time_feature_colnames,
                )
                for index in range(len(self.df))
            ]
            self.df_padded = pd.concat(padded_dfs).reset_index(drop=True)
            self.array_padded = np.array(self.df_padded).reshape(
                len(self.df), k, len(self.df_padded.columns)
            )
            return self.array_padded

    def get_torch_time_feature(
        self, time_feature: str = "timeline_index", standardise: bool = True
    ) -> torch.tensor:
        """
        Returns a `torch.tensor` object of the time_feature that is requested
        (the string passed has to be one of the strings in `._time_feature_choices`).

        Parameters
        ----------
        time_feature : str, optional
            Which time feature to obtain `torch.tensor` for, by default "timeline_index"
        standardise : bool, optional
            Whether or not to standardise the time feature, by default True

        Returns
        -------
        torch.tensor
            Time feature

        Raises
        ------
        ValueError
            if `time_feature` is not in the possible time_features
            (can be found in `._time_feature_choices` attribute)
        """
        if time_feature not in self._time_feature_choices:
            raise ValueError(f"time_feature should be in {self._time_feature_choices}")
        if not self.time_features_added:
            self.set_time_features()
        if standardise:
            feature_mean = self.df[time_feature].mean()
            feature_std = self.df[time_feature].std()
            feature = (self.df[[time_feature]].values - feature_mean) / feature_std
            return torch.tensor(feature)
        else:
            return torch.tensor(self.df[[time_feature]])

    def get_torch_path(self, include_time_features: bool = True) -> torch.tensor:
        """
        Returns a torch.tensor object of the path
        Includes the time features by default (if they are present after the padding)

        Parameters
        ----------
        include_time_features : bool, optional
            Whether or not to keep the time features, by default True

        Returns
        -------
        torch.tensor
            Path

        Raises
        ------
        ValueError
            if `self.array_padded` is `None`. In this case, need to call `.pad()` first.
        """
        if self.array_padded is None:
            raise ValueError("Need to first call .pad()")
        if include_time_features:
            # includes the time features (if they're present)
            return torch.from_numpy(self.array_padded[:, :, 2:])
        else:
            n_time_features = len(
                [item for item in self._time_feature_choices if item in self.df_padded]
            )
            index_from = n_time_features + 2
            return torch.from_numpy(self.array_padded[:, :, index_from:])

    def get_torch_embeddings(self, reduced_embeddings: bool = False) -> torch.tensor:
        """
        Returns a `torch.tensor` object of the the embeddings

        Parameters
        ----------
        reduced_embeddings : bool, optional
            If True, returns `torch.tensor` of dimension reduced embeddings,
            by default False

        Returns
        -------
        torch.tensor
            Embeddings
        """
        if reduced_embeddings:
            colnames = [col for col in self.df.columns if re.match(r"^d\w*[0-9]", col)]
        else:
            colnames = [col for col in self.df.columns if re.match(r"^e\w*[0-9]", col)]
        return torch.tensor(self.df[colnames].values)
