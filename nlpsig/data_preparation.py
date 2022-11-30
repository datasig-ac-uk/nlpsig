import re
from typing import List, Optional, Union

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
        self.embeddings = embeddings
        self.embeddings_reduced = embeddings_reduced
        self.df = None
        self.df = self._get_modeling_dataframe()
        self._time_feature_choices = ["timeline_index"]
        self.time_features_added = False
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
            if "timeline_id" not in self.dataset_df.columns:
                # set default value to timeline_id
                df["timeline_id"] = 0
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
            The date as a fraction of the year.
        """
        # compute how many seconds the date is into the year
        x_year_start = pd.Timestamp(x.year, 1, 1)
        seconds_into_cal_year = abs(x - x_year_start).total_seconds()
        # compute the time fraction into the year
        time_frac = seconds_into_cal_year / (365 * 24 * 60 * 60)
        return x.year + time_frac

    def set_time_features(self) -> pd.DataFrame:
        """
        Updates the dataframe in .df to include time features:
        - `time_encoding`: the date as a fraction of the year
        - `time_diff`: the difference in time (in minutes) between successive records
        - `timeline_index`: the index of each post for each timeline

        Returns
        -------
        pd.DataFrame
            Updated dataframe with time features.
        """
        if self.time_features_added:
            print("Time features have already been added")
            return
        print("[INFO] Adding time feature columns into dataframe in .df")
        if "datetime" in self.df.columns:
            self._time_feature_choices += ["time_encoding", "time_diff"]
            # obtain time encoding by computing the fraction of year it is in
            self.df["time_encoding"] = self.df["datetime"].map(
                lambda t: self._time_fraction(t)
            )
            # sort by the timeline id and the date
            self.df = self.df.sort_values(by=["timeline_id", "datetime"]).reset_index(
                drop=True
            )
            # calculate time difference between posts
            self.df["time_diff"] = 0
            for i in range(1, len(self.df)):
                if self.df["timeline_id"].iloc[i] != self.df["timeline_id"].iloc[i - 1]:
                    diff = self.df["datetime"].iloc[i] - self.df["datetime"].iloc[i - 1]
                    diff_in_mins = diff.total_seconds() / 60
                    self.df["time_diff"][i] = diff_in_mins
        else:
            print(
                "[INFO] datetime is not a column in .df, "
                + "so only timeline_index is being added"
            )
        # assign index for each post in each timeline
        self.df["timeline_index"] = 0
        first_index = 0
        for t_id in set(self.df["timeline_id"]):
            # obtain the indices for this timeline-id
            t_id_len = sum(self.df["timeline_id"] == t_id)
            last_index = first_index + t_id_len
            # assign indices for each post in this timeline-id
            self.df["timeline_index"][first_index:last_index] = list(range(t_id_len))
            first_index = last_index
        self.time_features_added = True

        return self.df

    def _pad_timeline(
        self,
        time_n: int,
        colnames: List[str],
        id_counts: pd.Series,
        id: int,
        time_feature: List[str],
    ) -> pd.DataFrame:
        """
        [Private] For a given timeline-id, id, the function slices the dataframe in .df
        by finding those with timeline_id == id and keeping only the columns
        found in colnames.
        The function returns a dataframe with time_n rows:
        - If the number of records with timeline_id == id is less than time_n, it "pads" the
        dataframe by adding in empty records (with label = -1 to indicate they're padded)
        - If the number of records with timeline_id == id is equal to time_n, it just returns
        the records with timeline_id == id

        Parameters
        ----------
        time_n : int
            Maximum number of records in a particular timeline.
        colnames : List[str]
            List of column names that we wish to keep from the dataframe.
        id_counts : pd.Series
            The number of records in associated to each timeline_id.
        id : int
            Which timeline_id are we padding.
        time_feature : List[str]
            List of time feature column names that we wish to keep from the dataframe.

        Returns
        -------
        pd.DataFrame
            Padded dataframe for timeline_id id.

        Raises
        ------
        ValueError
            if time_n is less thatn id_counts[id].
        """
        padding_n = time_n - id_counts[id]
        if padding_n > 0:
            data_dict = {
                **{"timeline_id": [id], "label": [-1]},
                **dict.fromkeys(time_feature, [0]),
                **{c: [0] for c in colnames},
            }
            df_padded = pd.concat(
                [
                    self.df[self.df["timeline_id"] == id][data_dict.keys()],
                    pd.concat([pd.DataFrame(data_dict)] * padding_n),
                ]
            )
            return df_padded.reset_index(drop=True)
        elif padding_n == 0:
            columns = ["timeline_id", "label"] + time_feature + colnames
            return self.df[self.df["timeline_id"] == id][columns]
        else:
            raise ValueError("time_n should be larger than id_counts[id]")

    def _obtain_columns(self, embeddings: str) -> List[str]:
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

    def pad_timelines(
        self,
        time_feature: Optional[Union[List[str], str]] = None,
        embeddings: str = "full",
    ) -> np.array:
        """
        Creates an array which stores each of the timelines.
        We "pad" each timeline which has fewer number of posts
        (by adding on empty records) to make them all have the same number of posts.
        - A concatenated of padded dataframes are stored in `.df_padded`
        - The method returns an 3 dimensional array storing each timeline and the embeddings,
          and this is stored in `.array_padded`

        Parameters
        ----------
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
            3 dimension array:
            - First dimension is timelines
            - Second dimension is the posts
            - Third dimension are the features (e.g. embeddings /
              dimension reduced embeddings, time features)

        Raises
        ------
        ValueError
            if `time_feature` is a string as is not in the possible time_features
            (can be found in `._time_feature_choices` attribute)
        ValueError
            if `time_feature` is a list of strings and if any of the strings
            are not in the possible time_features (can be found in
            `._time_feature_choices` attribute)
        """
        print(
            "[INFO] Padding timelines and storing in .df_padded and .array_padded attributes"
        )
        # obtain timeline_id counts and largest number of posts in a timeline
        id_counts = self.df["timelind_id"].value_counts()
        time_n = id_counts.max()
        # obtain time feature colnames
        time_feature_colnames = self._obtain_time_feature_columns(
            time_feature=time_feature
        )
        # obtain colnames of embeddings
        colnames = self._obtain_columns(embeddings=embeddings)
        # pad each of the timeline-ids and store them in a list
        padded_dfs = [
            self._pad_timeline(
                time_n=time_n,
                colnames=colnames,
                id_counts=id_counts,
                id=id,
                time_feature=time_feature_colnames,
            )
            for id in id_counts.index
        ]
        self.df_padded = pd.concat(padded_dfs).reset_index(drop=True)
        self.array_padded = np.array(self.df_padded).reshape(
            len(id_counts), time_n, len(self.df_padded.columns)
        )
        return self.array_padded

    def get_torch_time_feature(
        self, time_feature: str = "time_encoding", standardise: bool = True
    ) -> torch.tensor:
        """
        Returns a `torch.tensor` object of the time_feature that is requested
        (the string passed has to be one of the strings in `._time_feature_choices`).

        Parameters
        ----------
        time_feature : str, optional
            Which time feature to obtain `torch.tensor` for, by default "time_encoding"
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
            if `self.array_padded` is `None`. In this case,
            need to call `.pad_timelines()` method first.
        """
        if self.array_padded is None:
            raise ValueError("Need to first call .pad_timelines()")
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
