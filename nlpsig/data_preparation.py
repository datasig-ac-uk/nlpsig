import numpy as np
import pandas as pd
import re
from typing import List, Optional

class PrepareData:
    def __init__(self,
                 dataset_df: pd.DataFrame,
                 embeddings_sentence: np.array,
                 embeddings_reduced: Optional[np.array] = None):
        if dataset_df.shape[0] != embeddings_sentence.shape[0]:
            raise ValueError("dataset_df, embeddings_sentence and embeddings_reduced " +
                             "should have the same number of rows")
        if embeddings_reduced is not None:    
            if dataset_df.shape[0] != embeddings_reduced.shape[0]:
                raise ValueError("dataset_df, embeddings_sentence and embeddings_reduced " +
                                "should have the same number of rows")
        self.dataset_df = dataset_df
        self.embeddings_sentence = embeddings_sentence
        self.embeddings_reduced = embeddings_reduced
        self.df = None
        self.df = self.get_modeling_dataframe()
        self.df_padded = None
        self.array_padded = None
    
    def get_modeling_dataframe(self) -> pd.DataFrame:
        """
        Combine original dataset_df with embeddings and the embeddings after dimension reduction
        """
        if self.df is not None:
            return self.df
        else:
            embedding_sentence_df = pd.DataFrame(self.embeddings_sentence,
                                                 columns = [
                                                     f"e{i+1}"
                                                     for i in range(self.embeddings_sentence.shape[1])
                                                     ]
                                                 )
            if self.embeddings_reduced is not None:
                embeddings_reduced_df = pd.DataFrame(self.embeddings_reduced,
                                                     columns = [
                                                         f"d{i+1}"
                                                         for i in range(self.embeddings_reduced.shape[1])
                                                         ]
                                                     )
                df = pd.concat([self.dataset_df.reset_index(drop=True),
                                embeddings_reduced_df,
                                embedding_sentence_df],
                            axis = 1)
            else:
                df = pd.concat([self.dataset_df.reset_index(drop=True),
                                embedding_sentence_df],
                            axis = 1)
            return df
    
    @staticmethod
    def _time_fraction(x: pd.Timestamp) -> float:
        """
        Computes
        """
        # compute how many seconds the date is into the year
        x_year_start = pd.Timestamp(x.year, 1, 1)
        seconds_into_cal_year = abs(x - x_year_start).total_seconds()
        # compute the time fraction into the year
        time_frac = seconds_into_cal_year / (365*24*60*60)
        return x.year + time_frac

    def set_time_features(self) -> pd.DataFrame:
        """
        Updates the dataframe in .df to include time features:
        - time_encoding: the date as a fraction of the year
        - time_diff: the difference in time (in minutes) between successive records
        - timeline_index: the index of each post for each timeline
        """
        # obtain time encoding by computing the fraction of year it is in
        self.df['time_encoding'] = self.df['datetime'].map(lambda t: self._time_fraction(t))
        # sort by the timeline id and the date
        self.df = self.df.sort_values(by = ['timeline_id', 'datetime']).reset_index(drop=True)
        # calculate time difference between posts
        self.df['time_diff'] = 0
        for i in range(1, len(self.df)):
            if (self.df['timeline_id'].iloc[i] != self.df['timeline_id'].iloc[i-1]):
                diff = self.df['datetime'].iloc[i]-self.df['datetime'].iloc[i-1]
                diff_in_mins = diff.total_seconds() / 60
                self.df['time_diff'][i] = diff_in_mins
        # assign index for each post in each timeline
        self.df['timeline_index'] = 0
        first_index = 0
        for t_id in set(self.df['timeline_id']):
            # obtain the indices for this timeline-id
            t_id_len = sum(self.df['timeline_id'] == t_id)
            last_index = first_index + t_id_len
            # assign indices for each post in this timeline-id
            self.df['timeline_index'][first_index:last_index] = list(range(t_id_len))
            first_index = last_index
        
    def _pad_timeline(self,
                      time_n,
                      colnames: List[str],
                      id_counts: pd.Series,
                      id: int) -> pd.DataFrame:
        """
        For a given timeline-id, id, the function slices the dataframe in .df
        by finding those with timeline_id == id and keeping only the columns
        found in colnames. 
        The function returns a dataframe with time_n rows. If the number of
        records with timeline_id == id is less than time_n, it "pads" the
        dataframe by adding in empty records (with label = -1 to indicate they're padded)
        """
        padding_n = time_n - id_counts[id]
        if (padding_n > 0):
            data_dict = {**{'timeline_id': [id], 'label': [-1], 'time_encoding': [0]},
                         **{c:[0] for c in colnames}}
            df_padded = pd.concat([self.df[self.df["timeline_id"]==id][data_dict.keys()],
                                   pd.concat([pd.DataFrame(data_dict)]*padding_n)])
            return df_padded.reset_index(drop=True)
        elif padding_n == 0:
            colnames = ['timeline_id', 'label', 'time_encoding'] + colnames
            return self.df[self.df["timeline_id"]==id][colnames]
        else:
            raise ValueError("time_n should be larger than id_counts[id]")

    def pad_timelines(self,
                      keep_embedding_sentences: bool = False) -> np.array:
        """
        Creates an array which stores each of the timelines.
        Wwe "pad" each timeline which has fewer number of posts
        (by adding on empty records) to make them all have the same number of posts.
        
        A concatenated of padded dataframes are stored in .df_padded
        
        The method returns an 3 dimensional array storing each timeline and the embeddings,
        and this is stored in .array_padded
        """
        # obtain timeline_id counts and largest number of posts in a timeline
        id_counts = self.df.groupby(['timeline_id'])['timeline_id'].count()
        time_n = id_counts.max()
        # obtain columns for the dimension reduced sentence embeddings
        # these are columns which start with 'd' and have a number following it
        colnames = [col for col in self.df.columns if re.match("^d\w*[0-9]", col)]
        if keep_embedding_sentences:
            # add columns for the sentence embeddings
            # these are columns which start with 'e' and have a number following it
            colnames += [col for col in self.df.columns if re.match("^e\w*[0-9]", col)]
        # pad each of the timeline-ids and store them in a list
        padded_dfs =  [self._pad_timeline(time_n = time_n,
                                          colnames = colnames,
                                          id_counts = id_counts,
                                          id = id)
                       for id in id_counts.index]
        self.df_padded = pd.concat(padded_dfs).reset_index(drop=True)
        # reshape data and drop the
        self.array_padded = np.array(self.df_padded).reshape(len(id_counts),
                                                             time_n,
                                                             len(self.df_padded.columns))
        return self.array_padded
