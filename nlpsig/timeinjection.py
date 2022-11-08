import datetime
import numpy as np
import pandas as pd
import re

class TimeFeatures:
    def __init__(self):
        #How many hours in a year?
        self.total_year_hours = 365*24

    def time_fraction(self, x: pd.Timestamp) -> float:
        return x.year + abs(x - datetime.datetime(x.year, 1,1,0)).total_seconds() / 3600.0 / self.total_year_hours

    def get_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        #obtain time encoding
        df['time_encoding'] = df['datetime'].map(lambda t: self.time_fraction(t))

        df = df.sort_values(by=['timeline_id', 'datetime']).reset_index(drop=True)

        #calculate time difference between posts
        df['time_diff'] = 0
        for i in range(df.shape[0]):
            if (i > 0):
                if (df['timeline_id'][i] == df['timeline_id'][i-1]):
                    df['time_diff'][i] = (df['datetime'][i] - df['datetime'][i-1] ).total_seconds() / 60

        #assign index for post in the timeline       
        df['timeline_index'] = 0
        timelineid_list = df['timeline_id'].unique().tolist()
        first_index = 0
        for t_id in timelineid_list:
            t_id_len = len(df[df['timeline_id']==t_id])
            last_index = first_index + t_id_len
            df['timeline_index'][first_index:last_index] = np.arange(t_id_len)
            first_index = last_index
        
        return df


class Padding:
    def __init__(self):
        self.time_n = 40
    
    def _pad_timeline(self,
                      df: pd.DataFrame,
                      id_counts: pd.Series,
                      index: int,
                      i) -> pd.DataFrame:
        padding_n = self.time_n - id_counts[index]

        data_dict = {'timeline_id': [index], 'label': [-1], 'time_encoding': [0]} 

        for c in df.columns:
            if (re.match("^d\w*[0-9]", c)):
                data_dict.update({c:[0]})

        if (padding_n > 0) :
            pad = pd.DataFrame(data=data_dict)
            pad = pd.concat([pad] * padding_n, axis=0, ignore_index=True)
            df_pad = pd.concat([df[:(i+id_counts[index])], pad])
            df_pad = pd.concat([df_pad, df[(i+id_counts[index]):]]).reset_index(drop=True)
        else:
            df_pad = pd.concat([df[:i], df[i:(i+self.time_n)], df[(i+id_counts[index]):]]).reset_index(drop=True)

        return df_pad

    def pad_timelines(self, df: pd.DataFrame) -> np.array:

        #dataset specifics 
        id_counts = df.groupby(['timeline_id'])['timeline_id'].count()#.set_index(['timeline_id'])
        self.time_n = id_counts.max()

        #iterate to create slices
        i = 0
        for c in id_counts.index:
            place_holder_df = df[['timeline_id', 'label', 'time_encoding'] +
                                 [c for c in df.columns if re.match("^d\w*[0-9]", c)]]
            df = self._pad_timeline(df = place_holder_df,
                                    id_counts = id_counts,
                                    index = c,
                                    i = i)
            i += self.time_n
        
        #reshape data
        return np.array(df).reshape(id_counts.shape[0], self.time_n, df.shape[1])
