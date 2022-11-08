import numpy as np
import pandas as pd

def get_modeling_dataframe(dataset_df: pd.DataFrame,
                           embeddings_sentence: np.array,
                           embeddings_reduced: np.array = np.array([])) -> pd.DataFrame:
    """
    Combine original dataset_df with embeddings and the embeddings after dimension reduction
    """
    if dataset_df.shape[0] != embeddings_sentence.shape[0]:
        raise ValueError("dataset_df, embeddings_sentence and embeddings_reduced " +
                         "should have the same number of rows")
    elif dataset_df.shape[0] != embeddings_reduced.shape[0]:
        raise ValueError("dataset_df, embeddings_sentence and embeddings_reduced " +
                         "should have the same number of rows")
    embedding_sentence_df = pd.DataFrame(embeddings_sentence,
                                         columns = [
                                             'e'+str(i+1)
                                             for i in range(embeddings_sentence.shape[1])
                                             ]
                                         )
    if (len(embeddings_reduced)!=0):
        embeddings_reduced_df = pd.DataFrame(embeddings_reduced,
                                             columns = [
                                                 'd'+str(i+1)
                                                 for i in range(embeddings_reduced.shape[1])
                                                 ]
                                             )
        df = pd.concat([dataset_df.reset_index(drop=True),
                        embeddings_reduced_df,
                        embedding_sentence_df],
                       axis = 1)
    else:
        df = pd.concat([dataset_df.reset_index(drop=True),
                        embedding_sentence_df],
                       axis = 1)

    return df