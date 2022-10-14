import pandas as pd

def get_modeling_dataframe(annotations, embeddings_sentence, embeddings_reduced=[]):
    embedding_sentence_df = pd.DataFrame(embeddings_sentence, columns = ['e' + str(i+1) for i in range(embeddings_sentence.shape[1])])

    if (len(embeddings_reduced)!=0):
        embeddings_reduced_df = pd.DataFrame(embeddings_reduced, columns = ['d' + str(i+1) for i in range(embeddings_reduced.shape[1])])
        df = pd.concat([annotations.reset_index(drop=True), embeddings_reduced_df, embedding_sentence_df], axis=1)
    else:
        df = pd.concat([annotations.reset_index(drop=True), embedding_sentence_df], axis=1)

    return df