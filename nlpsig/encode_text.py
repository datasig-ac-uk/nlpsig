import pickle
from typing import Optional

import pandas as pd


class TextEncoder:
    """
    Class to obtain sentence embeddings
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pre_computed_embeddings_file: Optional[str] = None,
        col_name_text: str = "content",
        model_name: str = "all-MiniLM-L6-v2",
        model_args: dict = {
            "batch_size": 64,
            "show_progress_bar": True,
            "output_value": "sentence_embedding",
            "convert_to_numpy": True,
            "convert_to_tensor": False,
            "device": None,
            "normalize_embeddings": False,
        },
    ):
        """
        Class to obtain sentence embeddings

        Parameters
        ----------
        df : pd.DataFrame
            Dataset as a pandas dataframe
        pre_computed_embeddings_file : Optional[str], optional
            Path to pre-computed embeddings, by default None
        col_name_text : str, optional
            Column name which has the text in, by default "content"
        model_name : str, optional
            Name of model to obtain sentence embeddings, by default "all-MiniLM-L6-v2"
        model_args : _type_, optional
            Any keywords to be passed in to the model, by default have arguments to
            implement "all-MiniLM-L6-v2" model:
            {"batch_size": 64,
             "show_progress_bar": True,
             "output_value": "sentence_embedding",
             "convert_to_numpy": True,
             "convert_to_tensor": False,
             "device": None,
             "normalize_embeddings": False}
        """
        self.df = df
        self.col_name_text = col_name_text
        if pre_computed_embeddings_file is not None:
            with open(pre_computed_embeddings_file, "rb") as f:
                self.embeddings_sentence = pickle.load(f)
            self.model_name = "pre-computed"
            self.model_args = None
            self.model = "pre-computed"
        else:
            self.embeddings_sentence = None
            self.model_name = model_name
            self.model_args = model_args
            self.model = None
        self.model_dict = {"all-MiniLM-L6-v2": "sentence_embedding"}

    def encode_sentence_transformer(self) -> None:
        """
        Obtains sentence embeddings and saves in `.embeddings_sentence`
        """
        self.load_model()
        sentences = self.df[self.col_name_text].to_list()
        print(f"[INFO] number of sentences to encode: {len(sentences)}")
        self.embeddings_sentence = self.model.encode(sentences, **self.model_args)

    def load_model(self, force_reload=False) -> None:
        """
        Loads model into `.model`

        Parameters
        ----------
        force_reload : bool, optional
            Whether or not to overwrite current loaded model, by default False

        Raises
        ------
        NotImplementedError
            if `.model_name` is not in `.model_dict`
        """
        if (not force_reload) and (self.model is not None):
            print(f"[INFO] {self.model_name} model is already loaded.")
            return
        if (force_reload) and (self.model == "pre-computed"):
            print(
                "[INFO] the current embeddings were computed before "
                + "and were loaded into this class."
                + "First reset the 'model_name' and 'model_args' attributes "
                + "if you want to re-load different embeddings"
            )
        detected_model_library = self.detect_model_library()
        if detected_model_library is None:
            raise NotImplementedError(
                f"{self.model_name} is not implemented. "
                f"Try one of the following: " + f"{', '.join(self.model_dict.keys())}."
            )
        elif detected_model_library == "sentence_embedding":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)

    def detect_model_library(self) -> Optional[str]:
        """
        Checks if `.model_name` is a valid model in our library

        Returns
        -------
        Optional[str]
            Model as string if the model is in `.model_dict`
        """
        if self.model_name in self.model_dict.keys():
            return self.model_dict[self.model_name]
        else:
            return None
