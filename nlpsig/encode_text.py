import pickle
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer


class SentenceEncoder:
    """
    Class to obtain sentence embeddings using SentenceTransformer class in sentence_transformers.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pre_computed_embeddings_file: Optional[str] = None,
        col_name_text: str = "content",
        model_name: str = "all-MiniLM-L6-v2",
        model_modules: Optional[Iterable[nn.Module]] = None,
        model_encoder_args: dict = {
            "batch_size": 64,
            "show_progress_bar": True,
            "output_value": "sentence_embedding",
            "convert_to_numpy": True,
            "convert_to_tensor": False,
            "device": None,
            "normalize_embeddings": False,
        },
        model_fit_args: dict = {},
    ):
        """
        Class to obtain sentence embeddings using SentenceTransformer class in sentence_transformers.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset as a pandas dataframe
        pre_computed_embeddings_file : Optional[str], optional
            Path to pre-computed embeddings, by default None.
        col_name_text : str, optional
            Column name which has the text in, by default "content".
        model_name : str, optional
            Name of model to obtain sentence embeddings, by default "all-MiniLM-L6-v2".
            If loading a pretrained model using `.load_pretrained_model()` method,
            passes this to the `model_name_or_path` argument when initialising `SentenceTransformer` object
            A few alternative options are:
            - all-mpnet-base-v2
            - all-distilroberta-v1
            - all-MiniLM-L12-v2
            See more pre-trained SentenceTransformer models at https://www.sbert.net/docs/pretrained_models.html.
        model_modules : Optional[Iterable[nn.Module]], optional
            This parameter can be used to create custom SentenceTransformer models from scratch.
            See https://www.sbert.net/docs/training/overview.html#creating-networks-from-scratch for examples.
            If creating a custom model using `.load_custom_model()` method,
            passes this into the `modules` argument when initialising `SentenceTransformer` object
        model_encoder_args : dict, optional
            Any keywords to be passed in to the model, by default the
            following arguments to pass into the `.encode()` method of SentenceTransformer class:
            {"batch_size": 64,
             "show_progress_bar": True,
             "output_value": "sentence_embedding",
             "convert_to_numpy": True,
             "convert_to_tensor": False,
             "device": None,
             "normalize_embeddings": False}

        Raises
        ------
        KeyError
            if `col_name_text` is not a column in df
        """
        self.df = df
        if col_name_text not in df.columns:
            raise KeyError(f"{col_name_text} is not a column in df")
        else:
            self.col_name_text = col_name_text
        if pre_computed_embeddings_file is not None:
            with open(pre_computed_embeddings_file, "rb") as f:
                self.sentence_embeddings = np.array(pickle.load(f))
                if (self.sentence_embeddings.ndim != 2) or (
                    self.sentence_embeddings.shape[0] != len(self.df)
                ):
                    raise ValueError(
                        f"the loaded embeddings from {pre_computed_embeddings_file} "
                        "must be a (n x d) array where n is the number of sentences "
                        "and d is the dimension of the embeddings"
                    )
            self.model_name = "pre-computed"
            self.model_modules = None
            self.model_encoder_args = None
            self.model_fit_args = None
            self.model = "pre-computed"
        else:
            self.sentence_embeddings = None
            self.model_name = model_name
            self.model_modules = model_modules
            self.model_encoder_args = model_encoder_args
            self.model_fit_args = model_fit_args
            self.model = None

    def load_pretrained_model(self, force_reload: bool = False) -> None:
        """
        Loads pre-trained model into `.model` by passing in `.model_name` to
        the `model_name_or_path` argument when initialising `SentenceTransformer` object

        `.model_name` can also be path to a trained model

        Parameters
        ----------
        force_reload : bool, optional
            Whether or not to overwrite current loaded model, by default False

        Raises
        ------
        NotImplementedError
            if `.model_name` cannot be loaded by SentenceTransformer.
            This might happen if this is not a pre-trained model available.
            See https://www.sbert.net/docs/pretrained_models.html for examples.
        """
        if (not force_reload) and (self.model is not None):
            print(f"[INFO] {self.model_name} model is already loaded")
            return
        if (force_reload) and (self.model == "pre-computed"):
            print(
                "[INFO] the current embeddings were computed before "
                + "and were loaded into this class"
            )
            return
        try:
            self.model = SentenceTransformer(model_name_or_path=self.model_name)
        except:
            raise NotImplementedError(
                f"Loading model '{self.model_name}' with SentenceTransformer failed. "
                "See SentenceTransformer documentation in sentence_transformers."
            )

    def load_custom_model(self, force_reload: bool = False) -> None:
        """
        Loads pre-trained model into `.model` by passing in `.model_name` to
        the `modules` argument when initialising `SentenceTransformer` object

        Parameters
        ----------
        force_reload : bool, optional
            Whether or not to overwrite current loaded model, by default False

        Raises
        ------
        ValueError
            if there is nothing stored in `.model_modules` attribute to initialise
            SentenceTransformer model
        NotImplementedError
            if loading in a model using the modules in `.model_modules` was unsuccessful.
            This might happen if any of the items in `.model_modules` were not valid modules.
            See https://www.sbert.net/docs/training/overview.html#creating-networks-from-scratch for examples.
        """
        if (not force_reload) and (self.model is not None):
            print(f"[INFO] {self.model_name} model is already loaded")
            return
        if (force_reload) and (self.model == "pre-computed"):
            print(
                "[INFO] the current embeddings were computed before "
                + "and were loaded into this class"
            )
            return
        if self.model_modules is None:
            raise ValueError(
                "`.model_modules` must be a list of modules which define the network architecture."
            )
        try:
            self.model = SentenceTransformer(modules=self.model_modules)
        except:
            raise NotImplementedError(
                f"Loading model with modules: {self.model_modules}, with SentenceTransformer failed. "
                "See SentenceTransformer documentation in sentence_transformers "
                "and https://www.sbert.net/docs/training/overview.html#creating-networks-from-scratch "
                "for information on how to create the networks architectures from scratch by defining "
                "the individual layers"
            )

    def encode_sentence_transformer(self) -> None:
        """
        Obtains sentence embeddings (i.e. encodes sentences) via the `.encode` method,
        and saves in `.embeddings_sentence` attribute.

        Passes in `.model_encoder_args` into `.encode` method too.

        Raises
        ------
        NotImplementedError
            if `.model` attribute is None in which case need to load the model first using either
            `.load_pretrained_model()` or `.load_custom_model()` methods.
        """
        if self.model is None:
            raise NotImplementedError(
                "Model is not loaded. Call either `.load_pretrained_model()` "
                "or `.load_custom_model()` methods first"
            )
        sentences = self.df[self.col_name_text].to_list()
        print(f"[INFO] number of sentences to encode: {len(sentences)}")
        self.sentence_embeddings = self.model.encode(
            sentences, **self.model_encoder_args
        )

    def fit_sentence_transformer(
        self, train_objectives: Iterable[Tuple[DataLoader, nn.Module]]
    ) -> None:
        # TODO
        """
        Trains / fine-tunes SentenceTransformer model via the `.fit` method.

        Passes in `.model_fit_args` into `.fit` method too.

        Parameters
        ----------
        train_objectives : Iterable[Tuple[DataLoader, nn.Module]]
            Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning.
            See https://www.sbert.net/docs/training/overview.html for more details.

        Raises
        ------
        NotImplementedError
            if `.model` attribute is None in which case need to load the model first using either
            `.load_pretrained_model()` or `.load_custom_model()` methods.
        """
        if self.model is None:
            raise NotImplementedError(
                "Model is not loaded. Call either `.load_pretrained_model()` "
                "or `.load_custom_model()` methods first"
            )
        self.sentence_embeddings = self.model.fit(
            train_objectives, **self.model_fit_args
        )


class TextEncoder:
    """
    Class to obtain embeddings using Huggingface transformers.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        col_name_text: str = "content",
        model_name: str = "bert-base-uncased",
        skip_special_tokens: bool = True,
    ):
        self.df = df
        if col_name_text not in df.columns:
            raise KeyError(f"{col_name_text} is not a column in df")
        else:
            self.col_name_text = col_name_text
        self.sentence_embeddings = None
        self.model_name = model_name
        self.model = None
        self.skip_special_tokens = skip_special_tokens

    def load_pretrained_model(self):
        """loads in config, tokenizer and pretrained weights"""
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    def initialise_transformer(self, **config_args):
        """loads in config and tokenizer. initialises the transformer with random weights"""
        self.config = AutoConfig.from_pretrained(self.model_name, **config_args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_config(self.config)
        self.model.eval()

    def tokenize_text(self, text_id_col_name="text_id", **tokenizer_args):
        """tokenize each item in df[col_name_text]"""
        if not tokenizer_args:
            tokenizer_args = {"padding": True, "truncation": True}
        if tokenizer_args.get("return_tensors") != "pt":
            print("setting return_tensors='pt'")
            tokenizer_args["return_tensors"] = "pt"
        if not tokenizer_args.get("return_special_tokens_mask"):
            print("setting return_special_tokens_mask=True")
            tokenizer_args["return_special_tokens_mask"] = True
        self.tokens = self.tokenizer(
            self.df[self.col_name_text].to_list(), **tokenizer_args
        )
        self.special_tokens_mask = self.tokens.data.pop("special_tokens_mask")
        self.df["tokenized_text"] = [
            self.tokenizer.convert_ids_to_tokens(
                self.tokens["input_ids"][i],
                skip_special_tokens=self.skip_special_tokens,
            )
            for i in range(len(self.df))
        ]
        if self.skip_special_tokens:
            self.df["tokenized_text_with_special_tokens"] = [
                self.tokenizer.convert_ids_to_tokens(
                    self.tokens["input_ids"][i], skip_special_tokens=False
                )
                for i in range(len(self.df))
            ]
        # create new tokenized dataframe
        print(
            "[INFO] Creating tokenized dataframe and setting in .tokenized_df attribute"
        )
        self.tokenized_df = self.df.drop(
            columns=[self.col_name_text, "tokenized_text_with_special_tokens"],
            errors="ignore",
        ).explode("tokenized_text")
        self.tokenized_df = self.tokenized_df.reset_index()
        if text_id_col_name not in self.tokenized_df.columns:
            print(
                f"'{text_id_col_name}' is the column name for denoting the corresponding text id"
            )
            self.tokenized_df = self.tokenized_df.rename(
                columns={"index": text_id_col_name}
            )
            self.text_id_col_name = text_id_col_name
        else:
            raise ValueError(
                f"'{text_id_col_name}' is already a column name in the dataframe. "
                "please specify a different column name to denote the corresponding text ids"
            )
        return self.tokens

    def obtain_embeddings(self, method="hidden_layer", **kwargs):
        """obtain token embeddings for each item in df[col_name_text]
        has implemented various pooling strategies to get embeddings
        but can also just output the hidden states from the transformer
        options:
        - a particular layer hidden state (need to pass in the layer you want - default the second to last one)
        - concatenation of particular layer hidden states (need to pass in layers to concatenate - default last 4)
        - element-wise sum of particular layer hidden states (need to pass in layers to sum - default last 4)
        - mean of particular layer hidden states (need to pass in layers to take average over - default last 4)
        - output all or some hidden states - user can do their own custom pooling to obtain individual embeddings for each token
        """
        if self.skip_special_tokens:
            # finds indices of non-special tokens
            indices = [
                torch.where(self.special_tokens_mask[i] == 0)[0]
                for i in range(len(self.df))
            ]
        else:
            # finds indices of non-pad tokens
            indices = [
                torch.where(self.tokens["attention_mask"][i] == 1)[0]
                for i in range(len(self.df))
            ]
        with torch.no_grad():
            outputs = self.model(**self.tokens, output_hidden_states=True)
            hidden_states = torch.stack(outputs["hidden_states"], dim=0)
        if not kwargs:
            kwargs = {}
        # by default, we will concatenate the embeddings at the end unless we've requested to
        # obtain hidden states from multiple layers
        concatenate_embeddings = True
        if method == "hidden_layer":
            if not kwargs.get("layers"):
                # if not specified layers wanted, returns the second to last one
                kwargs["layers"] = hidden_states.shape[0] - 1
            # only consider layers requested
            if isinstance(kwargs["layers"], list):
                if any(
                    [
                        (layer < 0) or (layer >= hidden_states.shape[0])
                        for layer in kwargs["layers"]
                    ]
                ):
                    raise ValueError(
                        f"requested layers ({kwargs['layers']}) is out of range: only have "
                        f"{hidden_states.shape[0]} number of hidden layers"
                    )
                # hidden_states[kwargs["layers"]] is tensor with (layers, batch, tokens, embeddings)
                # change order of tensor to (batch, layers, tokens, embeddings)
                hidden_states = hidden_states[kwargs["layers"]].permute(1, 0, 2, 3)
                # obtain hidden layers for each item in the dataframe
                self.token_embeddings = [
                    hidden_states[i][:, indices[i], :].numpy()
                    for i in range(len(self.df))
                ]
                concatenate_embeddings = False
            elif isinstance(kwargs["layers"], int):
                if kwargs["layers"] >= hidden_states.shape[0]:
                    raise ValueError(
                        f"requested layer ({kwargs['layers']}) is out of range: only have "
                        f"{hidden_states.shape[0]} number of hidden layers"
                    )
                elif kwargs["layers"] < 0:
                    raise ValueError(
                        f"requested layer ({kwargs['layers']}) is out of range: "
                        "must be greater than or equal to 0, and we only have "
                        f"{hidden_states.shape[0]} number of hidden layers"
                    )
                # only want one layer so hidden_states[kwargs["layers"]] is tensor with (batch, tokens, embeddings)
                hidden_states = hidden_states[kwargs["layers"]]
                # obtain hidden layer for each item in the dataframe
                self.token_embeddings = [
                    hidden_states[i][indices[i], :] for i in range(len(self.df))
                ]
            else:
                raise ValueError(
                    "layers requested must be either integer or list of integers"
                )
        elif method in ["concatenate", "sum", "mean"]:
            if not kwargs.get("layers"):
                # if not specified layers wanted, concatenates last 4 hidden layers (if I can go that far)
                kwargs["layers"] = [
                    hidden_states.shape[0] - i
                    for i in range(1, 5)
                    if hidden_states.shape[0] - i >= 0
                ]
            if any(
                [
                    (layer < 0) or (layer >= hidden_states.shape[0])
                    for layer in kwargs["layers"]
                ]
            ):
                raise ValueError(
                    f"requested layers ({kwargs['layers']}) is out of range: only have "
                    f"{hidden_states.shape[0]} number of hidden layers"
                )
            # hidden_states[kwargs["layers"]] is tensor with (layers, batch, tokens, embeddings)
            # change order of tensor to (batch, tokens, layers, embeddings)
            hidden_states = hidden_states[kwargs["layers"]].permute(1, 2, 0, 3)
            if method == "concatenate":
                self.token_embeddings = [
                    torch.stack(
                        [
                            hidden_states[i][indices[i]][j].flatten()
                            for j in range(len(indices[i]))
                        ]
                    )
                    for i in range(len(self.df))
                ]
            elif method == "sum":
                self.token_embeddings = [
                    torch.stack(
                        [
                            hidden_states[i][indices[i]][j].sum(dim=0)
                            for j in range(len(indices[i]))
                        ]
                    )
                    for i in range(len(self.df))
                ]
            elif method == "mean":
                self.token_embeddings = [
                    torch.stack(
                        [
                            hidden_states[i][indices[i]][j].mean(dim=0)
                            for j in range(len(indices[i]))
                        ]
                    )
                    for i in range(len(self.df))
                ]
        else:
            raise NotImplementedError(
                f"method '{method}' for pooling hidden layers to obtain token embeddings has not been implemented"
            )
        if concatenate_embeddings:
            self.token_embeddings = torch.cat(self.token_embeddings).numpy()
        return self.token_embeddings

    def pool_token_embeddings(self, method="mean"):
        """once have token embeddings stored in self.token_embeddings (so each item in this column is a 2 dimensional torch tensor
        with first dimension as the tokens, second as the embeddings, we can optionally pool them). but we should mention that
        this is not as good as using sentence transformers - would pool them better
        also if we had skip_special_tokens=True, we would've gotten rid of the special tokens
        wouldn't be able to use cls - should return error if wanted to get cls one and we had skip_special_tokens=True"""
        if isinstance(self.token_embeddings, list) or self.token_embeddings.ndim != 2:
            raise ValueError(
                "the token embeddings must be a numpy array of dimension 2 with dimensions (token x embedding)"
                "currently the token embeddings are in a list. if each item in the list is a tensor or array of dimension two, simply concatenate them"
                "if each item is a three dimensional numpy array (i.e. they are hidden layers of the transformer), need to pool the layers first before concatentating to get the token embeddings"
            )
        if method == "mean":
            self.pooled_embeddings = [
                self.token_embeddings[
                    self.tokenized_df.index[
                        self.tokenized_df[self.text_id_col_name] == i
                    ]
                ].mean(axis=0)
                for i in range(len(self.df))
            ]
        elif method == "max":
            self.pooled_embeddings = [
                self.token_embeddings[
                    self.tokenized_df.index[
                        self.tokenized_df[self.text_id_col_name] == i
                    ]
                ].max(axis=0)
                for i in range(len(self.df))
            ]
        elif method == "sum":
            self.pooled_embeddings = [
                self.token_embeddings[
                    self.tokenized_df.index[
                        self.tokenized_df[self.text_id_col_name] == i
                    ]
                ].sum(axis=0)
                for i in range(len(self.df))
            ]
        elif method == "cls":
            if self.skip_special_tokens:
                raise ValueError(
                    "we skipped special tokens and so cls embedding has not been saved. "
                    "will need to obtain token embeddings again but set skip_special_tokens=False"
                )
            else:
                self.pooled_embeddings = [
                    self.token_embeddings[
                        self.tokenized_df.index[
                            self.tokenized_df[self.text_id_col_name] == i
                        ]
                    ][0]
                    for i in range(len(self.df))
                ]
        else:
            raise NotImplementedError(
                f"method '{method}' for pooling the token embeddings not been implemented"
            )
        self.pooled_embeddings = np.stack(self.pooled_embeddings)
        return self.pooled_embeddings

    def fit_transformer(self):
        """fit / fine-tune transformer model to some task"""
        # TODO
        pass
