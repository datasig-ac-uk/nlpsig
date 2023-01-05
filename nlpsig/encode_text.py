import pickle
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, BatchEncoding


class SentenceEncoder:
    """
    Class to obtain sentence embeddings using SentenceTransformer class in `sentence_transformers`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        col_name_text: str,
        pre_computed_embeddings_file: Optional[str] = None,
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
        Class to obtain sentence embeddings using SentenceTransformer class
        in `sentence_transformers`.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset as a pandas dataframe
        col_name_text : str
            Column name which has the text in
        pre_computed_embeddings_file : Optional[str], optional
            Path to pre-computed embeddings, by default None.
        model_name : str, optional
            Name of model to obtain sentence embeddings, by default "all-MiniLM-L6-v2".
            If loading a pretrained model using `.load_pretrained_model()` method,
            passes this to the `model_name_or_path` argument when initialising
            `SentenceTransformer` object
            A few alternative options are:
            - all-mpnet-base-v2
            - all-distilroberta-v1
            - all-MiniLM-L12-v2
            See more pre-trained SentenceTransformer models at
            https://www.sbert.net/docs/pretrained_models.html.
        model_modules : Optional[Iterable[nn.Module]], optional
            This parameter can be used to create custom SentenceTransformer models from scratch.
            See https://www.sbert.net/docs/training/overview.html#creating-networks-from-scratch
            for examples.
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
            See https://www.sbert.net/docs/training/overview.html#creating-networks-from-scratch
            for examples.
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
                f"Loading model with modules: {self.model_modules}, with SentenceTransformer "
                "failed. See SentenceTransformer documentation in sentence_transformers "
                "(https://www.sbert.net/docs/training/"
                "overview.html#creating-networks-from-scratch) "
                "for information on how to create the networks architectures from scratch."
            )

    def obtain_embeddings(self) -> np.array:
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
        self.sentence_embeddings = np.array(
            self.model.encode(sentences, **self.model_encoder_args)
        )
        return self.sentence_embeddings

    def fit_transformer(
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
        self.model.fit(train_objectives, **self.model_fit_args)


class TextEncoder:
    """
    Class to obtain token embeddings (and optionally pool them) using Huggingface transformers.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        col_name_text: str,
        model_name: str = "bert-base-uncased",
    ):
        """
        Class to obtain token embeddings (and optionally pool them) using Huggingface transformers.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset as a pandas dataframe
        col_name_text : str
            Column name which has the text in
        model_name : str, optional
            Name of transformer encoder model, by default "bert-base-uncased"

        Raises
        ------
        KeyError
            if `col_name_text` is not a column in df
        """
        self.df = df
        self.tokenized_df = None
        if col_name_text not in df.columns:
            raise KeyError(f"{col_name_text} is not a column in df")
        self.col_name_text = col_name_text
        self.token_embeddings = None
        self.pooled_embeddings = None
        self.model_name = model_name
        self.model = None
        self.config = None
        self.tokenizer = None
        self.tokens = None
        self.skip_special_tokens = None
        self.special_tokens_mask = None
        self.text_id_col_name = None

    def load_pretrained_model(self, force_reload: bool = False) -> None:
        """
        loads in config, tokenizer and pretrained weights

        Parameters
        ----------
        force_reload : bool, optional
            Whether or not to overwrite current loaded model, by default False
        """
        if (not force_reload) and (self.model is not None):
            print(f"[INFO] {self.model_name} model is already loaded")
            return
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    def initialise_transformer(self, force_reload: bool = False, **config_args) -> None:
        """
        loads in config and tokenizer. initialises the transformer with random weights

        Parameters
        ----------
        force_reload : bool, optional
            Whether or not to overwrite current loaded model, by default False
        """
        if (not force_reload) and (self.model is not None):
            print(f"[INFO] {self.model_name} model is already loaded")
            return
        self.config = AutoConfig.from_pretrained(self.model_name, **config_args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_config(self.config)
        self.model.eval()

    def tokenize_text(
        self,
        text_id_col_name="text_id",
        skip_special_tokens: bool = True,
        **tokenizer_args,
    ) -> BatchEncoding:
        """
        Method to tokenize each item in the `col_name_text` column of the dataframe.

        Will tokenize the text (which are then saved in `.tokens` attribute).
        The method will also create a new dataframe (and save it in `.tokenized_df` attribute)
        where each item in the `tokens` column is a token and `text_id_col_name`
        column denotes the text-id for which it belongs to (where text-id
        is just the index of the original dataframe stored in `.df`).

        Parameters
        ----------
        text_id_col_name : str, optional
            Column name to be used in `.tokenized_df` to
            denote the text-id for which the token belongs, by default "text_id".
        skip_special_tokens : bool, optional
            Whether or not to skip special tokens added by the
            transformer tokenizer, by default True.
        **tokenizer_args
            Passed along to the `.tokenizer()` method.

        Returns
        -------
        BatchEncoding
            The tokenized text as BatchEncoding type.

        Raises
        ------
        ValueError
            if `text_id_column_name` is already a column name in `.df` dataframe.
            In this case, will need to pass in a different string.
        """
        if text_id_col_name in self.df.columns:
            raise ValueError(
                f"'{text_id_col_name}' is already a column name in the `.df` dataframe. "
                "Use a different column name to denote the corresponding text-ids."
            )

        # record whether or not special tokens to be skipped when obtaining the tokenized text
        self.skip_special_tokens = skip_special_tokens
        if not tokenizer_args:
            tokenizer_args = {"padding": True, "truncation": True}
        if tokenizer_args.get("return_tensors") != "pt":
            print("setting return_tensors='pt'")
            tokenizer_args["return_tensors"] = "pt"
        if not tokenizer_args.get("return_special_tokens_mask"):
            print("setting return_special_tokens_mask=True")
            tokenizer_args["return_special_tokens_mask"] = True

        # tokenize text
        self.tokens = self.tokenizer(
            self.df[self.col_name_text].to_list(), **tokenizer_args
        )
        # save the special tokens mask
        # used to appropriately slice the output when obtaining token embeddings
        self.special_tokens_mask = self.tokens.data.pop("special_tokens_mask")

        # convert the token_ids back to tokens and save to the `.df` dataframe
        self.df["tokens"] = [
            self.tokenizer.convert_ids_to_tokens(
                self.tokens["input_ids"][i],
                skip_special_tokens=self.skip_special_tokens,
            )
            for i in range(len(self.df))
        ]
        # if decided to skip special tokens, also convert back to tokens
        # and include all the special tokens
        if self.skip_special_tokens:
            self.df["all_tokens"] = [
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
            columns=[self.col_name_text, "all_tokens"],
            errors="ignore",
        ).explode("tokens")
        self.tokenized_df = self.tokenized_df.reset_index()
        print(
            f"'{text_id_col_name}' is the column name for denoting the corresponding text id"
        )
        self.tokenized_df = self.tokenized_df.rename(
            columns={"index": text_id_col_name}
        )
        self.text_id_col_name = text_id_col_name

        return self.tokens

    def obtain_embeddings(
        self,
        method: str = "hidden_layer",
        layers: Optional[Union[int, List[int], Tuple[int]]] = None,
    ) -> np.array:
        """
        Once text has been tokenized (using `.tokenize_text`), can obtain token embeddings
        for each token in `.tokenized_df["tokens"]`.

        Method passes in the tokens (in `.tokens`) through the transformer model
        and obtains token embeddings by combining the hidden layers in some way.
        See `method` argument below for options.

        Parameters
        ----------
        method : str, optional
            Method for combining the layer hidden states, by default "hidden_layer".
            Options are:
            - "hidden_layer":
                - if `layers` is just an integer, token embedding will be taken as
                is taken from the hidden state in layer number `layers`.
                By default (if `layers` is not specified), the token embeddings are
                taken from the second-to-last layer hidden state.
                - if `layers` is a list of integers, will return the
                layer hidden states from the specified layers.
            - "concatenate":
                - if `layers` is a list of integers, will return the
                concatenation of the layer hidden states from the specified layers.
                By default (if `layers` is not specified), the token embeedings are
                computed by concatenating the layer hidden states from the last 4 layers
                (or all the hidden states if the number of hidden states is less than 4).
                - if `layers` is just an integer, token embedding will be taken as
                is taken from the hidden state in layer number `layers`
                (as concatenation of one layer hidden state is just that layer).
            - "sum":
                - if `layers` is a list of integers, will return the
                sum of the layer hidden states from the specified layers.
                By default (if `layers` is not specified), the token embeedings are
                computed by concatenating the layer hidden states from the last 4 layers
                (or all the hidden states if the number of hidden states is less than 4).
                - if `layers` is just an integer, token embedding will be taken as
                is taken from the hidden state in layer number `layers`
                (as sum of one layer hidden state is just that layer).
            - "mean":
                - if `layers` is a list of integers, will return the
                mean of the layer hidden states from the specified layers.
                By default (if `layers` is not specified), the token embeedings are
                computed by concatenating the layer hidden states from the last 4 layers
                (or all the hidden states if the number of hidden states is less than 4).
                - if `layers` is just an integer, token embedding will be taken as
                is taken from the hidden state in layer number `layers`
                (as mean of one layer hidden state is just that layer).
        layer : Optional[Union[int, Iterable[int]]]
            The layers to use when combining the hidden states of the transformer.

        Returns
        -------
        np.array
            Unless `method=hidden_layer` and `layers` is a list of integers, the
            method returns a 2 dimensional array with dimensions [token, embedding],
            i.e. the number of rows is the number of tokens in `.tokenized_df`,
            and the number of columns is the dimension of the embeddings.
            If `method=hidden_layer` and `layers` is a list of integers, the method
            returns a 3 dimensional array with dimensions [layer, token, embedding],
            i.e. the first dimension denotes the layers that was requested, the second
            dimension is the tokens (as found in `.tokenized_df`) and the third
            dimension is the embeddings. This option is added so that the user
            can combine the hidden layers in some custom way.

        Raises
        ------
        ValueError
            if `.tokens` is None. Means the text has not been tokenized yet and `.tokenize_text()`
            needs to be called first.
        ValueError
            if layers is not an integer or a list of integers.
        ValueError
            if any of the requested layers are out of range, i.e. if any are less than zero, or
            are larger than the number of hidden layers in the transformer architecture.
        NotImplementedError
            if requested `method` is not one of "hidden_layer", "concatenate", "sum" or "mean".
        """
        if self.tokens is None:
            raise ValueError(
                "Text has not been tokenized yet. Call `.tokenize_text()` first."
            )
        if (layers is not None) and (not isinstance(layers, (int, list, tuple))):
            raise ValueError(
                "layers requested must be either integer or list of integers."
            )

        # as the output of the transformer includes padded tokens or special tokens
        # find indices of the tokens that we are interested
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

        # pass tokens through the model to obtain embeddings
        with torch.no_grad():
            outputs = self.model(**self.tokens, output_hidden_states=True)
            hidden_states = torch.stack(outputs["hidden_states"], dim=0)

        # by default, we will concatenate the embeddings at the end unless we've requested to
        # obtain hidden states from multiple layers (in which case they cannot be concatenated)
        concatenate_embeddings = True
        if method == "hidden_layer":
            if layers is None:
                # if not specified layers wanted, returns the second to last one
                layers = hidden_states.shape[0] - 1
            # only consider layers requested
            if isinstance(layers, int):
                if layers >= hidden_states.shape[0]:
                    raise ValueError(
                        f"requested layer ({layers}) is out of range: only have "
                        f"{hidden_states.shape[0]} number of hidden layers"
                    )
                elif layers < 0:
                    raise ValueError(
                        f"requested layer ({layers}) is out of range: "
                        "must be greater than or equal to 0, and we only have "
                        f"{hidden_states.shape[0]} number of hidden layers"
                    )
                # only want one layer so hidden_states[layers] is tensor with
                # [batch, tokens, embeddings]
                hidden_states = hidden_states[layers]
                # obtain hidden layer for each item in the dataframe
                self.token_embeddings = [
                    hidden_states[i][indices[i], :] for i in range(len(self.df))
                ]
            else:
                if any(
                    [
                        (layer < 0) or (layer >= hidden_states.shape[0])
                        for layer in layers
                    ]
                ):
                    raise ValueError(
                        f"requested layers ({layers}) is out of range: only have "
                        f"{hidden_states.shape[0]} number of hidden layers"
                    )
                # hidden_states[layers] is tensor with [layers, batch, tokens, embeddings]
                # change order of tensor to [batch, layers, tokens, embeddings]
                hidden_states = hidden_states[layers].permute(1, 0, 2, 3)
                # obtain hidden layers for each item in the dataframe
                self.token_embeddings = [
                    hidden_states[i][:, indices[i], :].numpy()
                    for i in range(len(self.df))
                ]
                concatenate_embeddings = False
        elif method in ["concatenate", "sum", "mean"]:
            if layers is None:
                # if not specified layers wanted, concatenates last 4 hidden layers
                # (if is possible to go 4 layers back, otherwise takes all the hidden states)
                layers = [
                    hidden_states.shape[0] - i
                    for i in range(1, 5)
                    if hidden_states.shape[0] - i >= 0
                ]
            if any(
                [(layer < 0) or (layer >= hidden_states.shape[0]) for layer in layers]
            ):
                raise ValueError(
                    f"requested layers ({layers}) is out of range: only have "
                    f"{hidden_states.shape[0]} number of hidden layers"
                )
            # hidden_states[layers] is tensor with [layers, batch, tokens, embeddings]
            # change order of tensor to [batch, tokens, layers, embeddings]
            hidden_states = hidden_states[layers].permute(1, 2, 0, 3)
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
                f"method '{method}' for pooling hidden layers "
                "to obtain token embeddings has not been implemented."
            )
        if concatenate_embeddings:
            self.token_embeddings = torch.cat(self.token_embeddings).numpy()
        return self.token_embeddings

    def pool_token_embeddings(self, method: str = "mean") -> np.array:
        """
        Once token embeddings have been computed (using `.obtain_embeddings`),
        `.token_embeddings` is a 2 dimensional array with dimensions [token, embedding].
        We can pool these together to obtain an embedding for the whole sentence
        by pooling the token embeddings in the sentence.
        See `method` argument below for options.

        Note that if `.token_embeddings` is a 3 dimensional array with dimensions
        [layer, token, embedding], the hidden layers must be pooled first to obtain
        a 2 dimensional array with dimensions [token, embedding].

        Note that `SentenceEncoder` might be more appropriate for obtaining
        sentence embeddings which uses SBERT via the `sentence-transformers` package.

        Parameters
        ----------
        method : str, optional
            Method for combining the token embeddings, by default "mean".
            Options are:
            - "mean": takes the mean average of token embeddings
            - "max": takes the maximum in each dimension of the token embeddings
            - "sum": takes the element-wise sum of the token embeddings
            - "cls": takes the 'cls' embedding (only possible if
            `skip_special_tokens=False` was set when tokenizing the text)

        Returns
        -------
        np.array
            A 2 dimensional array with dimensions [sentences, embedding],
            i.e. the number of rows is the number of sentences/texts in `.df`,
            and the number of columns is the dimension of the embeddings.

        Raises
        ------
        ValueError
            if `.token_embeddings` is None. Means the token embeddings have not
            been computed yet and `.obtain_embeddings()` needs to be called first.
        ValueError
            if `.token_embeddings` is not a 2 dimensional array. In which case, it can
            be that it is a 3 dimensional array with dimensions [layer, token, embedding],
            and the hidden layers must be pooled first to obtain a 2 dimensional array
            with dimensions [token, embedding].
        ValueError
            if `method="cls"` but `skip_special_tokens=False` was set when
            tokenizing the text. This means that when the token embeddings were obtained,
            the embedding corresponding to the 'cls' token was not saved.
        NotImplementedError
            if requested `method` is not one of "mean", "max", "sum" or "cls".
        """
        if self.token_embeddings is None:
            raise ValueError(
                "Token embeddings have not been computed yet. "
                "Call `.obtain_embeddings()` first."
            )
        if self.token_embeddings.ndim != 2:
            raise ValueError(
                "The token embeddings (in `.token_embeddings`) must be a "
                "numpy array with dimensions [token, embedding]."
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
                    "We skipped special tokens and so cls embedding has not been saved. "
                    "If want cls embeddings, will need to obtain token "
                    "embeddings again but set `.skip_special_tokens=False` first."
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
                f"Method '{method}' for pooling the token embeddings not been implemented."
            )
        self.pooled_embeddings = np.stack(self.pooled_embeddings)
        return self.pooled_embeddings

    def fit_transformer(self):
        """fit / fine-tune transformer model to some task"""
        # TODO
        pass
