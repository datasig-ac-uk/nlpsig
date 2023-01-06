import pickle
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets.arrow_dataset import Dataset
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
)


class SentenceEncoder:
    """
    Class to obtain sentence embeddings using SentenceTransformer class in `sentence_transformers`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_name: str,
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
        feature_name : str
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
            if `feature_name` is not a column in df
        """
        self.df = df
        if feature_name not in df.columns:
            raise KeyError(f"{feature_name} is not a column in df")
        self.feature_name = feature_name
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
        sentences = self.df[self.feature_name].to_list()
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
        feature_name: str,
        full_dataset: Optional[Dataset] = None,
        model_name: str = "bert-base-uncased",
    ):
        """
        Class to obtain token embeddings (and optionally pool them)
        using Huggingface transformers.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset as a pandas dataframe
        feature_name : str
            Column name which has the text in
        model_name : str, optional
            Name of transformer encoder model, by default "bert-base-uncased"

        Raises
        ------
        KeyError
            if `feature_name` is not a column in df
        """
        if df is not None:
            # df is passed in, so create the Dataset object from the dataframe
            # will override any Dataset that is passed in to ensure consistency
            self.df = df
            self.dataset = Dataset.from_pandas(df)
            self._features = self.dataset.features.keys()
        else:
            # df is not passed in, must have Dataset passed into full_dataset
            if isinstance(full_dataset, Dataset):
                self.df = pd.DataFrame(full_dataset)
                self.dataset = full_dataset
                self._features = self.dataset.features.keys()
            else:
                raise ValueError(
                    "if df is not passed in, then full_dataset "
                    "must be a Dataset object"
                )
        if isinstance(feature_name, str):
            # convert to list of one element
            feature_name = [feature_name]
        elif isinstance(feature_name, list):
            # if feature_name is a list, it can only be of length 1
            # (if only one column we want to process), or of length 2
            # (if we have pairs of sentences to process)
            if len(feature_name) not in [1, 2]:
                raise ValueError(
                    "if feature_name is a list, it must be " "a list of length 1 or 2"
                )
            for col in feature_name:
                if col not in df.columns:
                    raise KeyError(f"{col} is not a column in df")
        else:
            raise ValueError(
                "if df is passed in, then feature_name "
                "must either be a string, or a list of strings"
            )
        self.feature_name = feature_name
        self.tokenized_df = None
        self.token_embeddings = None
        self.pooled_embeddings = None
        self.model_name = model_name
        self.model = None
        self.config = None
        self.tokenizer = None
        self.data_collator = None
        self.skip_special_tokens = None
        self.text_id_col_name = None

    def load_pretrained_model(self, force_reload: bool = False) -> None:
        """
        Loads in config, tokenizer and pretrained weights from transformers.

        Parameters
        ----------
        force_reload : bool, optional
            Whether or not to overwrite current loaded model, by default False.
        """
        if (not force_reload) and (self.model is not None):
            print(f"[INFO] {self.model_name} model is already loaded")
            return
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    def initialise_transformer(self, force_reload: bool = False, **config_args) -> None:
        """
        Loads in config and tokenizer. initialises the transformer with random weights
        from transformers.

        Parameters
        ----------
        force_reload : bool, optional
            Whether or not to overwrite current loaded model, by default False.
        **cofig_args :
            Passed along to `AutoConfig.from_pretrained()` method.
        """
        if (not force_reload) and (self.model is not None):
            print(f"[INFO] {self.model_name} model is already loaded")
            return
        self.config = AutoConfig.from_pretrained(self.model_name, **config_args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.model = AutoModel.from_config(self.config)
        self.model.eval()

    def tokenize_text(
        self,
        text_id_col_name: str = "text_id",
        skip_special_tokens: bool = True,
        batched=True,
        batch_size=1000,
        **tokenizer_args,
    ) -> Dataset:
        """
        Method to tokenize each item in the `feature_name` column of the dataframe.

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
            Passed along to the `.tokenizer()` method. By default, we pass the following
            arguments:
            - `padding` = False (as dynamic padding is used later)
            - `truncation` = True
            - `return_special_tokens_mask` = True
            (this is always used and overrides user option if passed)

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
        if self.model is None:
            raise NotImplementedError(
                "No model has been initialised yet. First call "
                "`.load_pretrained_model()` or `.initialise_transformer()`."
            )
        if text_id_col_name in self.df.columns:
            raise ValueError(
                f"'{text_id_col_name}' is already a column name in the `.df` dataframe. "
                "Use a different column name to denote the corresponding text-ids."
            )

        # record whether or not special tokens to be skipped when obtaining the tokenized text
        self.skip_special_tokens = skip_special_tokens
        if not tokenizer_args:
            # by default does not perform padding initially,
            # as will utilise dynamic padding later on
            tokenizer_args = {"padding": False, "truncation": True}
        if not tokenizer_args.get("return_special_tokens_mask"):
            print("[INFO] Setting return_special_tokens_mask=True")
            tokenizer_args["return_special_tokens_mask"] = True

        # define tokenize_function for mapping to Dataset object
        if len(self.feature_name) == 1:
            # we have just a list of sentences to tokenize
            def tokenize_function(dataset):
                return self.tokenizer(
                    dataset[self.feature_name[0]],
                    **tokenizer_args,
                )

        elif len(self.feature_name) == 2:
            # we have pairs of sentences to tokenize
            def tokenize_function(dataset):
                return self.tokenizer(
                    dataset[self.feature_name[0]],
                    dataset[self.feature_name[1]],
                    **tokenizer_args,
                )

        # tokenize the dataset and save the tokens in .tokens attribute
        print("[INFO] Tokenizing the datatset...")
        self.dataset = self.dataset.map(
            tokenize_function,
            batched=batched,
            batch_size=batch_size,
        )
        self.tokens = self.dataset.remove_columns(self._features)

        print(
            "[INFO] Saving the tokenized text for each sentence into .df['tokens']..."
        )
        # save the tokenized text to `.df["tokens"] (does not include special tokens)
        self.df["tokens"] = [
            self.tokenizer.convert_ids_to_tokens(
                self.tokens["input_ids"][i],
                skip_special_tokens=self.skip_special_tokens,
            )
            for i in tqdm(range(len(self.df)))
        ]

        # create new tokenized dataframe
        print(
            "[INFO] Creating tokenized dataframe and setting in .tokenized_df attribute"
        )
        self.tokenized_df = self.df.drop(
            columns=self.feature_name,
            errors="ignore",
        ).explode("tokens")
        self.tokenized_df = self.tokenized_df.reset_index()
        print(
            f"[INFO] Note: '{text_id_col_name}' is the "
            "column name for denoting the corresponding text id"
        )
        self.tokenized_df = self.tokenized_df.rename(
            columns={"index": text_id_col_name}
        )
        self.text_id_col_name = text_id_col_name

        return self.dataset

    def _obtain_embeddings_for_batch(
        self,
        batch_tokens: BatchEncoding,
        method: str,
        layers: Optional[Union[int, List[int], Tuple[int]]],
    ) -> Union[np.array, List[np.array]]:
        """
        For a given batch of tokens, `batch_tokens`, compute

        Method passes in the tokens (in `batch_tokens`) through the
        transformer model and obtains token embeddings by combining
        the hidden layers in some way.
        See `method` argument below for options.

        Parameters
        ----------
        batch_tokens : BatchEncoding
            Batch of tokens.
        method : str
            See overview of methods in `.obtain_embeddings()` method.
        layers : Optional[Union[int, List[int], Tuple[int]]]
            See description of layers in `.obtain_embeddings()` method.

        Returns
        -------
        Union[np.array, List[np.array]]
            Unless `method=hidden_layer` and `layers` is a list of integers, the
            method returns a 2 dimensional array with dimensions [token, embedding],
            i.e. the number of rows is the number of tokens in `.tokenized_df`,
            and the number of columns is the dimension of the embeddings.

            If `method=hidden_layer` and `layers` is a list of integers, the method
            returns a list of 3 dimensional arrays with dimensions [layer, token, embedding].
            Each item in the list is the output of the hidden layers requested for each
            sentence, i.e. for each item, the first dimension denotes the layers
            that was requested, the second dimension is the tokens
            (as found in `.tokenized_df`) and the third dimension is the embeddings.
            This option is added so that the user can combine the hidden layers
            in some custom way.
        """
        # as the output of the transformer includes padded tokens or special tokens
        # find indices of the tokens that we are interested
        if self.skip_special_tokens:
            # finds indices of non-special tokens
            indices = [
                torch.where(batch_tokens["special_tokens_mask"][i] == 0)[0]
                for i in range(len(batch_tokens["input_ids"]))
            ]
        else:
            # finds indices of non-pad tokens
            indices = [
                torch.where(batch_tokens["attention_mask"][i] == 1)[0]
                for i in range(len(batch_tokens["input_ids"]))
            ]

        # remove "special_tokens_mask"
        del batch_tokens["special_tokens_mask"]
        # pass tokens through the model to obtain embeddings
        with torch.no_grad():
            outputs = self.model(
                **batch_tokens,
                output_hidden_states=True,
            )
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
                # [sentence, tokens, embeddings]
                hidden_states = hidden_states[layers]
                # obtain hidden layer for each item in the dataframe
                token_embeddings = [
                    hidden_states[i][indices[i], :]
                    for i in range(len(batch_tokens["input_ids"]))
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
                # hidden_states[layers] is tensor with [layers, sentence, tokens, embeddings]
                # change order of tensor to [sentence, layers, tokens, embeddings]
                hidden_states = hidden_states[layers].permute(1, 0, 2, 3)
                # obtain hidden layers for each item in the dataframe
                token_embeddings = [
                    hidden_states[i][:, indices[i], :].numpy()
                    for i in range(len(batch_tokens["input_ids"]))
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
            # hidden_states[layers] is tensor with [layers, sentence, tokens, embeddings]
            # change order of tensor to [sentence, tokens, layers, embeddings]
            hidden_states = hidden_states[layers].permute(1, 2, 0, 3)
            if method == "concatenate":
                token_embeddings = [
                    torch.stack(
                        [
                            hidden_states[i][indices[i]][j].flatten()
                            for j in range(len(indices[i]))
                        ]
                    )
                    for i in range(len(batch_tokens["input_ids"]))
                ]
            elif method == "sum":
                token_embeddings = [
                    torch.stack(
                        [
                            hidden_states[i][indices[i]][j].sum(dim=0)
                            for j in range(len(indices[i]))
                        ]
                    )
                    for i in range(len(batch_tokens["input_ids"]))
                ]
            elif method == "mean":
                token_embeddings = [
                    torch.stack(
                        [
                            hidden_states[i][indices[i]][j].mean(dim=0)
                            for j in range(len(indices[i]))
                        ]
                    )
                    for i in range(len(batch_tokens["input_ids"]))
                ]
        else:
            raise NotImplementedError(
                f"method '{method}' for pooling hidden layers "
                "to obtain token embeddings has not been implemented."
            )
        if concatenate_embeddings:
            token_embeddings = torch.cat(token_embeddings).numpy()
        return token_embeddings

    def obtain_embeddings(
        self,
        method: str = "hidden_layer",
        batch_size: int = 1000,
        layers: Optional[Union[int, List[int], Tuple[int]]] = None,
    ) -> Union[np.array, List[np.array]]:
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
        Union[np.array, List[np.array]]
            Unless `method=hidden_layer` and `layers` is a list of integers, the
            method returns a 2 dimensional array with dimensions [token, embedding],
            i.e. the number of rows is the number of tokens in `.tokenized_df`,
            and the number of columns is the dimension of the embeddings.

            If `method=hidden_layer` and `layers` is a list of integers, the method
            returns a list of 3 dimensional arrays with dimensions [layer, token, embedding].
            Each item in the list is the output of the hidden layers requested for each
            sentence, i.e. for each item, the first dimension denotes the layers
            that was requested, the second dimension is the tokens
            (as found in `.tokenized_df`) and the third dimension is the embeddings.
            This option is added so that the user can combine the hidden layers
            in some custom way.

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

        # obtain batches of the tokens using dynamic padding
        data_loader = DataLoader(
            self.tokens,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=self.data_collator,
        )

        self.token_embeddings = [
            self._obtain_embeddings_for_batch(batch, method, layers)
            for batch in tqdm(data_loader)
        ]
        if isinstance(self.token_embeddings[0], list):
            # have a list of lists, need to flatten it
            self.token_embeddings = [
                item for sublist in self.token_embeddings for item in sublist
            ]
        else:
            # have a list of numpy arrays which we can concatenate
            self.token_embeddings = np.concatenate(self.token_embeddings)

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
        if isinstance(self.token_embeddings, list):
            raise ValueError(
                "The token embeddings (in `.token_embeddings`) is currently a list. "
                "This might be because no pooling of the hidden states have been done "
                "to obtain the embeddings for each token. "
                "The token embeddings must be a numpy array with dimensions "
                "[token, embedding] in order to pool them."
            )
        if self.token_embeddings.ndim != 2:
            raise ValueError(
                "The token embeddings (in `.token_embeddings`) must be a "
                "numpy array with dimensions [token, embedding] "
                "in order to pool them."
            )
        if method == "mean":
            self.pooled_embeddings = [
                self.token_embeddings[
                    self.tokenized_df.index[
                        self.tokenized_df[self.text_id_col_name] == i
                    ]
                ].mean(axis=0)
                for i in tqdm(range(len(self.df)))
            ]
        elif method == "max":
            self.pooled_embeddings = [
                self.token_embeddings[
                    self.tokenized_df.index[
                        self.tokenized_df[self.text_id_col_name] == i
                    ]
                ].max(axis=0)
                for i in tqdm(range(len(self.df)))
            ]
        elif method == "sum":
            self.pooled_embeddings = [
                self.token_embeddings[
                    self.tokenized_df.index[
                        self.tokenized_df[self.text_id_col_name] == i
                    ]
                ].sum(axis=0)
                for i in tqdm(range(len(self.df)))
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
                    for i in tqdm(range(len(self.df)))
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
