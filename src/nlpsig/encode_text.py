from __future__ import annotations

import pickle
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    DataCollator,
    DataCollatorWithPadding,
    EvalPrediction,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


class SentenceEncoder:
    """
    Class to obtain sentence embeddings using SentenceTransformer class in `sentence_transformers`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_name: str,
        model_name: str = "all-MiniLM-L6-v2",
        model_modules: Iterable[nn.Module] | None = None,
        model_encoder_args: dict | None = None,
        model_fit_args: dict | None = None,
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
        model_modules : Iterable[nn.Module] | None, optional
            This parameter can be used to create custom
            SentenceTransformer models from scratch. See
            https://www.sbert.net/docs/training/overview.html#creating-networks-from-scratch
            for examples.
            If creating a custom model using `.load_custom_model()` method,
            passes this into the `modules` argument when initialising
            `SentenceTransformer` object.
        model_encoder_args : dict | None, optional
            Any keywords to be passed into the model for encoding sentences,
            by default the following arguments to pass into the
            `.encode()` method of SentenceTransformer class:
            {"batch_size": 64,
             "show_progress_bar": True,
             "output_value": "sentence_embedding",
             "convert_to_numpy": True,
             "convert_to_tensor": False,
             "device": None,
             "normalize_embeddings": False}
        model_fit_args : dict | None, optional
            Any keywords to be passed into the model to fine-tune sentence transformer,
            by default None

        Raises
        ------
        KeyError
            if `feature_name` is not a column in df.
        """
        self.df = df
        if feature_name not in df.columns:
            raise KeyError(f"{feature_name} is not a column in df")
        self.feature_name = feature_name
        self.sentence_embeddings = None
        self.model_name = model_name
        self.model_modules = model_modules
        if model_encoder_args is None:
            model_encoder_args = {
                "batch_size": 64,
                "show_progress_bar": True,
                "output_value": "sentence_embedding",
                "convert_to_numpy": True,
                "convert_to_tensor": False,
                "device": None,
                "normalize_embeddings": False,
            }
        self.model_encoder_args = model_encoder_args
        if model_fit_args is None:
            model_fit_args = {}

        self.model_fit_args = model_fit_args
        self.model = None

    def load_pre_computed_embeddings(self, pre_computed_embeddings_file: str) -> None:
        """
        Loads in pre-computed sentence embeddings.

        Parameters
        ----------
        pre_computed_embeddings_file : str
            Path to pre-computed embeddings, by default None.

        Raises
        ------
        ValueError
            if the loaded embeddings is not a (n x d) array,
            where n is the number of sentences (in `.df`)
            and d is the dimension of the embeddings.
        """
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

    def load_pretrained_model(self, force_reload: bool = False) -> None:
        """
        Loads pre-trained model into `.model` by passing in `.model_name` to
        the `model_name_or_path` argument when initialising `SentenceTransformer` object.

        `.model_name` can also be path to a trained model.

        Parameters
        ----------
        force_reload : bool, optional
            Whether or not to overwrite current loaded model, by default False.

        Raises
        ------
        NotImplementedError
            if `.model_name` cannot be loaded by SentenceTransformer.
            This might happen if this is not a pre-trained model available.
            See https://www.sbert.net/docs/pretrained_models.html for examples.
        """
        if (not force_reload) and (self.model is not None):
            print(f"[INFO] '{self.model_name}' model is already loaded")
            return
        if (force_reload) and (self.model == "pre-computed"):
            print(
                "[INFO] The current embeddings were computed before "
                "and were loaded into this class"
            )
            return
        try:
            self.model = SentenceTransformer(model_name_or_path=self.model_name)
        except Exception as err:
            raise NotImplementedError(
                f"Loading model '{self.model_name}' with SentenceTransformer failed. "
                "See SentenceTransformer documentation in sentence_transformers."
            ) from err

    def load_custom_model(self, force_reload: bool = False) -> None:
        """
        Loads pre-trained model into `.model` by passing in `.model_name` to
        the `modules` argument when initialising `SentenceTransformer` object.

        Parameters
        ----------
        force_reload : bool, optional
            Whether or not to overwrite current loaded model, by default False.

        Raises
        ------
        ValueError
            if there is nothing stored in `.model_modules` attribute to initialise
            SentenceTransformer model.
        NotImplementedError
            if loading in a model using the modules in `.model_modules` was unsuccessful.
            This might happen if any of the items in `.model_modules` were not valid modules.
            See https://www.sbert.net/docs/training/overview.html#creating-networks-from-scratch
            for examples.
        """
        if (not force_reload) and (self.model is not None):
            print(f"[INFO] '{self.model_name}' model is already loaded")
            return
        if (force_reload) and (self.model == "pre-computed"):
            print(
                "[INFO] The current embeddings were computed before "
                "and were loaded into this class"
            )
            return
        if self.model_modules is None:
            raise ValueError(
                "`.model_modules` must be a list of modules which define the network architecture."
            )
        try:
            self.model = SentenceTransformer(modules=self.model_modules)
        except Exception as err:
            raise NotImplementedError(
                f"Loading model with modules: {self.model_modules}, with SentenceTransformer "
                "failed. See SentenceTransformer documentation in sentence_transformers "
                "(https://www.sbert.net/docs/training/"
                "overview.html#creating-networks-from-scratch) "
                "for information on how to create the networks architectures from scratch."
            ) from err

    def obtain_embeddings(self) -> np.array:
        """
        Obtains sentence embeddings via the `.encode` method,
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
        self, train_objectives: Iterable[tuple[DataLoader, nn.Module]]
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
    Class to obtain token embeddings (and optionally pool them)
    using Huggingface transformers.
    """

    def __init__(
        self,
        feature_name: str,
        df: pd.DataFrame | None = None,
        dataset: Dataset | None = None,
        model_name: str | None = None,
        model: PreTrainedModel | None = None,
        config: PretrainedConfig | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        data_collator: DataCollator | None = None,
    ):
        """
        Class to obtain token embeddings (and optionally pool them)
        using Huggingface transformers.

        Parameters
        ----------

        feature_name : str
            Column name which has the text in.
        df : pd.DataFrame | None, optional
            Dataset as a pandas dataframe, by default None.
            If `df` is not provided, `dataset` must be provided.
            A dataframe will then be created from it.
        dataset : Dataset | None, optional
            Huggingface Dataset object for the full dataset, by default None.
            If `df` is a dataframe, a Dataset will be created from it,
            even if `dataset` is provided.
        model_name : str | None, optional
            Name of transformer encoder model from Huggingface Hub, by default None.
            To be used if want to load in a pretrained model.
        model : PreTrainedModel | None, optional
            Huggingface transformer model class, by default None.
        config : PretrainedConfig | None, optional
            Huggingface configuration class, by default None.
        tokenizer : PreTrainedTokenizer | None, optional
            Huggingface tokenizer class, by default None.
        data_collator : DataCollator | None, optional
            Data collator to use, by default None.
            Should work with the tokenizer that is passed in.
        """
        # check feature name is a string or list of length 1 or 2 of strings
        if isinstance(feature_name, str):
            # convert to list of one element
            feature_name = [feature_name]
        elif isinstance(feature_name, list):
            # if feature_name is a list, it can only be of length 1
            # (if only one column we want to process), or of length 2
            # (if we have pairs of sentences to process)
            if len(feature_name) not in [1, 2]:
                raise ValueError(
                    "If `feature_name` is a list, it must be a list of length 1 or 2."
                )
        else:
            raise ValueError(
                "`feature_name` must either be a string, or a list of strings."
            )

        # load in dataframe or Dataset
        if df is not None:
            # df is passed in, so create the Dataset object from the dataframe
            # check feature names passed in exist in the dataframe passed
            for col in feature_name:
                if col not in df.columns:
                    raise KeyError(f"'{col}' is not a column in `df`.")

            # will override any Dataset that is passed in to ensure consistency
            self.df: pd.DataFrame = df
            self.dataset: Dataset = Dataset.from_pandas(df)
        else:
            # df is not passed in, must have Dataset passed into dataset
            if isinstance(dataset, Dataset):
                self.df: pd.DataFrame = pd.DataFrame(dataset)
                self.dataset: Dataset = dataset
            else:
                raise TypeError(
                    "If `df` is not passed in, then `dataset` "
                    "must be a Dataset object."
                )

            # check feature names passed in exist in the Dataset passed
            for col in feature_name:
                if col not in dataset.features:
                    raise KeyError(f"'{col}' is not a column in `df`.")

        self._features = list(self.dataset.features.keys())
        self.feature_name = feature_name
        self.tokenized_df = None
        self.token_embeddings = None
        self.pooled_embeddings = None
        self.model_name = model_name
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.training_args = None
        self.trainer = None
        self.dataset_split = None
        self.skip_special_tokens = None
        self.text_id_col_name = None

    def load_pretrained_model(self, force_reload: bool = False) -> None:
        """
        Loads in config, tokenizer and pretrained weights from transformers,
        using `AutoConfig`, `AutoTokenizer`, `AutoModel`.

        If another model is required, e.g. model for masked language modelling,
        then recommended to load the model using the appropriate class,
        e.g. `AutoModelForMaskedLM()` to load in the model and reset
        `.model` attribute to this object. This is required if
        you wish to train / pre-train the model to the data later.

        Parameters
        ----------
        force_reload : bool, optional
            Whether or not to overwrite current loaded model, by default False.
        """
        if (not force_reload) and (self.model is not None):
            print(f"[INFO] '{self.model_name}' model is already loaded.")
            return
        if self.model_name is None:
            raise TypeError("")
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        Warning(
            "[INFO] By default, `.load_pretrained_model()` uses "
            "`AutoModel` to load in the model. "
            "If you want to load the model for a specific task, "
            "reset the `.model` attribute."
        )
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    def initialise_transformer(self, force_reload: bool = False, **config_args) -> None:
        """
        Loads in config and tokenizer. Initialises the transformer with random weights
        from transformers.

        Parameters
        ----------
        force_reload : bool, optional
            Whether or not to overwrite current loaded model, by default False.
        **config_args :
            Passed along to `AutoConfig.from_pretrained()` method.
        """
        if (not force_reload) and (self.model is not None):
            print(f"[INFO] '{self.model_name}' model is already loaded.")
            return
        if self.model_name is None:
            raise TypeError("")
        self.config = AutoConfig.from_pretrained(self.model_name, **config_args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        Warning(
            "[INFO] By default, `.initialise_transformer()` uses "
            "`AutoModel` to load in the model. "
            "If you want to load the model for a specific task, "
            "reset the `.model` attribute."
        )
        self.model = AutoModel.from_config(self.config)
        self.model.eval()

    def _check_model(self) -> None:
        if self.model is None:
            raise NotImplementedError(
                "No model has been initialised yet. First pass in a model "
                "into the `.model` attribute, or call "
                "`.load_pretrained_model()` or `.initialise_transformer()`."
            )
        if self.tokenizer is None:
            raise NotImplementedError(
                "No tokenizer has been initialised yet. First pass in a tokenizer "
                "into the `.tokenizer` attribute, or call "
                "`.load_pretrained_model()` or `.initialise_transformer()`."
            )
        if self.data_collator is None:
            raise NotImplementedError(
                "No data collator has been initialised yet. First pass in a data collator "
                "into the `.data_collator` attribute, or call "
                "`.load_pretrained_model()` or `.initialise_transformer()`."
            )

    def tokenize_text(
        self,
        text_id_col_name: str = "text_id",
        skip_special_tokens: bool = True,
        batched: bool = True,
        batch_size: int = 1000,
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
        batched : bool, optional
            Whether or not to tokenize the text in batches, by default True.
        batch_size: int, optional
            The size of the batches (if used), by default 1000.
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
        # check model, tokenizer and data_collator have been passed into the class
        self._check_model()

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
        print("[INFO] Tokenizing the dataset...")
        self.dataset = self.dataset.map(
            tokenize_function,
            batched=batched,
            batch_size=batch_size,
        )
        self.tokens = self.dataset.remove_columns(self._features)

        # save the tokenized text to `.df["tokens"] (does not include special tokens)
        print(
            "[INFO] Saving the tokenized text for each sentence into `.df['tokens']`..."
        )

        cls_token_avail = self.tokenizer.cls_token is not None

        def tokenize_decoder(dataset):
            tokens = []
            for i in range(len(dataset["input_ids"])):
                # find the indices we're interested depending on whether or not we want to skip special tokens
                if self.skip_special_tokens:
                    ind = torch.where(
                        torch.tensor(dataset["special_tokens_mask"][i]) == 0
                    )[0].tolist()
                else:
                    ind = torch.where(torch.tensor(dataset["attention_mask"][i]) == 1)[
                        0
                    ].tolist()
                # if no tokens (i.e. empty string), then just return the first embedding (CLS token, if available)
                if len(ind) == 0 and cls_token_avail:
                    ind = [0]
                tokens.append(
                    self.tokenizer.convert_ids_to_tokens(
                        torch.tensor(dataset["input_ids"][i])[ind]
                    )
                )
            return {"tokens": tokens}

        self.dataset = self.dataset.map(
            tokenize_decoder,
            batched=batched,
            batch_size=batch_size,
        )
        self.df["tokens"] = self.dataset["tokens"]

        # create new tokenized dataframe
        print(
            "[INFO] Creating tokenized dataframe and setting in `.tokenized_df` attribute..."
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
        layers: int | list[int] | tuple[int] | None,
    ) -> np.array | list[np.array]:
        """
        [Private] For a given batch of tokens, `batch_tokens`,
        compute token embeddings from hidden layers.

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
        layers : int | list[int] | tuple[int] | None
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
                if len(torch.where(batch_tokens["special_tokens_mask"][i] == 0)[0]) > 0
                else [0]
                for i in range(len(batch_tokens["input_ids"]))
            ]
        else:
            # finds indices of non-pad tokens
            indices = [
                torch.where(batch_tokens["attention_mask"][i] == 1)[0]
                if len(torch.where(batch_tokens["attention_mask"][i] == 1)[0]) > 0
                else [0]
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
                # if not specified layers wanted, returns the last hidden layer
                layers = hidden_states.shape[0] - 1
            # only consider layers requested
            if isinstance(layers, int):
                if layers >= hidden_states.shape[0]:
                    raise ValueError(
                        f"Requested layer ({layers}) is out of range: only have "
                        f"{hidden_states.shape[0]} number of hidden layers."
                    )
                if layers < 0:
                    raise ValueError(
                        f"Requested layer ({layers}) is out of range: "
                        "must be greater than or equal to 0, and we only have "
                        f"{hidden_states.shape[0]} number of hidden layers."
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
                    (layer < 0) or (layer >= hidden_states.shape[0]) for layer in layers
                ):
                    raise ValueError(
                        f"Requested layers ({layers}) is out of range: only have "
                        f"{hidden_states.shape[0]} number of hidden layers."
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
                (layer < 0) or (layer >= hidden_states.shape[0]) for layer in layers
            ):
                raise ValueError(
                    f"Requested layers ({layers}) is out of range: only have "
                    f"{hidden_states.shape[0]} number of hidden layers."
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
                f"Method '{method}' for pooling hidden layers "
                "to obtain token embeddings has not been implemented."
            )
        if concatenate_embeddings:
            token_embeddings = torch.cat(token_embeddings).numpy()
        return token_embeddings

    def obtain_embeddings(
        self,
        method: str = "hidden_layer",
        batch_size: int = 100,
        layers: int | list[int] | tuple[int] | None = None,
    ) -> np.array | list[np.array]:
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
        batch_size: int = 100, optional
            The size of the batches, by default 100.
        layer : int | list[int] | tuple[int] | None, optional
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
        # check model, tokenizer and data_collator have been passed into the class
        self._check_model()
        if self.tokens is None:
            raise ValueError(
                "Text has not been tokenized yet. Call `.tokenize_text()` first."
            )
        if (layers is not None) and (not isinstance(layers, (int, list, tuple))):
            raise ValueError(
                "Layers requested must be either integer or list of integers."
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
        # check model, tokenizer and data_collator have been passed into the class
        self._check_model()
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

    def split_dataset(
        self,
        train_size: float = 0.8,
        valid_size: float | None = 0.33,
        indices: tuple[list[int], list[int], list[int]] | None = None,
        shuffle: bool = False,
        random_state: int = 42,
    ) -> DatasetDict:
        """
        Split up dataset into train, validation, test sets for training / fine-tuning.

        Parameters
        ----------
        train_size : float, optional
            How to split the initial dataset into train, test/validation, by default 0.8.
            Ignored if indices are passed.
        valid_size : float | None, optional
            Proportion of training data to use as validation data, by default 0.33.
            If None, will not create a validation set.
            Ignored if indices are passed.
        indices : tuple[list[int], list[int] | None, list[int]] | None, optional
            Train, validation, test indices to use. If passed, will split the data
            according to these indices rather than splitting it within the method
            using the train_size and valid_size provided.
            First item in the tuple should be the indices for the training set,
            second item should be the indices for the validaton set (this could
            be None if no validation set is required), and third item should be
            indices for the test set.
        shuffle : bool, optional
            Whether or not to shuffle the dataset, by default False.
        random_state : int, optional
            Seed number, by default 42.

        Returns
        -------
        DatasetDict
            A dictionary of Datasets with training (`train`), validation (`valid`)
            (if `valid_size` is not None), and test (`test`) Datasets.
        """
        if self.dataset_split is not None:
            print(
                "[INFO] Dataset has already been split. "
                "If required to split again, first set `.dataset_split` attribute to None"
            )
            return self.dataset_split

        if indices is not None:
            # indices are provided, so use these to split the dataset
            if not isinstance(indices, tuple):
                msg = "if indices are provided, it must be a tuple of length 3"
                raise TypeError(msg)
            if len(indices) != 3:
                raise ValueError(msg)

            self.dataset_split = DatasetDict(
                {
                    "train": Dataset.from_dict(self.dataset[indices[0]]),
                    "test": Dataset.from_dict(self.dataset[indices[2]]),
                    "validation": Dataset.from_dict(self.dataset[indices[1]])
                    if indices[1] is not None
                    else None,
                }
            )
        else:
            # indices are not provided, so split the dataset
            if valid_size is None:
                print(
                    "[INFO] Splitting up dataset into train / test sets, "
                    "and saving to `.dataset_split`."
                )
            else:
                print(
                    "[INFO] Splitting up dataset into train / validation / test sets, "
                    "and saving to `.dataset_split`."
                )

            # first split data into train/valid set, test set
            train_test = self.dataset.train_test_split(
                train_size=train_size,
                shuffle=shuffle,
                seed=random_state,
            )

            if valid_size is not None:
                # further split the test set into a test, valid set
                test_valid = train_test["train"].train_test_split(
                    test_size=valid_size,
                    shuffle=shuffle,
                    seed=random_state,
                )
                # gather datasetes together
                self.dataset_split = DatasetDict(
                    {
                        "train": test_valid["train"],
                        "test": train_test["test"],
                        "validation": test_valid["test"],
                    }
                )
            else:
                self.dataset_split = DatasetDict(
                    {
                        "train": train_test["train"],
                        "test": train_test["test"],
                        "validation": None,
                    }
                )

        return self.dataset_split

    def set_up_training_args(self, output_dir: str, **kwargs) -> TrainingArguments:
        """
        Set up `TrainingArguments` object and save to `.trainer` attribute.

        Parameters
        ----------
        output_dir : str
            The output directory where the model predictions and checkpoints will be written.
        **kwargs :
            Passed along to `TrainingArguments()` class.

        Returns
        -------
        TrainingArguments
            `TrainingArguments` object.
        """
        print(
            "[INFO] Setting up TrainingArguments object and saving to `.training_args`."
        )
        if kwargs is None:
            kwargs = {}
        if "evaluation_strategy" not in kwargs:
            kwargs["evaluation_strategy"] = "epoch"

        # initialise TrainingArguments object
        self.training_args = TrainingArguments(output_dir=output_dir, **kwargs)

        return self.training_args

    def set_up_trainer(
        self,
        data_collator: DataCollator | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        **kwargs,
    ) -> Trainer:
        """
        Set up `Trainer` object and save to `.trainer` attribute.

        Parameters
        ----------
        data_collator : DataCollator | None, optional
            The function to use to form a batch from a list of elements
            of `train_dataset` or `eval_dataset`, to pass into `Trainer()`,
            by default None.
        compute_metrics : Callable[[EvalPrediction], dict] | None, optional
            The function that will be used to compute metrics at evaluation.
            Must take a `EvalPrediction` object and return a dictionary
            string to metric values, by default None.
        **kwargs :
            Passed along to `Trainer()` class, by default None.

        Returns
        -------
        Trainer
            `Trainer` object.
        """
        # check model, tokenizer and data_collator have been passed into the class
        self._check_model()

        print("[INFO] Setting up Trainer object, and saving to `.trainer`.")
        if self.training_args is None:
            raise NotImplementedError(
                "TrainingArgments have not been set in `.training_args`. "
                "Call `.set_up_training_args()` first."
            )
        if self.dataset_split is None:
            raise ValueError(
                "Dataset has not been split up into train / test (and validation) sets. "
                "Call `.split_dataset()` first."
            )
        if data_collator is None:
            # use the existing data collator
            data_collator = self.data_collator

        # initialise Trainer object
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset_split["train"],
            eval_dataset=self.dataset_split["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            **kwargs,
        )

        return self.trainer

    def fit_transformer_with_trainer_api(
        self,
        output_dir: str | None = None,
        data_collator: DataCollator | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        training_args: dict | None = None,
        trainer_args: dict | None = None,
    ):
        """
        Train / fine-tune transformer model to some task.

        If the dataset hasn't been split up, or either of the training arguments
        or trainer hasn't been set up, can pass in arguments to do that here.
        Otherwise uses the split dataset, training arguments and trainer saved in
        `.dataset_split`, `.training_args` and `.trainer`, respectively.

        Parameters
        ----------
        output_dir : str
            The output directory where the model predictions
            and checkpoints will be written, by default None.
        data_collator : DataCollator | None, optional
            The function to use to form a batch from a list of elements
            of `train_dataset` or `eval_dataset`, to pass into `Trainer()`,
            by default None.
        compute_metrics : Callable[[EvalPrediction], dict] | None, optional
            The function that will be used to compute metrics at evaluation.
            Must take a `EvalPrediction` object and return a dictionary
            string to metric values, by default None.
        training_args : dict | None, optional
            Passed along to `TrainingArguments()` class, by default None.
        trainer_args : dict | None, optional
            Passed along to `Trainer()` class, by default None.
        """
        if self.dataset_split is None:
            # split up dataset if it hasn't been done yet
            self.split_dataset()
        if self.training_args is None:
            if output_dir is None:
                output_dir = self.model_name
            self.set_up_training_args(output_dir=output_dir, **training_args)
        if self.trainer is None:
            self.set_up_trainer(
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                **trainer_args,
            )

        print(f"[INFO] Training model with {self.model.num_parameters()} parameters...")
        self.trainer.train()
        print("[INFO] Training completed!")
