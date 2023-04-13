from __future__ import annotations

__version__ = "0.1.0"

from .classification_utils import Folds, set_seed
from .data_preparation import PrepareData
from .dimensionality_reduction import DimReduce
from .encode_text import SentenceEncoder, TextEncoder
from .plot_embedding import PlotEmbedding

__all__ = (
    "__version__",
    "Folds",
    "set_seed",
    "PrepareData",
    "DimReduce",
    "SentenceEncoder",
    "TextEncoder",
    "PlotEmbedding",
)
