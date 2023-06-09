from __future__ import annotations

__version__ = "0.1.0"

import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

from .classification_utils import DataSplits, Folds, set_seed  # noqa: E402
from .data_preparation import PrepareData  # noqa: E402
from .dimensionality_reduction import DimReduce  # noqa: E402
from .encode_text import SentenceEncoder, TextEncoder  # noqa: E402
from .plot_embedding import PlotEmbedding  # noqa: E402

__all__ = (
    "__version__",
    "DataSplits",
    "Folds",
    "set_seed",
    "PrepareData",
    "DimReduce",
    "SentenceEncoder",
    "TextEncoder",
    "PlotEmbedding",
)
