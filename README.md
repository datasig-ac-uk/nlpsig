# nlpsig

[![Actions Status][actions-badge]][actions-link]
[![Codecov Status][codecov-badge]][codecov-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

<!-- [![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![Gitter][gitter-badge]][gitter-link] -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/datasig-ac-uk/nlpsig/workflows/CI/badge.svg
[actions-link]:             https://github.com/datasig-ac-uk/nlpsig/actions
[codecov-badge]:            https://codecov.io/gh/datasig-ac-uk/nlpsig/branch/main/graph/badge.svg?token=SU9HZ9NH70
[codecov-link]:             https://codecov.io/gh/datasig-ac-uk/nlpsig
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/nlpsig
[conda-link]:               https://github.com/conda-forge/nlpsig-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/datasig-ac-uk/nlpsig/discussions
[gitter-badge]:             https://badges.gitter.im/https://github.com/datasig-ac-uk/nlpsig/community.svg
[gitter-link]:              https://gitter.im/https://github.com/datasig-ac-uk/nlpsig/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[pypi-link]:                https://pypi.org/project/nlpsig/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/nlpsig
[pypi-version]:             https://img.shields.io/pypi/v/nlpsig
[rtd-badge]:                https://readthedocs.org/projects/nlpsig/badge/?version=latest
[rtd-link]:                 https://nlpsig.readthedocs.io/en/latest/?badge=latest
<!-- prettier-ignore-end -->

NLPSig (`nlpsig`) is a Python package for constructing streams/paths of
embeddings obtained from transformers. The key contributions are:

- A simple API for taking streams of textual data and constructing streams of
  embeddings from transformers
  - The
    [`nlpsig.SentenceEncoder`](https://nlpsig.readthedocs.io/en/latest/encode_text.html#nlpsig.encode_text.SentenceEncoder)
    and
    [`nlpsig.TextEncoder`](https://nlpsig.readthedocs.io/en/latest/encode_text.html#nlpsig.encode_text.TextEncoder
    classes allow you to pass in a corpus of text data (in a variety of formats)
    and obtain corresponding embeddings using the
    [`sentence-transformer`](https://github.com/UKPLab/sentence-transformers)
    and HuggingFace
    [`transformers`](https://github.com/huggingface/transformers) libraries,
    respectively.
  - The
    [`nlpsig.PrepareData`](https://nlpsig.readthedocs.io/en/latest/data_preparation.html)
    allows you to easily construct paths/streams of embeddings which can be used
    for several downstream tasks.
- Simple API for performing dimensionality reduction on the embeddings obtained
  from transformers by some simple wrappers over popular dimensionality
  reduction algorithms such as PCA, UMAP, t-SNE, etc.
  - This is particularly useful if we wish to use path signatures in any
    downstream model since the dimensionality of the embeddings obtained from
    transformers is usually very high.
  - We present some _Signature Network_ models for longitudinal NLP tasks in the
    [`sig-networks`](https://github.com/ttseriotou/sig-networks) library which
    uses these paths constructed in this library as inputs to neural networks
    which utilise path signature methodology.
- We also have simple classes for constructing train/test splits of the data and
  for K-fold cross-validation which are specific for the Signature Networks in
  the [`sig-networks`](https://github.com/ttseriotou/sig-networks) library.

## Installation

NLPSig is available on PyPI and can be installed with `pip`:

```
pip install nlpsig
```

## Contributing

To take advantage of `pre-commit`, which will automatically format your code and
run some basic checks before you commit:

```
pip install pre-commit  # or brew install pre-commit on macOS
pre-commit install  # will install a pre-commit hook into the git repo
```

After doing this, each time you commit, some linters will be applied to format
the codebase. You can also/alternatively run `pre-commit run --all-files` to run
the checks.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information on running the test
suite using `nox`.
