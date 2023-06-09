# nlpsig

[![Actions Status][actions-badge]][actions-link]

<!-- [![Documentation Status][rtd-badge]][rtd-link] -->

[![PyPI version][pypi-version]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- [![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![Gitter][gitter-badge]][gitter-link] -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/datasig-ac-uk/nlpsig/workflows/CI/badge.svg
[actions-link]:             https://github.com/datasig-ac-uk/nlpsig/actions
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

## Installation

In the root directory of this repository, perform a verbose, editable install
with pip into a new virtual environment. For example using `conda`:

```bash
git clone git@github.com:datasig-ac-uk/nlpsig.git
cd nlpsig
conda create -n nlpsig
conda activate nlpsig
pip install -v -e .
```

Or with `venv`:

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install -v -e .
```

- For using within Jupyter, you can create a kernel with:

```bash
python -m ipykernel install --user --name nlpsig
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information on running the test
suite using `nox`.
