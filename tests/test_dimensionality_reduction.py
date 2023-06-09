from __future__ import annotations

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)

from nlpsig.dimensionality_reduction import DimReduce


def test_umap(X_fit, X_new):
    n_comp = 25
    reduction = DimReduce(method="umap", n_components=n_comp)
    assert reduction.method == "umap"
    assert reduction.n_components == n_comp
    assert reduction.kwargs is None
    assert reduction.reducer is None
    assert reduction.embedding is None

    reduced_X_fit = reduction.fit_transform(X_fit)
    assert type(reduction.reducer) is umap.UMAP
    assert (reduction.embedding == reduced_X_fit).all()
    assert reduction.embedding.shape == (X_fit.shape[0], n_comp)

    reduced_X_new = reduction.reducer.transform(X_new)
    assert reduced_X_new.shape == (X_new.shape[0], n_comp)


def test_pca(X_fit, X_new):
    n_comp = 25
    reduction = DimReduce(method="pca", n_components=n_comp)
    assert reduction.method == "pca"
    assert reduction.n_components == n_comp
    assert reduction.kwargs is None
    assert reduction.reducer is None
    assert reduction.embedding is None

    reduced_X_fit = reduction.fit_transform(X_fit)
    assert type(reduction.reducer) is PCA
    assert (reduction.embedding == reduced_X_fit).all()
    assert reduction.embedding.shape == (X_fit.shape[0], n_comp)

    reduced_X_new = reduction.reducer.transform(X_new)
    assert reduced_X_new.shape == (X_new.shape[0], n_comp)


def test_tsne(X_fit):
    n_comp = 3
    reduction = DimReduce(method="tsne", n_components=n_comp)
    assert reduction.method == "tsne"
    assert reduction.n_components == n_comp
    assert reduction.kwargs is None
    assert reduction.reducer is None
    assert reduction.embedding is None

    reduced_X_fit = reduction.fit_transform(X_fit)
    assert type(reduction.reducer) is TSNE
    assert (reduction.embedding == reduced_X_fit).all()
    assert reduction.embedding.shape == (X_fit.shape[0], n_comp)


def test_grp(X_fit, X_new):
    n_comp = 25
    reduction = DimReduce(method="gaussian_random_projection", n_components=n_comp)
    assert reduction.method == "gaussian_random_projection"
    assert reduction.n_components == n_comp
    assert reduction.kwargs is None
    assert reduction.reducer is None
    assert reduction.embedding is None

    reduced_X_fit = reduction.fit_transform(X_fit)
    assert type(reduction.reducer) is GaussianRandomProjection
    assert (reduction.embedding == reduced_X_fit).all()
    assert reduction.embedding.shape == (X_fit.shape[0], n_comp)

    reduced_X_new = reduction.reducer.transform(X_new)
    assert reduced_X_new.shape == (X_new.shape[0], n_comp)


def test_srp(X_fit, X_new):
    n_comp = 25
    reduction = DimReduce(method="sparse_random_projection", n_components=n_comp)
    assert reduction.method == "sparse_random_projection"
    assert reduction.n_components == n_comp
    assert reduction.kwargs is None
    assert reduction.reducer is None
    assert reduction.embedding is None

    reduced_X_fit = reduction.fit_transform(X_fit)
    assert type(reduction.reducer) is SparseRandomProjection
    assert (reduction.embedding == reduced_X_fit).all()
    assert reduction.embedding.shape == (X_fit.shape[0], n_comp)

    reduced_X_new = reduction.reducer.transform(X_new)
    assert reduced_X_new.shape == (X_new.shape[0], n_comp)


def test_ppapca(X_fit):
    n_comp = 25
    reduction = DimReduce(method="ppapca", n_components=n_comp)
    assert reduction.method == "ppapca"
    assert reduction.n_components == n_comp
    assert reduction.kwargs is None
    assert reduction.reducer is None
    assert reduction.embedding is None

    reduced_X_fit = reduction.fit_transform(X_fit)
    assert reduction.reducer is None
    assert (reduction.embedding == reduced_X_fit).all()
    assert reduction.embedding.shape == (X_fit.shape[0], n_comp)


def test_pcappapca(X_fit):
    n_comp = 25
    reduction = DimReduce(method="ppapcappa", n_components=n_comp)
    assert reduction.method == "ppapcappa"
    assert reduction.n_components == n_comp
    assert reduction.kwargs is None
    assert reduction.reducer is None
    assert reduction.embedding is None

    reduced_X_fit = reduction.fit_transform(X_fit)
    assert reduction.reducer is None
    assert (reduction.embedding == reduced_X_fit).all()
    assert reduction.embedding.shape == (X_fit.shape[0], n_comp)
