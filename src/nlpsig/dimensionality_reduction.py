from __future__ import annotations

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)


class DimReduce:
    """
    Class to perform dimension reduction on word or sentence embeddings
    """

    def __init__(
        self,
        method: str = "ppapca",
        n_components: int = 5,
        dim_reduction_kwargs: dict | None = None,
    ) -> None:
        """
        Class to perform dimension reduction on word or sentence embeddings

        Parameters
        ----------
        method : str, optional
            Which dimensionality reduction technique to use, by default "ppapca"
            Options are
            - "umap" (UMAP): implemented using `umap-learn` package
            - "pca" (PCA): implemented using `scikit-learn`
            - "tsne" (TSNE): implemented using `scikit-learn`
            - "gaussian_random_projection" (Gaussian random projection): implemented using `scikit-learn`
            - "sparse_random_projection" (sparse random projection): implemented using `scikit-learn`
            - "ppapca" (Post Processing Algorithm (PPA) with PCA)
              (see Mu, J., Bhat, S., and Viswanath, P. (2017). All-but-the-top:
              Simple and effective postprocessing for word representations.
              arXiv preprint arXiv:1702.01417.)
            - "ppapacppa" (PPA-PCA-PPA)
              (see Raunak, V., Gupta, V., and Metze, F. (2019). Effective dimensionality
              reduction for word embeddings. In Proceedings of the 4th Workshop on
              Representation Learning for NLP (RepL4NLP- 2019), pages 235-243.)
        n_components : int, optional
            Number of n_components to keep, by default 5
        dim_reduction_kwargs : dict | None
            Any keywords to be passed into the functions which perform the
            dimensionality reduction, by default None
        """
        self.method = method
        self.n_components = n_components
        self.kwargs = dim_reduction_kwargs
        self.reducer = None
        self.embedding = None

    def fit_transform(self, embeddings: np.array, random_state: int = 42) -> np.array:
        """
        Fit embeddings into an embedded space and return that transformed output

        Parameters
        ----------
        embeddings : np.array
            Word or sentence embeddings which we wish to reduce the dimensions of
        random_state : int, optional
            Seed number, by default 42

        Returns
        -------
        np.array
            Dimension reduced embeddings in transformed space.

        Raises
        ------
        NotImplementedError
            if `method` attribute of the class is not one of the implemented methods
            Options are
            - "umap" (UMAP): implemented using `umap-learn` package
            - "pca" (PCA): implemented using `scikit-learn`
            - "tsne" (TSNE): implemented using `scikit-learn`
            - "gaussian_random_projection" (Gaussian random projection): implemented using `scikit-learn`
            - "sparse_random_projection" (sparse random projection): implemented using `scikit-learn`
            - "ppapca" (Post Processing Algorithm (PPA) with PCA)
            - "ppapcappa" (PPA-PCA-PPA)
        """
        implemented_methods = [
            "pca",
            "umap",
            "tsne",
            "gaussian_random_projection",
            "sparse_random_projection",
            "ppapca",
            "ppapcappa",
        ]
        if self.method in implemented_methods:
            if self.method == "umap":
                if self.kwargs is None:
                    self.kwargs = {}
                self.reducer = umap.UMAP(
                    n_components=self.n_components,
                    random_state=random_state,
                    transform_seed=random_state,
                    **self.kwargs,
                )
                self.reducer.fit_transform(embeddings)
                self.embedding = self.reducer.embedding_
            elif self.method == "pca":
                if self.kwargs is None:
                    self.kwargs = {}
                self.reducer = PCA(n_components=self.n_components, **self.kwargs)
                self.embedding = self.reducer.fit_transform(embeddings)
            elif self.method == "tsne":
                if self.kwargs is None:
                    self.kwargs = {"learning_rate": "auto"}
                self.reducer = TSNE(
                    n_components=self.n_components,
                    random_state=random_state,
                    **self.kwargs,
                )
                self.reducer.fit_transform(embeddings)
                self.embedding = self.reducer.embedding_
            elif self.method == "gaussian_random_projection":
                if self.kwargs is None:
                    self.kwargs = {}
                self.reducer = GaussianRandomProjection(
                    n_components=self.n_components,
                    random_state=random_state,
                    **self.kwargs,
                )
                self.embedding = self.reducer.fit_transform(embeddings)
            elif self.method == "sparse_random_projection":
                if self.kwargs is None:
                    self.kwargs = {}
                self.reducer = SparseRandomProjection(
                    n_components=self.n_components,
                    random_state=random_state,
                    **self.kwargs,
                )
                self.embedding = self.reducer.fit_transform(embeddings)
            elif self.method == "ppapca":
                self.embedding = self.ppa_pca(
                    embeddings,
                    n_components=self.n_components,
                    dim=3,
                    extra_ppa=False,
                )
            elif self.method == "ppapcappa":
                self.embedding = self.ppa_pca(
                    embeddings,
                    n_components=self.n_components,
                    dim=3,
                    extra_ppa=True,
                )
        else:
            raise NotImplementedError(
                f"{self.method} is not implemented. "
                f"Try one of the following: "
                f"{', '.join(implemented_methods)}"
            )
        return self.embedding

    def ppa_pca(
        self,
        embeddings: np.array,
        n_components: int = 5,
        pca_dim: int = 50,
        dim: int = 3,
        extra_ppa: bool = False,
    ) -> np.array:
        """
        Post Processing Algorithm with PCA (with option to apply PPA again)

        Parameters
        ----------
        embeddings : np.array
            Word or sentence embeddings which we wish to reduce the dimensions of
        n_components : int, optional
            Number of n_components to keep, by default 5
        pca_dim: int, optional
            Number of components for PCA algorithm
            (must be greater than n_components), by default 50
        dim : int, optional
            Threshold parameter D in Post Processing Algorithm
            (must be smaller than n_components), by default 3
        extra_ppa : bool, optional
            Whether or not to apply PPA again, by default False

        Returns
        -------
        np.array
            Dimension reduced embeddings in transformed space.

        Raises
        ------
        ValueError
            if n_components is less than dim, or if n_components is greater than pca_dim
        """
        if n_components < dim:
            raise ValueError("n_components must be greater than or equal to dim")
        if n_components > pca_dim:
            raise ValueError("n_components must be less than or equal to pca_dim")

        # PPA NO 1
        # Subtract mean embedding
        embeddings = embeddings - np.mean(embeddings)
        # Compute PCA Components
        pca = PCA(n_components=embeddings.shape[1])
        embeddings_fit = pca.fit_transform(embeddings)
        U1 = pca.components_
        # Remove top-D components
        z = []
        for x in embeddings:
            x_tmp = x
            for u in U1[0:dim]:
                x_tmp = x_tmp - np.dot(u.transpose(), x_tmp) * u
            z.append(x_tmp)
        z = np.asarray(z).astype(np.float32)

        # Main PCA
        pca = PCA(n_components=pca_dim)
        embeddings_pca = z - np.mean(z)
        embeddings_fit = pca.fit_transform(embeddings_pca)
        embs_reduced = embeddings_fit[:, :n_components]

        if extra_ppa:
            # PPA NO 2
            # Subtract mean embedding
            embeddings_fit = embeddings_fit - np.mean(embeddings_fit)
            # Compute PCA Components
            pca = PCA(n_components=pca_dim)
            pca.fit_transform(embeddings_fit)
            U2 = pca.components_
            # Remove top-D components
            z_new = []
            for x in embeddings_fit:
                x_tmp = x
                for u in U2[1:dim]:
                    x_tmp = x_tmp - np.dot(u.transpose(), x_tmp) * u
                z_new.append(x_tmp)
            embs_reduced = z[:, :n_components]

        return embs_reduced
