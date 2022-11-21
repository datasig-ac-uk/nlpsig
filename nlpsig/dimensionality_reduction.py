from typing import Optional

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DimReduce:
    """
    Class to perform dimension reduction on word or sentence embeddings
    """

    def __init__(
        self,
        method: str = "ppapca",
        components: int = 5,
        dim_reduction_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Class to perform dimension reduction on word or sentence embeddings

        Parameters
        ----------
        method : str, optional
            Which dimensionality reduction technique to use, by default "ppapca"
            Options are
            - "pca" (PCA): implented using scikit-learn
            - "umap" (UMAP): implemented using `umap-learn` package
            - "tsne" (TSNE): implemented using scikit-learn
            - "ppapca" (Post Processing Algorithm (PPA) with PCA)
            - "ppapacppa" (PPA-PCA-PPA)
        components : int, optional
            Number of components to keep, by default 5
        dim_reduction_kwargs : Optional[dict], optional
            Any keywords to be passed into the functions which perform the
            dimensionality reduction, by default None
        """
        self.method = method
        self.components = components
        self.kwargs = dim_reduction_kwargs
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
            - "pca" (PCA): implented using scikit-learn
            - "umap" (UMAP): implemented using `umap-learn` package
            - "tsne" (TSNE): implemented using scikit-learn
            - "ppapca" (Post Processing Algorithm (PPA) with PCA)
            - "ppapacppa" (PPA-PCA-PPA)
        """
        implemented_methods = ["pca", "umap", "tsne", "ppapca", "ppapcappa"]
        if self.method in implemented_methods:
            if self.method == "pca":
                if self.kwargs is None:
                    self.kwargs = {}
                pca = PCA(n_components=self.components, **self.kwargs)
                self.embedding = pca.fit_transform(embeddings)
            elif self.method == "umap":
                if self.kwargs is None:
                    self.kwargs = {
                        "min_dist": 0.99,
                        "n_neighbors": 50,
                        "metric": "cosine",
                    }
                reducer = umap.UMAP(
                    n_components=self.components,
                    random_state=random_state,
                    transform_seed=random_state,
                    **self.kwargs,
                )
                reducer.fit_transform(embeddings)
                self.embedding = reducer.embedding_
            elif self.method == "tsne":
                if self.kwargs is None:
                    self.kwargs = {}
                tsne = TSNE(
                    n_components=self.components,
                    learning_rate="auto",
                    random_state=random_state,
                    **self.kwargs,
                )
                tsne.fit_transform(embeddings)
                self.embedding = tsne.embedding_
            elif self.method == "ppapca":
                self.embedding = self.ppa_pca(
                    embeddings,
                    components=self.components,
                    dim=3,
                    extra_ppa=False,
                )
            elif self.method == "ppapcappa":
                self.embedding = self.ppa_pca(
                    embeddings,
                    components=self.components,
                    dim=3,
                    extra_ppa=True,
                )
        else:
            raise NotImplementedError(
                f"{self.method} is not implemented. "
                f"Try one of the following: " + f"{', '.join(implemented_methods)}."
            )
        return self.embedding

    def ppa_pca(
        self,
        embeddings: np.array,
        components: int = 5,
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
        components : int, optional
            Number of components to keep, by default 5
        pca_dim: int, optional
            Number of components for PCA algorithm
            (must be greater than components), by default 50
        dim : int, optional
            Threshold parameter D in Post Processing Algorithm
            (must be smaller than components), by default 3
        extra_ppa : bool, optional
            Whether or not to apply PPA again, by default False

        Returns
        -------
        np.array
            Dimension reduced embeddings in transformed space.

        Raises
        ------
        ValueError
            if components is less than dim, or if components is greater than pca_dim
        """
        if components < dim:
            raise ValueError("components must be greater than or equal to dim")
        elif components > pca_dim:
            raise ValueError("components must be less than or equal to pca_dim")

        # PPA NO 1
        # Subtract mean embedding
        embeddings = embeddings - np.mean(embeddings)
        # Compute PCA Components
        pca = PCA(n_components=embeddings.shape[1])
        embeddings_fit = pca.fit_transform(embeddings)
        U1 = pca.components_
        # Remove top-D components
        z = []
        for i, x in enumerate(embeddings):
            for u in U1[0:dim]:
                x = x - np.dot(u.transpose(), x) * u
            z.append(x)
        z = np.asarray(z).astype(np.float32)

        # Main PCA
        pca = PCA(n_components=pca_dim)
        embeddings_pca = z - np.mean(z)
        embeddings_fit = pca.fit_transform(embeddings_pca)
        embs_reduced = embeddings_fit[:, :components]

        if extra_ppa:
            # PPA NO 2
            # Subtract mean embedding
            embeddings_fit = embeddings_fit - np.mean(embeddings_fit)
            # Compute PCA Components
            pca = PCA(n_components=pca_dim)
            # embeddings_fit_2 = pca.fit_transform(embeddings_fit)
            U2 = pca.components_
            # Remove top-D components
            z_new = []
            for i, x in enumerate(embeddings_fit):
                for u in U2[1:dim]:
                    x = x - np.dot(u.transpose(), x) * u
                z_new.append(x)
            embs_reduced = z[:, :components]

        return embs_reduced
