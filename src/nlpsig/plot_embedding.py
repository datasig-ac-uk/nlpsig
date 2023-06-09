from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from distinctipy import colorsets


class PlotEmbedding:
    """
    Class to visualise word or sentence embeddings
    """

    def __init__(self, x_data: np.array, y_data: np.array) -> None:
        """
        Class to visualise word or sentence embeddings.

        Parameters
        ----------
        x_data : np.array
            features
        y_data : np.array
            y labels
        """
        self.x_data = x_data
        self.y_data = y_data
        self.embed = {}

    def plt_2d(
        self,
        embed_args: dict | None = None,
        line_args: dict | None = None,
    ) -> None:
        """
        Plots the embedding in 2d space after first performing dimension reduction.

        Parameters
        ----------
        embed_args : dict | None, optional
            Any keywords to be passed into the functions which perform
            the dimensionality reduction, by default {"method": "pca", "dim": 2}.
        line_args : dict | None, optional
            Any keywords to be passed into the functions which plots the embeddings
            (arguments for `matplotlib.pyplot.scatter()`),
            by default {"marker": "o", "alpha": 0.3}.
        """
        if embed_args is None:
            embed_args = {"method": "pca", "dim": 2}
        if line_args is None:
            line_args = {"marker": "o", "alpha": 0.3}
        assert embed_args["dim"] >= 2, "dim in embed_args should be >= 2"
        self.embedding_dim_reduce(**embed_args)
        colors = colorsets.get_colors()[0 : len(np.unique(self.y_data.astype(int)))]
        colors_plt = [colors[i] for i in self.y_data]
        embed_name = f"{embed_args['method']}_{embed_args['dim']}"
        plt.scatter(
            self.embed[embed_name][:, 0],
            self.embed[embed_name][:, 1],
            color=colors_plt,
            **line_args,
        )
        plt.grid(zorder=-10)
        plt.show()

    def embedding_dim_reduce(
        self,
        method: str = "pca",
        dim: int = 3,
        overwrite: bool = False,
        random_state: int = 42,
    ) -> None:
        """
        Performs dimension reduction to the data and adds reduced embeddings to .embed.

        Parameters
        ----------
        method : str, optional
            Which dimensionality reduction technique to use, by default "pca".
            Options:
            - "pca" (PCA): implemented using scikit-learn
            - "umap" (UMAP): implemented using `umap-learn` package
            - "tsne" (TSNE): implemented using scikit-learn
        dim : int, optional
            Number of components to keep, by default 3.
        overwrite : bool, optional
            Whether or not to overwrite current implemented embedding, by default False.
        random_state : int, optional
            Seed number, by default 42.

        Raises
        ------
        NotImplementedError
            if `method` is not one of the implemented methods
            Options are
            - "pca" (PCA): implemented using scikit-learn
            - "umap" (UMAP): implemented using `umap-learn` package
            - "tsne" (TSNE): implemented using scikit-learn
        """
        embed_name = f"{method}_{dim}"
        if (not overwrite) and (embed_name in self.embed):
            print(
                f"[INFO] {method} with dim={dim} is already calculated. "
                "You can set overwrite = True to recompute"
            )
            return
        implemented_methods = ["pca", "umap", "tsne"]
        if method in implemented_methods:
            from sklearn.preprocessing import StandardScaler

            self.x_data_std = StandardScaler().fit_transform(self.x_data)
            if method == "pca":
                from sklearn.decomposition import PCA

                pca = PCA(n_components=dim)
                self.embed[embed_name] = pca.fit_transform(self.x_data_std)
            elif method == "umap":
                import umap

                reducer = umap.UMAP(
                    n_components=dim,
                    random_state=random_state,
                    transform_seed=random_state,
                )
                reducer.fit_transform(self.x_data_std)
                self.embed[embed_name] = reducer.embedding_
            elif method == "tsne":
                from sklearn.manifold import TSNE

                tsne = TSNE(
                    n_components=dim, learning_rate="auto", random_state=random_state
                )
                tsne.fit_transform(self.x_data_std)
                self.embed[embed_name] = tsne.embedding_
        else:
            raise NotImplementedError(
                f"{method} to reduce dimensions of embeddings "
                "is not implemented. Try one of the following: "
                f"{', '.join(implemented_methods)}"
            )
