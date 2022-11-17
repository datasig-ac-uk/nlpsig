from typing import Optional

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DimReduce:
    def __init__(
        self,
        method: str = "ppapca",
        components: int = 5,
        dim_reduction_kwargs: Optional[dict] = None,
    ) -> None:
        self.method = method
        self.components = components
        self.kwargs = dim_reduction_kwargs
        self.embedding = None

    def fit_transform(
        self, embeddings_sentence: np.array, random_state: int = 42
    ) -> np.array:
        implemented_methods = ["pca", "umap", "tsne", "ppapca", "ppapcappa"]
        if self.method in implemented_methods:
            if self.method == "pca":
                if self.kwargs is None:
                    self.kwargs = {}
                pca = PCA(n_components=self.components, **self.kwargs)
                self.embedding = pca.fit_transform(embeddings_sentence)
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
                reducer.fit_transform(embeddings_sentence)
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
                tsne.fit_transform(embeddings_sentence)
                self.embedding = tsne.embedding_
            elif self.method == "ppapca":
                self.embedding = self.ppa_pca(
                    embeddings_sentence,
                    components=self.components,
                    dim=3,
                    extra_ppa=False,
                )
            elif self.method == "ppapcappa":
                self.embedding = self.ppa_pca(
                    embeddings_sentence,
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
        embeddings_sentence: np.array,
        components: int = 5,
        dim: int = 3,
        extra_ppa: bool = False,
    ) -> np.array:
        # PPA NO 1

        # Subtract mean embedding
        embeddings_sentence = embeddings_sentence - np.mean(embeddings_sentence)

        # Compute PCA Components
        pca = PCA(n_components=embeddings_sentence.shape[1])
        embeddings_sentence_fit = pca.fit_transform(embeddings_sentence)
        U1 = pca.components_

        # Remove top-D components
        z = []
        for i, x in enumerate(embeddings_sentence):
            for u in U1[0:dim]:
                x = x - np.dot(u.transpose(), x) * u
            z.append(x)
        z = np.asarray(z).astype(np.float32)

        # Main PCA
        pca = PCA(n_components=50)
        embeddings_sentence_pca = z - np.mean(z)
        embeddings_sentence_fit = pca.fit_transform(embeddings_sentence_pca)

        embs_reduced = embeddings_sentence_fit[:, :components]

        if extra_ppa:
            # PPA NO 2

            # Subtract mean embedding
            embeddings_sentence_fit = embeddings_sentence_fit - np.mean(
                embeddings_sentence_fit
            )

            # Compute PCA Components
            pca = PCA(n_components=50)
            embeddings_sentence_fit_2 = pca.fit_transform(embeddings_sentence_fit)
            U2 = pca.components_

            # Remove top-D components
            z_new = []

            for i, x in enumerate(embeddings_sentence_fit):
                for u in U2[1:dim]:
                    x = x - np.dot(u.transpose(), x) * u
                z_new.append(x)

            embs_reduced = z[:, :components]

        return embs_reduced
