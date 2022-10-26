import matplotlib.pyplot as plt
import numpy as np

from distinctipy import colorsets


class plotEmbedding:
    def __init__(self, 
                 x_data: np.array,
                 y_data: np.array) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.embed = {}
    
    def plt_2d(self, 
               embed_args: dict = {
                   "method": "pca",
                   "dim": 2
               },
               line_args: dict = {
                   "marker": "o",
                   "alpha": 0.3
               }) -> None:
        """
        Plots the embedding in 2d space after first performing dimension reduction
        """
        assert embed_args["dim"] >= 2, "dim in embed_args should be >= 2"
        self.embedding_dim_reduce(**embed_args)
        colors = colorsets.get_colors()[0:len(np.unique(self.y_data.astype(int)))]
        colors_plt = [colors[i] for i in self.y_data]
        embed_name = f"{embed_args['method']}_{embed_args['dim']}"
        plt.scatter(self.embed[embed_name][:, 0], 
                    self.embed[embed_name][:, 1],
                    color=colors_plt,
                    **line_args)
        plt.grid(zorder=-10)
        plt.show()
    
    def embedding_dim_reduce(self,
                             method: str = "pca",
                             dim: int = 3,
                             overwrite: bool = False,
                             random_state: int = 42) -> None:
        """
        Performs dimension reduction to the data and adds reduced embeddings to .embed
        """
        embed_name = f"{method}_{dim}"
        if (not overwrite) and (embed_name in self.embed):
            print(f"[INFO] {method} with dim={dim} is already calculated. " +
                  "You can set overwrite = True to recompute.")
            return
        implemented_methods = ["pca", "umap", "tsne"]
        if method in implemented_methods:
            from sklearn.preprocessing import StandardScaler
            self.x_data_std = StandardScaler().fit_transform(self.x_data)
            if method == "pca":
                from sklearn.decomposition import PCA
                pca = PCA(n_components = dim)
                self.embed[embed_name] = pca.fit_transform(self.x_data_std)
            elif method == "umap":
                import umap
                reducer = umap.UMAP(n_components = dim,
                                    random_state = random_state,
                                    transform_seed = random_state)
                reducer.fit_transform(self.x_data_std)
                self.embed[embed_name] = reducer.embedding_
            elif method == "tsne":
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components = dim,
                            learning_rate = 'auto',
                            random_state = random_state)
                tsne.fit_transform(self.x_data_std)
                self.embed[embed_name] = tsne.embedding_
        else:
            raise NotImplementedError(f"{method} to reduce dimensions of embeddings " +
                                      "is not implemented. Try one of the following: " + 
                                      f"{', '.join(implemented_methods)}.")
                
