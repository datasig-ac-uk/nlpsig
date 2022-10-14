import matplotlib.pyplot as plt
import numpy as np

from distinctipy import colorsets


class plotEmbedding:
    def __init__(self, 
                 x_data,
                 y_data):
        
        self.x_data = x_data
        self.y_data = y_data
        self.embed = {}
    
    def plt_2d(self, 
               embed_args={
                   "method": "pca",
                   "dim": 2
               },
               line_args={
                   "marker": "o",
                   "alpha": 0.3
               }
              ):
        
        assert embed_args["dim"] >= 2, "dim in embed_args should be >= 2"
        
        self.calc_embedding(**embed_args)
        colors = colorsets.get_colors()[0:len(np.unique(self.y_data.astype(int)))]
        colors_plt = [colors[i] for i in self.y_data]
        
        embed_name = f"{embed_args['method']}_{embed_args['dim']}"
        plt.scatter(self.embed[embed_name][:, 0], 
                    self.embed[embed_name][:, 1],
                    color=colors_plt,
                    **line_args
                   )
        plt.grid(zorder=-10)
        plt.show()
    
    def calc_embedding(self, method="pca", dim=3, overwrite=False, random_state=42):
        embed_name = f"{method}_{dim}"
        
        if (not overwrite) and (embed_name in self.embed):
            print(f"[INFO] {method} with dim: {dim} is already calculated. You can set overwrite=True to recompute.")
            return
        
        if method == "pca":
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            self.x_data_std = StandardScaler().fit_transform(self.x_data)
            pca = PCA(n_components=dim)
            self.embed[embed_name] = pca.fit_transform(self.x_data_std)
            
        elif method == "umap":
            import umap
            from sklearn.preprocessing import StandardScaler
            
            self.x_data_std = StandardScaler().fit_transform(self.x_data)
            reducer = umap.UMAP(random_state=random_state, n_components=dim)
            embedding = reducer.fit_transform(self.x_data_std)
            self.embed[embed_name] = reducer.embedding_
            
        else:
            raise NotImplementedError
                
