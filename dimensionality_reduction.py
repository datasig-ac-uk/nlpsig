from sklearn.decomposition import PCA
import umap
import numpy as np

class DimensionalityReduction:

    def __init__(self, method='ppapca', components=5):
        self.method = method
        self.components = components

    def fit_transform(self, embeddings_sentence):

        if (self.method == 'umap'):
            #summary of umap hyperparameters
            #n_neighbors: This parameter controls how UMAP balances local versus global structure in the data. It does this by constraining the size of the local neighborhood
            #min_dist: controls how tightly UMAP is allowed to pack points together. It, quite literally, provides the minimum distance apart that points are allowed to be in the low dimensional representation.
            reducer = umap.UMAP(n_components = self.components, min_dist=0.99, n_neighbors=50, metric = 'cosine', random_state=12, transform_seed=12)
            embeddings_reduced = reducer.fit_transform(embeddings_sentence)

        elif (self.method=='ppapca'):
            embeddings_reduced = self.ppa_pca(embeddings_sentence, components=self.components, d=3, extra_ppa=False)
        elif (self.method=='ppapcappa'):
            embeddings_reduced = self.ppa_pca(embeddings_sentence, components=self.components, d=3, extra_ppa=True)
        else:
            print('ERROR: need to define a valid dimensionality reduction method')
        
        return embeddings_reduced


    def ppa_pca(self,embeddings_sentence, components = 5, d = 3, extra_ppa = False):

        #PPA NO 1

        #Subtract mean embedding
        embeddings_sentence = embeddings_sentence - np.mean(embeddings_sentence)

        #Compute PCA Components
        pca =  PCA(n_components = embeddings_sentence.shape[1])
        embeddings_sentence_fit = pca.fit_transform(embeddings_sentence)
        U1 = pca.components_

        #Remove top-D components
        z = []

        for i, x in enumerate(embeddings_sentence):
            for u in U1[0:d]:
                x = x - np.dot(u.transpose(),x) * u
            z.append(x)
        
        z = np.asarray(z).astype(np.float32)

        #Main PCA
        pca =  PCA(n_components = 50)
        embeddings_sentence_pca = z - np.mean(z)
        embeddings_sentence_fit = pca.fit_transform(embeddings_sentence_pca)

        embs_reduced = embeddings_sentence_fit[:,:components]

        if extra_ppa:
            #PPA NO 2

            #Subtract mean embedding
            embeddings_sentence_fit = embeddings_sentence_fit - np.mean(embeddings_sentence_fit)
            
            #Compute PCA Components
            pca = PCA(n_components = 50)
            embeddings_sentence_fit_2 = pca.fit_transform(embeddings_sentence_fit)
            U2 = pca.components_

            #Remove top-D components
            z_new = []

            for i, x in enumerate(embeddings_sentence_fit):
                for u in U2[1:d]:
                    x = x - np.dot(u.transpose(),x) * u
                z_new.append(x)
            
            embs_reduced = z[:,:components]

        return embs_reduced
