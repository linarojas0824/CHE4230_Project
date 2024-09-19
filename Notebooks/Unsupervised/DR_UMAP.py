from umap import UMAP
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DR_UMAP:
    def __init__(self, n_components):
        self.n_components = n_components
    def dt(self, data):
        self.Norm =MinMaxScaler()
        self.DR_UMAP = UMAP (n_components= self.n_components)
        self.normalize = pd.DataFrame(self.Norm.fit_transform(data))
        self.DR_UMAP_R = self.DR_UMAP.fit_transform(self.normalize)
        return self.DR_UMAP_R
    