from umap import UMAP
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class KMeansClustering:
    def __init__(self, num_clusters, n_init):
        self.num_clusters = num_clusters
        self.n_init = n_init
    def kmeans(self, data):
        self.kmean = KMeans (n_clusters=self.num_clusters, init='k-means++', max_iter=300, n_init=self.n_init, random_state=0)
        self.data = pd.DataFrame(data)
        self.y_kmeans = self.kmean.fit_predict(self.data)
        return  self.y_kmeans
    