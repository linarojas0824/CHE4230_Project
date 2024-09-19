from Data_Preprocess import DataPreprocessing
from DR_UMAP import DR_UMAP
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from KMeansClustering import KMeansClustering
import matplotlib.pyplot as plt

class ModelDataProcessor:
    
    def __init__(self, data_path):
        self.preprocessor = DataPreprocessing()
        self.data = self.preprocessor.load_data(data_path)
        self.train_validation, self.test = train_test_split(self.data, train_size=0.7, test_size=0.3, random_state=60)
        self.train, self.validation = train_test_split(self.train_validation,train_size=0.50, test_size=0.50, random_state=60)
        self.train_data = None  # initialize to None   

    def apply_dataSplit(self):
        data = self.data
        train_validation = self.train_validation
        train = self.train
        validation = self.validation
        test = self.test
        return data, train_validation, train, validation, test
    
    def apply_umap(self, n_components):
        drUMAP = DR_UMAP(n_components=n_components)
        self.train_data = drUMAP.dt(self.train)
        self.validation_data = drUMAP.dt(self.validation)
        self.test_data = drUMAP.dt(self.test)
        return self.train_data,self.validation_data,self.test_data
    
    def apply_kmeans(self,num_clusters, n_init):
        if self.train_data is None:
            raise ValueError("train_data is not defined. Please call apply_umap first.")
        
        kmeans = KMeansClustering (num_clusters=num_clusters, n_init= n_init)
        self.labels_train = kmeans.fit_predict(self.train_data)
        self.labels_validation = kmeans.fit_predict(self.validation_data)
        self.labels_test = kmeans.fit_predict(self.test_data)
        return self.labels_train, self.labels_validation,self.labels_test 
    
    def apply_prot(self, x, y, c, cmap):
        plt.scatter(x=self.train_data[:, 0],  y = self.train_data[:, 1], c=self.labels_train, cmap='viridis')
