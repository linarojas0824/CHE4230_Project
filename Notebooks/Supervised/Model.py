from sklearn.preprocessing import MinMaxScaler
from CHE4230_Project.Notebooks.Data_Preprocess import DataPreprocessing
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle
import joblib
from umap import UMAP


class Model(DataPreprocessing):
    def __init__(self,*args, **kwargs):
        super(Model,self).__init__(*args, **kwargs)
    def load_labels (self,datapath):
        self.preprocessor = DataPreprocessing()
        self.labels = self.preprocessor.load_data(datapath)
        self.label_map = {3:0,8:1,2:2,13:3,6:4}
        self.y_eval_mapped = np.array([self.label_map[y] for y in self.labels[:,1]])
        return self.y_eval_mapped
    def raw_data (self,datapath2):
        self.preprocessor = DataPreprocessing()
        self.data = self.preprocessor.load_data(datapath2)
        self.Norm =MinMaxScaler()
        self.data_normalized=pd.DataFrame(self.Norm.fit_transform(self.data))
        self.UMAP_normal = UMAP (n_components= 2)
        self.df_UMAP_normal = self.UMAP_normal.fit_transform(self.data_normalized)
        return (self.df_UMAP_normal)
    def predict (self,data):
        with open('CHE4230project_LLGModel/model.pkl', 'rb') as file:
            self.model = pickle.load(file)
        self.y_predict = self.model.predict(data)
        return  self.y_predict
    def eval_accuracy (self,prediction,labels):
        self.accuracy = accuracy_score(labels,prediction)
        return (self.accuracy)