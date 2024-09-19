import pandas as pd
import numpy as np


class DataPreprocessing():
    def __init__(self):
        pass

    def load_data(self, path):
        data = pd.read_csv(path).dropna()

        data = data.drop(columns=data.columns[(data == 0).all()])
        
        #substite zeros for mean of the same column
        mean = data.mean()
        data = data.replace(0,np.nan)
        data = data.fillna(mean)

        print(data.head())

        data = data.to_numpy()

        return data