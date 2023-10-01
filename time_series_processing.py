# Time Series Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split


class CombinedCardiacPressure():
    def __init__(self):
        self.modelLearn = False
        self.stats = 0

        
    def _cleanup(self, data, sigma=1):
        # interpolate data
        data = 

        # filter noise (use scipy.ndimage.gaussian_filter1d) 
        # estimate the standard deviation (sigma)
        data = 

        return data

    def model_learn(self):
        # Importing the dataset
        carotid_df = 
        illiac_df = 

        # Set up X input and y target
        y = 
        X_carotid = 
        X_illiac = 


        X_carotid = self._cleanup(X_carotid, )
        X_illiac = self._cleanup(X_illiac, )

  
        # scale data
        
 
        # Combine Carotid and Illiac inputs
        X = 

        # # Splitting the dataset into the Training set and Test set
        # from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

        # # Training the Naive Bayes model on the Training set
        from sklearn.ensemble import RandomForestClassifier
        self.classifier = RandomForestClassifier()
        self.classifier.fit(X_train, y_train)

        # # Predicting the Test set results
        y_pred = self.classifier.predict(X_test)

        # # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        
        self.stats =  accuracy_score(y_test, y_pred)
        self.modelLearn = True

        
    def model_infer(self, time_series_carotid_filename, time_series_illiac_filename):
        if(self.modelLearn != True):
            self.model_learn()


        # Read in filenames 
        carotid_df = 
        illiac_df = 

        # Clean up data using self._cleanup() function
        time_series_carotid = 
        time_series_illiac = 

        # Scale dataset
        time_series_carotid = 
        time_series_illiac = 
        
        # Combine Carotid and Illiac inputs
        dataOne = np.concatenate([time_series_carotid, time_series_illiac], axis=1)
        
        # Use classifier to predict the value
        y_pred = self.classifier.predict(dataOne)
        
        return y_pred



class CarotidPressure():
    def __init__(self):
        pass

        
    def _cleanup(self, data, sigma=1):
        pass

    def model_learn(self):
        pass
        
    def model_infer(self, time_series_carotid_filename):
        pass
    
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)




class IlliacPressure():
    def __init__(self):
        pass

        
    def _cleanup(self, data, sigma=1):
        pass

    def model_learn(self):
        pass
        
    def model_infer(self, time_series_illiac_filename):
        pass
    
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)


if __name__ == '__main__':
        # m = CarotidPressure()
        # m = IlliacPressure()
        m = CombinedCardiacPressure()

        m.model_learn()

        result = m.model_infer(pd.read_csv('data/carotid_pressure_test_1.csv'), pd.read_csv('data/illiac_pressure_test_1.csv'))
        print(result)

        result = m.model_infer(pd.read_csv('data/carotid_pressure_test_2.csv'), pd.read_csv('data/illiac_pressure_test_2.csv'))
        print(result)

        result = m.model_infer(pd.read_csv('data/carotid_pressure_test_3.csv'), pd.read_csv('data/illiac_pressure_test_3.csv'))
        print(result)

        result = m.model_infer(pd.read_csv('data/carotid_pressure_test_4.csv'), pd.read_csv('data/illiac_pressure_test_4.csv'))
        print(result)
            
        print(m.model_stats())

