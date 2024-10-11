
import json
import numpy as np
import pandas as pd
import os
from constants import CONFIG_DIR
from torch import nn

# Import required libraries for various models
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from keras.models import load_model
import torch
from models.pytorch_models import pytorchModel
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MLModel:
    def __init__(self,json_file):
        """Initialize the model wrapper."""
        self.model = None

        
        with open(json_file, 'r') as file:
            self.config = json.load(file)
            
        self.load_model()

    def load_model(self):
        """
        Load a model configuration from a JSON file and instantiate the model.

        Args:
            json_file (str): Path to the JSON file containing model configuration.

        Raises:
            ValueError: If the model type is not recognized.
        """

        model_type = self.config.get("model_type")

        if model_type == "xgboost":
            self.model = xgb.XGBRegressor(**self.config.get("params", {}))
        elif model_type == "sklearn_decision_tree":
            self.model = DecisionTreeRegressor(**self.config.get("params", {}))
        elif model_type == "sklearn_random_forest":
            self.model = RandomForestRegressor(**self.config.get("params", {}))
        elif model_type == "transformer":
            self.model = pytorchModel(model_type = "transformer",**self.config.get("params", {}))
        elif model_type == "pytorch_simple_ffn":
            self.model = pytorchModel(model_type = "dense",**self.config.get("params", {}))
        else:
            raise ValueError(f"Model type '{model_type}' is not recognized.")

    

    def fit(self, X, y, **kwargs):
        """
        Train the loaded model on the provided data.

        Args:
            X (array-like): Feature data for training.
            y (array-like): Target data for training.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        self.model.fit(X, y, **kwargs)

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X (array-like): Feature data for predictions.

        Returns:
            numpy.ndarray: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        return self.model.predict(X)

# Example usage
if __name__ == "__main__":
    # Load model configuration from a JSON file
    config = 'xgboost_config.json'
    model = MLModel(os.path.join(CONFIG_DIR,config))

    from dataset_timeseries import GridmaticTimeseries
    t = GridmaticTimeseries('data.csv', target_col = 'CAISO_system_load', time_index = 'interval_start_time')
    t.fill_missing_values()
    t.clean_and_implement_features(lag_range = 96,convert_to_cyclic = False)
    # t.plot_pca_explained_variance()


    X_train,X_val,X_test,y_train,y_val,y_test = t.return_training_data(n_steps_ahead_to_predict = 24,rng=True)


    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Evaluate the model on the test set
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print("Mean squared error on test set: ", mse)
    print("Mean absolute error on test set: ", mae)

    # xgb_model = xgb.train(param, dtrain, 180, eval_list, early_stopping_rounds=3)


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(y_pred[24*30-10,:])
    plt.plot(y_val[24*30-10,:])
    plt.show()






