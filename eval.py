import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)

warnings.filterwarnings("ignore")


class TimeSeriesModelComparer:
    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, model_name, model):
        """Add a model to the comparer."""
        self.models[model_name] = model

    def predict(self, X_test):
        """Make predictions with all models."""
        for name, model in self.models.items():
            if isinstance(model, xgb.XGBRegressor) and X_test.ndim == 3:
                X_test_xgboost = X_test.copy().reshape(X_test.shape[0], -1)
                predictions = model.predict(X_test_xgboost)
                self.results[name] = predictions
                print(f"{name} model made predictions.")
            else:
                predictions = model.predict(X_test)
                self.results[name] = predictions
                print(f"{name} model made predictions.")

    def evaluate(self, y_test):
        """Evaluate all models using various metrics."""
        metrics = {}
        for name, predictions in self.results.items():
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            min_error = np.min(np.abs(y_test - predictions))
            max_error = np.max(np.abs(y_test - predictions))
            medae = median_absolute_error(y_test, predictions)

            metrics[name] = {
                "MSE": mse,
                "MAE": mae,
                "R2 Score": r2,
                "MAPE": mape,
                "Min Error": min_error,
                "Max Error": max_error,
                "MedAE": medae,
            }

        return pd.DataFrame(metrics)

    def plot_results(self, y_test, max_size):
        """Plot actual vs predicted values for all models."""
        plt.figure(figsize=(15, 8))

        plt.plot(
            y_test.flatten()[:max_size], label="Actual", color="black", linewidth=2
        )

        for name, predictions in self.results.items():
            plt.plot(predictions.flatten()[:max_size], label=f"{name} Predictions")

        plt.title("Actual vs Predicted")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    def get_best_model(self):
        """Get the model with the best R2 score."""
        best_model = max(self.results.keys(), key=lambda k: self.results[k]["R2 Score"])
        return best_model, self.results[best_model]


# Example Usage
if __name__ == "__main__":

    from dataset_timeseries import TimeseriesData
    from models.pytorch_models import pytorchModel

    t = TimeseriesData(
        "data.csv", target_col="CAISO_system_load", time_index="interval_start_time"
    )
    t.create_weekday_feature()
    t.create_month_feature(convert_to_cyclic=True)
    t.create_hour_feature(convert_to_cyclic=True)
    t.create_seasonal_feature()
    t.create_fft_feature(256)
    t.fill_missing_values()
    t.scale_features()
    t.apply_pca()

    n_steps_predict = 24
    past_states = 128
    xTrain, xVal, xTest, yTrain, yVal, yTest = t.return_training_data_recurrent(
        past_states, n_steps_predict
    )
    input_size = xTrain.shape[-1]

    # Initialize the comparer
    comparer = TimeSeriesModelComparer()

    # Add models
    kwargs = {
        "input_size": input_size,
        "output_size": n_steps_predict,
        "hidden_size": 128,
        "n_layers": 2,
        "batch_size": 16,
        "lr": 0.00025,
        "epochs": 2,
    }

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,  # max_depth= 180,
        subsample=0.9,
        colsample_bytree=0.95,
        alpha=0.5,
        gamma=0.5,
        eval_metric="rmse",
        min_child_weight=10,
        early_stopping_rounds=20,
    )
    mamba_model = pytorchModel(model_type="mamba", **kwargs)
    hyena_model = pytorchModel(model_type="hyena", **kwargs)
    gru_model = pytorchModel(model_type="gru", **kwargs)
    transformer_model = pytorchModel(model_type="transformer", **kwargs)
    lstm_model = pytorchModel(model_type="lstm", **kwargs)

    hyena_model.fit(xTrain, yTrain)

    xgb_model.fit(
        xTrain.copy().reshape(xTrain.shape[0], -1),
        yTrain,
        eval_set=[(xVal.copy().reshape(xVal.shape[0], -1), yVal)],
    )
    gru_model.fit(xTrain, yTrain)
    mamba_model.fit(xTrain, yTrain)
    transformer_model.fit(xTrain, yTrain)
    lstm_model.fit(xTrain, yTrain)

    # add models for comparison
    comparer.add_model("gru", gru_model)
    comparer.add_model("hyena", hyena_model)
    comparer.add_model("xgboost", xgb_model)
    comparer.add_model("Mamba", mamba_model)
    comparer.add_model("Transformer", transformer_model)
    comparer.add_model("LSTM", lstm_model)

    # Make predictions
    comparer.predict(xTest)

    # Evaluate models
    evaluation_results = comparer.evaluate(yTest)
    print("\nEvaluation Results:")
    print(evaluation_results)

    # Plot results
    comparer.plot_results(yTest.flatten(), max_size=n_steps_predict)
