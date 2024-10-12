import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import torch
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


class TimeseriesData(object):
    def __init__(self, file_path, time_index, target_col, drop_col=[]):
        self.target_col = target_col
        self.data = None
        self.features = None
        self.target = None
        self.time_index = time_index
        self._load_data(file_path, drop_col)

        self.convert_time_index_from_string_to_datetime()

        print("Data Head:")
        print(self.data.head())

        print("Data description: ")
        self._print_description()

        print("Null Values: ")
        print(self.data.isnull().sum(axis=0))

    def _print_description(self):
        print(self.data.describe().round(2))

    def _load_data(self, file_path, drop_col):
        # Load data from a CSV file
        self.data = pd.read_csv(file_path)
        self.target = self.data[self.target_col].values.reshape(-1, 1)
        nonfeatures = drop_col
        self.features = self.data.drop(columns=nonfeatures)

    def convert_dtypes(self, convert_from, convert_to):
        cols = self.features.select_dtypes(include=[convert_from]).columns
        for col in cols:
            self.features[col] = self.features[col].values.astype(convert_to)

    def fill_missing_values(self):

        imputer = KNNImputer(n_neighbors=5)
        tmp = self.features.drop(columns=[self.time_index])
        tmp = pd.DataFrame(imputer.fit_transform(tmp), columns=tmp.columns)
        tmp[self.time_index] = self.features[self.time_index]
        self.features = tmp

    def analyze_feature_importance(self, method="tree", n_features=10, random_state=42):
        X = self.features.drop(columns=[self.time_index, self.target_col]).values
        y = self.features[self.target_col]

        if method == "tree":
            # Analyze feature importance using a Random Forest classifier
            rf_regressor = RandomForestRegressor(random_state=random_state)
            rf_regressor.fit(X, y)

            feature_importances = pd.Series(
                rf_regressor.feature_importances_, index=self.features.columns
            )
            top_features = feature_importances.nlargest(n_features)

        if method == "information":
            mi_scores = mutual_info_regression(X, y)
            top_features = pd.Series(
                [X.columns[i] for i in np.argsort(mi_scores)[-n_features:]]
            )  # Select top 5 features

        # Plot feature importance
        self.plot_feature_importance(top_features)

    def plot_seasonal_decomposition(self):
        self._check_datetime()
        res = sm.tsa.seasonal_decompose(
            pd.Series(
                self.features[self.target_col].values,
                index=self.features[self.time_index],
            ),
            model="multiplicative",
        )
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 12))
        res.observed.plot(ax=ax1, title="Observed")
        res.trend.plot(ax=ax2, title="Trend")
        res.resid.plot(ax=ax3, title="Residual")
        res.seasonal.plot(ax=ax4, title="Seasonal")
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_importance_series):
        # Plot feature importance
        feature_importance_series.plot(kind="barh", figsize=(10, 6))
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Top Important Features")
        plt.show()

    def plot_box_plot(self, feature_name):
        plt.figure()
        sns.boxplot(x=self.features[feature_name])
        plt.show()

    def plot_pca_explained_variance(self):
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        X = self.features.drop(columns=[self.time_index, self.target_col]).values
        X_norm = scaler_X.fit_transform(X)
        pca = PCA()
        X_pca = pca.fit(X_norm)
        num_components = len(X_pca.explained_variance_ratio_)
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(num_components), X_pca.explained_variance_ratio_)
        plt.plot(np.cumsum(X_pca.explained_variance_ratio_))
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance")
        plt.show()

    def trim_outliers(self, feature_name, max_val, min_val):
        # note, be sure to do this before filling in missing values
        self.features.loc[self.features[feature_name] > max_val, feature_name] = np.nan
        self.features.loc[self.features[feature_name] < min_val, feature_name] = np.nan

    def stationarity_test(self):
        # Visualize the time series
        time_series = self.target.flatten()
        plt.plot(time_series)
        plt.title("Time Series Data")
        plt.show()

        # Augmented Dickey-Fuller test
        result = adfuller(time_series)
        print("Augmented Dickey-Fuller Test:")
        print(f"Test Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        print(f"Critical Values: {result[4]}")

        # Interpret the results
        if result[1] <= 0.05:
            print("The time series is stationary.")
        else:
            print("The time series is not stationary.")

    def pairplot(self):
        plt.figure(figsize=(10, 6))
        sns.pairplot(self.data.drop(columns=[self.time_index]))
        plt.suptitle("Time Series Data Pair Plot", y=1.02)
        plt.show()

    def plot_covariance_matrix(self):
        # Plot covariance matrix of the time series data
        cov_estimator = EmpiricalCovariance()
        cov_estimator.fit(self.features)

        cov_matrix = pd.DataFrame(
            cov_estimator.covariance_,
            columns=self.features.columns,
            index=self.features.columns,
        )
        plt.figure(figsize=(10, 8))
        plt.imshow(cov_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Covariance Matrix")
        plt.colorbar()
        plt.xticks(range(len(cov_matrix)), cov_matrix.columns, rotation="vertical")
        plt.yticks(range(len(cov_matrix)), cov_matrix.columns)
        plt.show()

    def autocovariance_plot(self, max_lag=24 * 8):
        time_series = self.target.flatten()
        _, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))
        plot_acf(time_series, lags=max_lag, ax=ax1)
        plot_pacf(time_series, lags=max_lag, ax=ax2)
        plt.tight_layout()
        plt.show()

    def scale_features(self):
        times = self.features[self.time_index]
        self.features = self.features.drop(columns=[self.time_index])

        self.scaler = StandardScaler()
        self.features.iloc[:, :] = self.scaler.fit_transform(self.features)
        self.scaled = True
        self.features[self.time_index] = times

    def unscale_features(self):
        times = self.features[self.time_index]
        self.features = self.features.drop(columns=[self.time_index])

        if self.scaled:
            self.features.iloc[:, :] = self.scaler.inverse_transform(self.features)
            self.scaled = False

        self.features[self.time_index] = times

    def get_data(self):
        return (
            self.features.drop(columns=[self.target_col]).values,
            self.target.flatten(),
        )

    def convert_time_index_from_string_to_datetime(self):
        self.data[self.time_index] = pd.to_datetime(
            self.data[self.time_index], utc=True, infer_datetime_format=True
        )
        self.features[self.time_index] = pd.to_datetime(
            self.features[self.time_index], utc=True, infer_datetime_format=True
        )

    def _check_datetime(self):
        if not pd.api.types.is_datetime64_any_dtype(self.features[self.time_index]):
            raise TypeError(
                """Time index must be of datetime type. 
                Please try calling self.convert_time_index_from_string_to_datetime()."""
            )

    def create_day_feature(self, convert_to_cyclic=False):
        self._check_datetime()

        self.features["day"] = self.features[self.time_index].dt.day_of_year
        if convert_to_cyclic:
            # larger period accounts for leap years
            self.replace_cyclic_feature_with_rotary("day", 366)

    def create_weekday_feature(self, add_holidays=True):
        self._check_datetime()

        weekdays = (
            self.features[self.time_index].dt.day_of_week.astype(int).values > 4
        ).astype(int)
        self.features["weekdays"] = weekdays
        if add_holidays:
            us_holidays = holidays.US()
            times = self.features[self.time_index]
            for i in range(len(times)):
                if times[i] in us_holidays:
                    self.features["weekdays"][i] = 1

    def create_month_feature(self, convert_to_cyclic=False):
        self._check_datetime()

        self.features["month"] = self.features[self.time_index].dt.month
        if convert_to_cyclic:
            # larger period accounts for leap years
            self.replace_cyclic_feature_with_rotary("month", 12)

    def create_hour_feature(self, convert_to_cyclic=False):
        self._check_datetime()

        self.features["hour"] = self.features[self.time_index].dt.hour
        if convert_to_cyclic:
            self.replace_cyclic_feature_with_rotary("hour", 24)

    def create_lag_feature(self, feature, lag, target=False):
        if target:
            self.features["target" + "lag" + str(lag)] = self.data[
                self.target_col
            ].shift(lag)
        else:
            self.features[feature + "lag" + str(lag)] = self.features[feature].shift(
                lag
            )

    def create_rolling_mean_feature(self, feature, window, lag=0, target=False):
        if target:
            self.features[
                "target" + "rolling_mean" + str(window) + "lag" + str(lag)
            ] = (self.data[self.target_col].shift(lag).rolling(window=window).mean())
        else:
            self.features[feature + "rolling_mean" + str(window) + "lag" + str(lag)] = (
                self.data[self.target_col].shift(lag).rolling(window=window).mean()
            )

    def create_rolling_min_feature(self, feature, window, lag=0, target=False):
        if target:
            self.features["target" + "rolling_min" + str(window) + "lag" + str(lag)] = (
                self.data[self.target_col].shift(lag).rolling(window=window).min()
            )
        else:
            self.features[feature + "rolling_min" + str(window) + "lag" + str(lag)] = (
                self.data[self.target_col].shift(lag).rolling(window=window).min()
            )

    def create_rolling_max_feature(self, feature, window, lag=0, target=False):
        if target:
            self.features["target" + "rolling_max" + str(window) + "lag" + str(lag)] = (
                self.data[self.target_col].shift(lag).rolling(window=window).max()
            )
        else:
            self.features[feature + "rolling_max" + str(window) + "lag" + str(lag)] = (
                self.data[self.target_col].shift(lag).rolling(window=window).max()
            )

    def create_rolling_std_feature(self, feature, window, lag=0, target=False):
        if target:
            self.features["target" + "rolling_std" + str(window) + "lag" + str(lag)] = (
                self.data[self.target_col].shift(lag).rolling(window=window).std()
            )
        else:
            self.features[feature + "rolling_std" + str(window) + "lag" + str(lag)] = (
                self.data[self.target_col].shift(lag).rolling(window=window).std()
            )

    def create_fft_feature(self, window=48, lag=0, target=False):
        def rolling_fft(window):
            # Apply FFT and return the absolute values of the first half
            fft_values = np.fft.fft(window)
            return np.abs(fft_values[: len(fft_values) // 2])

        results = pd.DataFrame(
            [rolling_fft(df_) for df_ in self.features[self.target_col].rolling(window)]
        )
        self.features[["fft_" + str(i) for i in range(results.shape[1])]] = results

    def create_seasonal_feature(self, lag=24):
        vals = self.data[self.target_col].shift(lag).ffill().bfill().values
        seasonal_data = sm.tsa.seasonal_decompose(
            pd.Series(vals, index=self.data[self.time_index]), model="additive"
        )
        # self.features['seasonal_feature'] = seasonal_data.seasonal.values
        self.features["trend_feature"] = seasonal_data.trend.values

    def create_diff_feature(self, feature, target=False):
        if target:
            self.features["target_diff"] = self.data[self.target_col].diff()
        else:
            self.features[feature + "_diff"] = self.features[feature].diff()

    def replace_cyclic_feature_with_rotary(self, feature, period):
        self.features[feature + "x"] = np.cos(
            ((2.0 * np.pi) / period) * self.features[feature].astype(float)
        )
        self.features[feature + "y"] = np.sin(
            ((2.0 * np.pi) / period) * self.features[feature].astype(float)
        )
        self.features.drop(columns=[feature], inplace=True)

    def apply_pca(self, variance=0.8):
        pca = PCA(n_components=variance)  # Retain 75% of the variance

        data = self.features.drop(columns=[self.time_index, self.target_col]).values
        data = pca.fit_transform(data)

        # turn this into a function in dataset_timeseries
        times = self.features[self.time_index]
        target = self.features[self.target_col]
        col_names = ["pca_" + str(i) for i in range(data.shape[1])]
        self.features = pd.DataFrame(data, columns=col_names)
        self.features[self.time_index] = times
        self.features[self.target_col] = target

    def visualize_adfuller_results(self, feature, title):

        series = self.data[feature].interpolate().values
        result = adfuller(series)
        significance_level = 0.05
        adf_stat = result[0]
        p_val = result[1]
        crit_val_1 = result[4]["1%"]
        crit_val_5 = result[4]["5%"]
        crit_val_10 = result[4]["10%"]

        if (p_val < significance_level) & ((adf_stat < crit_val_1)):
            linecolor = "forestgreen"
        elif (p_val < significance_level) & (adf_stat < crit_val_5):
            linecolor = "orange"
        elif (p_val < significance_level) & (adf_stat < crit_val_10):
            linecolor = "red"
        else:
            linecolor = "purple"

        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 9))
        sns.lineplot(x=self.features[self.time_index], y=series, ax=ax, color=linecolor)
        ax.set_title(
            f"ADF Statistic {adf_stat:0.3f}, p-value: {p_val:0.3f}\nCritical Values 1%: {crit_val_1:0.3f}, 5%: {crit_val_5:0.3f}, 10%: {crit_val_10:0.3f}",
            fontsize=14,
        )
        ax.set_ylabel(ylabel=title, fontsize=14)
        f.show()

    def return_training_data(
        self,
        n_steps_ahead_to_predict=0,
        train_size=0.7,
        validation_size=0.15,
        rng=False,
        scale=True,
    ):

        if n_steps_ahead_to_predict < 0:
            raise ValueError("Value of n_steps_ahead_to_predict must be positive.")

        training_data = self.features
        if n_steps_ahead_to_predict == 0:
            X = training_data.drop(columns=[self.time_index, self.target_col]).values
            y = training_data[self.target_col].values

        else:
            X = (
                training_data.drop(columns=[self.time_index, self.target_col])
                .iloc[:-n_steps_ahead_to_predict, :]
                .values
            )
            y = (
                training_data[self.target_col]
                .shift(-n_steps_ahead_to_predict)
                .values[:-n_steps_ahead_to_predict]
            )

        if rng:
            y = pd.DataFrame(
                {
                    f"step_{i}": training_data[self.target_col].shift(-i)
                    for i in range(n_steps_ahead_to_predict)
                }
            ).values[:-n_steps_ahead_to_predict]

        train_len = int(X.shape[0] * train_size)
        validation_len = int(X.shape[0] * validation_size)

        if scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        X_train, y_train = X[:train_len, :], y[:train_len]
        X_val, y_val = (
            X[train_len : train_len + validation_len, :],
            y[train_len : train_len + validation_len],
        )
        X_test, y_test = (
            X[train_len + validation_len :, :],
            y[train_len + validation_len :],
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def toSequence(self, sequenceSize, steps_to_predict=1):

        self.scale_features()
        data = self.features.drop(columns=[self.time_index])
        target_index = data.columns.get_loc(self.target_col)
        data = data.values
        X = []
        y = []

        for i in range(len(data) - sequenceSize - steps_to_predict + 1):
            window = data[i : (i + sequenceSize), :]
            target = data[
                i + sequenceSize : i + sequenceSize + steps_to_predict, target_index
            ]

            X.append(window)
            y.append(target)
        X = np.array(X)
        y = np.array(y)

        return (
            torch.tensor(X, dtype=torch.float)
            .reshape(-1, sequenceSize, data.shape[1])
            .numpy(),
            torch.tensor(y, dtype=torch.float).reshape(-1, steps_to_predict).numpy(),
        )

    def return_training_data_recurrent(
        self,
        sequenceSize,
        n_steps_ahead_to_predict,
        train_size=0.85,
        validation_size=0.1,
    ):
        train_len = int(self.features.shape[0] * train_size)
        validation_len = int(self.features.shape[0] * validation_size)
        X, y = self.toSequence(sequenceSize, n_steps_ahead_to_predict)
        X_train, y_train = X[:train_len], y[:train_len]
        X_val, y_val = (
            X[train_len : train_len + validation_len],
            y[train_len : train_len + validation_len],
        )
        X_test, y_test = (
            X[train_len + validation_len :],
            y[train_len + validation_len :],
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


class GridmaticTimeseries(TimeseriesData):
    def __init__(
        self,
        filepath,
        target_col,
        time_index,
    ):
        super().__init__(filepath, target_col=target_col, time_index=time_index)

    def clean_and_implement_features(self, lag_range=72, convert_to_cyclic=True):
        self.fill_missing_values()

        self.create_fft_feature(window=240)
        self.create_seasonal_feature()
        for col in self.features:
            if col != self.time_index and (col == self.target_col or "trend" in col):
                self.create_rolling_mean_feature(
                    feature=col, target=False, lag=1, window=24 * 31
                )
                # self.create_rolling_mean_feature(feature=col,target = False,lag = 1, window = 24 * 14)
                self.create_rolling_mean_feature(
                    feature=col, target=False, lag=1, window=24 * 7
                )
                self.create_rolling_mean_feature(
                    feature=col, target=False, lag=1, window=24 * 3
                )
                self.create_rolling_mean_feature(
                    feature=col, target=False, lag=1, window=24
                )

                # self.create_rolling_max_feature(feature=col,target = False,lag = 1, window = 24 * 7)

                # self.create_rolling_min_feature(feature=col,target = False,lag = 1, window = 24 * 7)

        for col in self.features:
            if col != self.time_index and "mean" in col:
                self.create_diff_feature(col)

        for i in range(1, lag_range + 1):
            self.create_lag_feature(feature="", lag=i, target=True)

        self.create_month_feature(convert_to_cyclic=convert_to_cyclic)
        self.create_weekday_feature()
        self.create_hour_feature(convert_to_cyclic=convert_to_cyclic)

        self.features = self.features.dropna()


if __name__ == "__main__":

    target = "CAISO_system_load"

    t = GridmaticTimeseries(
        "data.csv", target_col="CAISO_system_load", time_index="interval_start_time"
    )
    t.fill_missing_values()
    t.plot_seasonal_decomposition()
    t.pairplot()
