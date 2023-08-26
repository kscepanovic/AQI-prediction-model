import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest


class AqiDataHourlyCleaning:
    """A class for cleaning and preprocessing hourly AQI (Air Quality Index) data.

    This class provides methods to load data from a CSV file, perform regression-based imputation
    to fill missing values, detect and remove outliers using Isolation Forest, and offers modular
    data cleaning functionalities for AQI dataset."""

    def __init__(self, aqi_parameters_hourly):
        """Initialize the AqiDataHourlyCleaning instance."""

        self.aqi_data_hourly = aqi_parameters_hourly

    
    def load_data(self):
        """Load and preprocess the AQI data from the provided CSV file."""

        df = pd.read_csv(self.aqi_data_hourly, header=0, parse_dates=[['Date', 'Time']], 
                         dayfirst=True, date_format='%d.%m.%Y %H:%M')
        df.loc[df['NO2'] < 0, 'NO2'] = np.NAN
        df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        df.set_index('Date_Time', inplace=True)
        return df
    

    def regression_training(self, x_training_set, y_training_set, x_testing_set, y_testing_set, x_filling_set):
        """Perform SVR-based regression training for imputing missing AQI values."""

        pipe = Pipeline([('scaler', MinMaxScaler()), ('svr', SVR())])
        parameters = {'svr__kernel': ['rbf'], 'svr__C':[1, 2, 4, 8, 10], 'svr__epsilon':[0.01, 0.1, 1, 10]}

        grid_search = GridSearchCV(pipe, parameters).fit(x_training_set, y_training_set)
        best_parameters = grid_search.best_params_
        print(f"\tBest parameters: {best_parameters}")

        y_predicted_testing = grid_search.predict(x_testing_set)

        r2_sc = r2_score(y_testing_set, y_predicted_testing)
        mse = mean_squared_error(y_testing_set, y_predicted_testing)
        print(f"\tR squared: {r2_sc}")
        print(f"\tMean squared error: {mse}\n")

        calculated_values = grid_search.predict(x_filling_set)
        return calculated_values
    

    def fill_missing_values(self, dataframe, columns_to_fill):
        """Impute missing AQI values using regression-based imputation."""

        filled_df = dataframe.copy()

        for column in columns_to_fill:
            df_no_missing_val = filled_df[[f"C_{column}", column]].dropna()
            df_missing_val = filled_df[[f"C_{column}", column]][filled_df[column].isna()]

            x = df_no_missing_val.iloc[:, 0].values.reshape(-1, 1)
            y = df_no_missing_val.iloc[:, -1].values.reshape(-1, 1).ravel()
            x_fill = df_missing_val.iloc[:, 0].values.reshape(-1, 1)

            test_size = round(len(x) * 0.75)
            x_train = x[: test_size]
            x_test = x[test_size:]
            y_train = y[: test_size]
            y_test = y[test_size:]

            print(f"\tImputing missing values for {column} data using SVR.")
            calculated_values = self.regression_training(x_train, y_train, x_test, y_test, x_fill).ravel()
            filled_df.loc[df_missing_val.index, column] = calculated_values
            filled_df.drop([f"C_{column}"], axis=1, inplace=True)
        return filled_df
    

    def detect_and_remove_outliers(self, dataframe):
        """Detect and remove outliers from the AQI dataset using Isolation Forest."""

        cleaned_df = dataframe.copy()

        isolation_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=0)
        outliers = isolation_forest.fit_predict(cleaned_df)

        outlier_mask = outliers == -1
        outlier_indices = cleaned_df.index[outlier_mask]
        cleaned_df.drop(outlier_indices, inplace=True)
        return cleaned_df
