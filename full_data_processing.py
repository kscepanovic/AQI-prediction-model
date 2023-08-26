import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class FullDataProcessing:
    """
    A class for processing and analyzing AQI data in order to predict future AQI.

    This class provides methods for data transformation, correlation calculation,
    XGBoost model training, and test set analysis.
    """

    def data_transformation(self, dataframe):
        """Transform the input dataframe by dropping missing values and encoding 
        categorical features."""

        dataframe = dataframe.dropna(how='any', axis=0)
        
        try:
            dummies = pd.get_dummies(dataframe.Wind)
            dataframe = pd.concat([dataframe, dummies], axis='columns')
            dataframe.drop(['Wind', 'East'], axis=1, inplace=True)

            column_list_rearranged = ['Temperature_C', 'Humidity_%', 'Speed_kmh',
                                'Pressure_hPa', 'Precip_Accum_mm', 'NO2', 'PM10',
                                'PM25', 'ENE', 'ESE', 'NE', 'NNE', 'NNW', 'NW',
                                'North', 'SE', 'SSE', 'AQI']
            dataframe = dataframe[column_list_rearranged]
            return dataframe
        except:
            return dataframe


    def correlation_calculation(self, dataframe):
        """Calculate and visualize correlations between features in the dataFrame."""

        correlations = dataframe.corr()
        plt.figure(figsize=(12, 10))
        heat_map = sb.heatmap(correlations, vmin=-1, vmax=1, annot=True,
                            linewidth=.5, square=True)
        print('\tData correlation using heatmap')
        plt.show()

        if dataframe.shape[1] > 4:
            final_columns = ['Temperature_C', 'Speed_kmh','Pressure_hPa',
                            'Precip_Accum_mm', 'NE', 'NO2', 'PM10', 'PM25','AQI']
            dataframe = dataframe[final_columns]
        return dataframe
    

    def xgboosting_training(self, dataframe):
        """Train an XGBoost model, perform hyperparameter tuning, and predict on the test set."""

        print('\tTraining XGBoost model...')
        X = dataframe.iloc[:, :-1].values
        y = dataframe.iloc[:, -1].values

        xgb_regressor = XGBRegressor(random_state=0)
        param_grid = {'objective': ['reg:squarederror'],
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.001, 0.01, 0.1, 1],
            'max_depth': [1, 5, 9, 13],
            'min_child_weight': [1, 5, 9, 13, 17]}

        tscv = TimeSeriesSplit(n_splits=4)
        grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, 
                                   cv=tscv, n_jobs=-1)
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        print(f"\tBest Hyperparameters: {best_params}")

        *_, last = tscv.split(X, y)
        _, test_index = last

        X_test, y_test = X[test_index], y[test_index]
        y_pred = best_model.predict(X_test)
        return y_pred, y_test
    

    def test_set_analysis(self, y_pred, y_test):
        """Analyze the model's performance on the test set and generate a scatter plot."""

        mse = mean_squared_error(y_test, y_pred)
        r2_sc = r2_score(y_test, y_pred)
        print(f"\tR squared: {r2_sc}")
        print(f"\tMean squared error: {mse}")

        test_set_data = {'y_pred': y_pred, 'y_test': y_test}
        df_test_set = pd.DataFrame(data=test_set_data)
        return df_test_set
