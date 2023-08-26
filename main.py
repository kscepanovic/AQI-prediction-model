import pandas as pd
from weather_data_daily import WeatherDataDailyCalculation
from aqi_data_cleaning import AqiDataHourlyCleaning
from aqi_data_daily import AqiDataDailyCalculation
from full_data_processing import FullDataProcessing


meteo_data_input_file_path = 'Data\Meteo_data_5_min.csv'
aqi_parameters_input_file_path = 'Data\AQI_parameters_AMP_CAMS_hourly.csv'


def main():
    """
    Main function to process weather and AQI data, perform calculations,
    test set analysis and generate output files.
    """
    print('START\n')
    # Daily weather data calculation
    print('Calculating daily meteorological data...')
    weather_data = WeatherDataDailyCalculation(meteo_data_input_file_path)
    weather_dataframe = weather_data.load_data()
    wind_mode = weather_data.wind_mode(weather_dataframe)
    weather_data_daily = weather_data.parameters_daily_calculation(weather_dataframe, wind_mode)
    print('\tSaving "Meteo_data_daily.csv"\n')
    weather_data_daily.to_csv('Data\Meteo_data_daily.csv')

    # Hourly AQI data processing
    print('Preprocessing hourly AQI data...')
    aqi_parameters = AqiDataHourlyCleaning(aqi_parameters_input_file_path)
    aqi_dataframe = aqi_parameters.load_data()
    columns_to_fill = ['NO2', 'PM10', 'PM25']
    aqi_dataframe = aqi_parameters.fill_missing_values(aqi_dataframe, columns_to_fill)
    aqi_parameters_hourly = aqi_parameters.detect_and_remove_outliers(aqi_dataframe)
    print('\tSaving "AQI_parameters_hourly.csv"\n')
    aqi_parameters_hourly.to_csv('Data\AQI_parameters_hourly.csv')

    # Daily AQI calculation
    print('Calculating daily AQI data...')
    aqi_data = AqiDataDailyCalculation()
    aqi_dataframe_daily = aqi_data.calculate_rolling_avg_and_sum_daily(aqi_parameters_hourly)
    aqi_data_daily = aqi_data.aqi_index_calculation(aqi_dataframe_daily)
    print('\tSaving "AQI_parameters_daily.csv"\n')
    aqi_data_daily.to_csv('Data\AQI_parameters_daily.csv')

    # Combine weather and AQI data
    print('Merging meteorological and AQI data...')
    all_data_daily = pd.concat([weather_data_daily, aqi_data_daily], axis=1)
    all_data_daily.index.name = 'Date'
    last_columns_to_shift = all_data_daily.columns[-2:]
    all_data_daily[last_columns_to_shift] = all_data_daily[last_columns_to_shift].shift(-1)
    all_data_daily.drop(all_data_daily.index[-1], inplace=True)
    print('\tSaving "All_data_daily.csv"\n')
    all_data_daily.to_csv('Data\All_data_daily.csv')

    # Final data processing
    print('Full data processing...')
    final_data_proccessing = FullDataProcessing()
    final_data = final_data_proccessing.data_transformation(all_data_daily)
    final_data = final_data_proccessing.correlation_calculation(final_data)
    y_pred, y_test = final_data_proccessing.xgboosting_training(final_data)
    output_dataframe = final_data_proccessing.test_set_analysis(y_pred, y_test)
    print('\tSaving "Output_test_set.csv"\n')
    output_dataframe.to_csv('Data\Output_test_set.csv', index=False)

    # Final data processing without meteo parameters
    print("Full data processing without meteorological parameters...")
    final_data_processing_no_meteo_params = FullDataProcessing()
    columns_to_use = ['NO2', 'PM10', 'PM25', 'AQI']
    all_data_daily_no_meteo_params= all_data_daily[columns_to_use]
    final_data_without_meteo_params = final_data_processing_no_meteo_params.data_transformation(all_data_daily_no_meteo_params)
    final_data_without_meteo_params = final_data_processing_no_meteo_params.correlation_calculation(final_data_without_meteo_params)
    y_pred_no_meteo_params, y_test_no_meteo_params = final_data_processing_no_meteo_params.xgboosting_training(final_data_without_meteo_params)
    output_dataframe_no_meteo_params = final_data_processing_no_meteo_params.test_set_analysis(y_pred_no_meteo_params, y_test_no_meteo_params)
    print('\tSaving "Output_test_set_no_meteo_parameters.csv"\n')
    output_dataframe_no_meteo_params.to_csv('Data\Output_test_set_no_meteo_parameters.csv', index=False)
    print('END\n')
    print('**All files prepared for further analysis in Power BI.')


if __name__ == "__main__":
    main()
