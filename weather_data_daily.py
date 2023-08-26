import pandas as pd


class WeatherDataDailyCalculation:
    """A class for performing daily calculations on meteorological weather data.
    
    This class provides methods to load meteorological data from a CSV file with 5-minute intervals,
    calculate daily wind mode, and compute daily mean values for temperature, humidity, pressure,
    wind speed, and accumulated precipitation.
    """

    def __init__(self, meteo_data_5_min):
        """Initialize the WeatherDataDailyCalculation instance."""

        self.meteo_data = meteo_data_5_min

    
    def load_data(self):
        """Load and preprocess meteorological data from the provided CSV file."""

        columns_to_load = ['Date','Time', 'Temperature_C', 'Humidity_%', 
                           'Pressure_hPa', 'Wind', 'Speed_kmh', 'Precip_Accum_mm']
        
        df = pd.read_csv(self.meteo_data, usecols=columns_to_load, header=0, 
                         parse_dates=[['Date', 'Time']], dayfirst=True, 
                         date_format='%d/%m/%Y %I:%M %p')
        
        df['Date'] = pd.to_datetime(df['Date_Time'])
        df.set_index('Date', inplace=True)
        return df
    

    def wind_mode_calculation(self, daily_group):
            """Calculate the mode (most frequent value) of wind direction for a daily group."""

            try:
                return daily_group.value_counts().idxmax()
            except ValueError:
                pass


    def wind_mode(self, dataframe):
        """Return the daily wind mode (most frequent value) of wind direction."""

        wind_mode = dataframe['Wind'].resample(rule='D').apply(self.wind_mode_calculation)
        return wind_mode
    

    def parameters_daily_calculation(self, dataframe, wind_mode):
        """Calculate daily mean values of all parameters and include the daily wind mode in the data."""
        
        dataframe = dataframe.drop(['Wind'], axis=1)
        dataframe = dataframe.resample(rule='D').mean(numeric_only=True).round(2)
        dataframe['Wind'] = wind_mode
        return dataframe
