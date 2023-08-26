class AqiDataDailyCalculation:
    """A class for calculating daily AQI (Air Quality Index) values from hourly data.
    
    This class provides methods to compute rolling averages of PM10 and PM2.5 concentrations,
    identify the maximum hourly NO2 concentration in a day, and calculate AQI and its corresponding
    dominant pollutant.
    """

    def calculate_rolling_avg_and_sum_daily(self, dataframe):
        """Calculate 24 hours rolling averages, daily means of those averages, and maximum hourly 
        NO2 concentration of the day."""

        averages_df = dataframe.copy()

        averages_df['PM10'] = averages_df['PM10'].rolling(24, min_periods=18).mean()
        averages_df['PM25'] = averages_df['PM25'].rolling(24, min_periods=18).mean()

        averages_df.drop(averages_df.index[:17], inplace=True)
        no2_hourly_max = averages_df['NO2'].resample(rule ='D').max().round(2)

        averages_df = averages_df.resample(rule='D').mean(numeric_only=True).round(2)
        averages_df['NO2'] = no2_hourly_max
        averages_df.index.name = 'Date'
        return averages_df
    

    def aqi_formula_calculation(self, row):
        """Calculate AQI and dominant pollutant based on pollutant concentrations."""

        map_NO2 = {0: [0.5, 0, 0], 1: [0.48, 50, 25], 2: [0.24, 100, 50], 
                   3: [0.12, 200, 75]}
        map_PM10 = {0: [1.67, 0, 0], 1: [1.6, 15, 25], 2: [1.20, 30, 50], 
                    3: [0.48, 50, 75]}
        map_PM25 = {0: [2.5, 0, 0], 1: [2.4, 10, 25], 2: [2.4, 20, 50], 
                    3: [0.8, 30, 75]}
        map_dict = {0: map_NO2, 1: map_PM10, 2: map_PM25}

        val_NO2 = [50, 100, 200, 400]
        val_PM10 = [15, 30, 50, 100]
        val_PM25 = [10, 20, 30, 60]
        val_dict = {0: val_NO2, 1: val_PM10, 2: val_PM25}

        max_aqi = 0
        for item_id in range(len(row)):
            for value in val_dict[item_id]:
                if row[item_id] > val_dict[item_id][-1]:
                    aqi = 100.99
                    break
                elif row[item_id] > value:
                    continue

                class_index = val_dict[item_id].index(value)
                class_values = map_dict[item_id][class_index]
                aqi = class_values[0] * (row[item_id] - class_values[1]) + class_values[2]
                break

            if aqi > max_aqi:
                max_aqi = aqi
                pollutant_id = item_id

        pollutant_id_map = {0: "NO2", 1: "PM10", 2: "PM25"}
        pollutant_name = pollutant_id_map[pollutant_id]

        return max_aqi, pollutant_name


    def aqi_index_calculation(self, dataframe):
        """Return AQI and dominant pollutant for each day's pollutant concentrations."""

        aqi_index_df = dataframe.copy() 

        aqi_index_df['AQI'], aqi_index_df['Dominant_pollutant'] = zip(*aqi_index_df.apply(self.aqi_formula_calculation, axis=1))
        return aqi_index_df
