# AQI-prediction-model
<img src="https://github.com/kscepanovic/AQI-prediction-model/assets/142613209/55374770-6509-40b1-8e78-688d27b1a989" width="600">

---
## Description

The main goal of this model is to predict AQI values for the next 24 hours using air quality data (from three different pollutants) and meteorological data. The input data is gathered from a monitoring station in Pula, Croatia, located in the vicinity of a local garbage disposal center. Machine learning algorithms were employed to predict future AQI values, and the impact of meteorological data on the model's output was also assessed. Dashboards were created using input and output data for easier interpretation and visualization.

## Input data
Air quality data was obtained from one of the monitoring stations in Croatia (AMP Kaštijun), which measures concentrations of different pollutants in the air. The data is publicly available and was downloaded from: https://iszz.azo.hr/iskzl/postaja.html?id=285. Since some of the records were incomplete, another source of data was needed to fill missing values. For this purpose, data from the Copernicus satellite mission was obtained using the Atmosphere monitoring service from: https://atmosphere.copernicus.eu/data (data is free to use but registration is needed). Meteorological data was downloaded from: https://www.wunderground.com/dashboard/pws/IMEDUL3.

## How it works
This prediction model can be divided into three main parts:
- Data preprocessing
- Data analysis using machine learning algorithms
- Data visualization

> Data preprocessing involves calculating daily meteorological parameters from 5-minute intervals. For input AQI parameters (at 1-hour intervals), missing values are imputed using Support Vector Regression, and outliers are detected using the Isolation Forest technique. From this data, daily values of pollutants are calculated and used in the Common Air Quality Index formula to derive the daily air quality index. The dominant pollutant for each day is also identified.

> Data analysis begins with the calculation of correlation coefficients between all features, and some features with very low correlation are not considered for further analysis (such as certain meteorological parameters). An XGBoost model is used to predict AQI values for the next day, along with test set analysis (best hyperparameters are found using Grid Search).

> Using input data and data provided by the output of the model, dashboards are created in Power BI to provide a better understanding of the data and to visualize the final results. There are dashboards for weather parameters, AQI values, and different test statistics.

## Conclusion
Predicting future AQI values is a challenging task because the root causes of air pollution (emissions from combustion, industrial activities, agricultural practices, etc.) are hard to track and measure. Typically, concentrations of different air pollutants are used to predict AQI values, but this model also attempted to improve predictions using meteorological data. The final data clearly showed that meteorological data improved the model's accuracy in classifying AQI values into five classes (ranging from very low to very high values). Interestingly, as all data was gathered in the area of a local garbage disposal center, it was observed that the average yearly AQI was low, indicating that there is no significant air pollution in the area, and air quality is generally very good.

## Requirements and how to use the model
Python 3.10 is needed, and some additional packages are listed in the 'requirements.txt' file. To use the model, you only need to run the 'main.py' file. All '.py' files have docstrings and comments for easier understanding of the code. Input data and outputs of the model are located in the 'Data' folder. Data visualizations can be accessed through the 'Project_report.pbix' file. 
