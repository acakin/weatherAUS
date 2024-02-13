Weather Australia Dataset
Overview
The Weather Australia dataset is a collection of historical weather observations recorded at various weather stations across Australia. It contains a comprehensive set of meteorological variables, including temperature, rainfall, humidity, wind speed, atmospheric pressure, and more. The dataset is widely used by researchers, meteorologists, and data scientists for weather analysis, forecasting, and climate studies.

Features
Date: Date of the weather observation.
Location: Name or code of the weather station.
MinTemp: Minimum temperature recorded (in degrees Celsius).
MaxTemp: Maximum temperature recorded (in degrees Celsius).
Rainfall: Amount of rainfall recorded (in millimeters).
Evaporation: Water evaporation (in millimeters).
Sunshine: Hours of bright sunshine recorded.
WindGustSpeed: Maximum wind gust speed (in kilometers per hour).
WindSpeed9am: Wind speed at 9 am (in kilometers per hour).
WindSpeed3pm: Wind speed at 3 pm (in kilometers per hour).
Humidity9am: Relative humidity at 9 am (in percentage).
Humidity3pm: Relative humidity at 3 pm (in percentage).
Pressure9am: Atmospheric pressure at 9 am (in hPa).
Pressure3pm: Atmospheric pressure at 3 pm (in hPa).
Cloud9am: Cloud cover at 9 am (in octas).
Cloud3pm: Cloud cover at 3 pm (in octas).
Temp9am: Temperature at 9 am (in degrees Celsius).
Temp3pm: Temperature at 3 pm (in degrees Celsius).
RainToday: Binary variable indicating if it rained today (1 for "Yes", 0 for "No").
RainTomorrow: Binary target variable indicating if it will rain tomorrow (1 for "Yes", 0 for "No").

Usage
Feel free to use this dataset for research, analysis, or machine learning projects. Ensure to cite the source appropriately.

Data Sources
Australian Bureau of Meteorology (BOM) - Daily Weather Observations
Australian Bureau of Meteorology (BOM) - Climate Data Online
Open Meteo Archive API

Models Used
Logistic Regression: A simple linear model used for binary classification tasks.
Random Forest Classifier: An ensemble learning method based on decision trees, known for its robustness and accuracy.
LightGBM: A gradient boosting framework that uses tree-based learning algorithms and is optimized for speed and efficiency.
