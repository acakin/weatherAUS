
# Weather Australia Dataset

## Overview
The Weather Australia dataset is a collection of historical weather observations recorded at various weather stations across Australia. It contains a comprehensive set of meteorological variables, including temperature, rainfall, humidity, wind speed, atmospheric pressure, and more. The dataset is widely used by researchers, meteorologists, and data scientists for weather analysis, forecasting, and climate studies.

## Features
- **Date:** Date of the weather observation.
- **Location:** Name or code of the weather station.
- **MinTemp:** Minimum temperature recorded (in degrees Celsius).
- **MaxTemp:** Maximum temperature recorded (in degrees Celsius).
- **Rainfall:** Amount of rainfall recorded (in millimeters).
- **Evaporation:** Water evaporation (in millimeters).
- **Sunshine:** Hours of bright sunshine recorded.
- **WindGustSpeed:** Maximum wind gust speed (in kilometers per hour).
- **WindSpeed9am:** Wind speed at 9 am (in kilometers per hour).
- **WindSpeed3pm:** Wind speed at 3 pm (in kilometers per hour).
- **Humidity9am:** Relative humidity at 9 am (in percentage).
- **Humidity3pm:** Relative humidity at 3 pm (in percentage).
- **Pressure9am:** Atmospheric pressure at 9 am (in hPa).
- **Pressure3pm:** Atmospheric pressure at 3 pm (in hPa).
- **Cloud9am:** Cloud cover at 9 am (in octas).
- **Cloud3pm:** Cloud cover at 3 pm (in octas).
- **Temp9am:** Temperature at 9 am (in degrees Celsius).
- **Temp3pm:** Temperature at 3 pm (in degrees Celsius).
- **RainToday:** Binary variable indicating if it rained today (1 for "Yes", 0 for "No").
- **RainTomorrow:** Binary target variable indicating if it will rain tomorrow (1 for "Yes", 0 for "No").


## Usage
Feel free to use this dataset for research, analysis, or machine learning projects. Ensure to cite the source appropriately.

## Data Sources
- [Australian Bureau of Meteorology (BOM) - Daily Weather Observations](http://www.bom.gov.au/climate/dwo/)
- [Australian Bureau of Meteorology (BOM) - Climate Data Online](http://www.bom.gov.au/climate/data)
- [Open Meteo Archive API](https://archive-api.open-meteo.com/v1/archive)

## Models Used
- **Logistic Regression:** A simple linear model used for binary classification tasks.
- Logistic Regression:
              precision    recall  f1-score   support
           0       0.90      0.96      0.93     22672
           1       0.80      0.61      0.69      6420
    accuracy                           0.88     29092
   macro avg       0.85      0.78      0.81     29092
weighted avg       0.87      0.88      0.87     29092

- **Random Forest Classifier:** An ensemble learning method based on decision trees, known for its robustness and accuracy.
- Random Forest Classifier:
              precision    recall  f1-score   support
           0       0.91      0.95      0.93     22672
           1       0.80      0.68      0.73      6420
    accuracy                           0.89     29092
   macro avg       0.86      0.82      0.83     29092
weighted avg       0.89      0.89      0.89     29092

- **LightGBM:** A gradient boosting framework that uses tree-based learning algorithms and is optimized for speed and efficiency.
- LightGBM Classifier:
              precision    recall  f1-score   support
           0       0.92      0.95      0.93     22672
           1       0.79      0.70      0.74      6420
    accuracy                           0.89     29092
   macro avg       0.85      0.82      0.84     29092
weighted avg       0.89      0.89      0.89     29092

