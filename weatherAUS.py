import numpy as np
import pandas as pd
import requests
import seaborn as sns
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report
from api import scrapper
import time
from api import df_list_loc
import lightgbm as lgb


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv('weatherAUS.csv')
df.head()

# Number of unique values
df.nunique()

# Data types
df.dtypes

# mean, sum, quantiles, max, min, std
df.describe()

# Number of observations and variables
df.shape

# Identifying numerical and categorical variables
def grab_col_names(dataframe, cat_th=3, car_th=50):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    # 1.3
    # Kategorik ve numerik değişken analizi

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Get dummies for categorical variables
df = pd.get_dummies(df, columns=["RainToday", "RainTomorrow"], drop_first=True)
df = df.rename(columns={"RainToday_Yes": "RainToday", "RainTomorrow_Yes": "RainTomorrow"})
df["RainToday"] = df["RainToday"].astype(int)
df["RainTomorrow"] = df["RainTomorrow"].astype(int)

# Average of target variable according to categorical variables

df.groupby(cat_cols).agg({"RainTomorrow": "mean"})


# Average of numerical variables according to target variable

for col in num_cols:
    print(df.groupby("RainTomorrow").agg({col: "mean"}))


# Outliers with graphical technique
for col in df[num_cols]:
    plt.figure()
    sns.boxplot(x=df[col])

def calculate_outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculate the lower and upper outlier thresholds based on the interquartile range (IQR).

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - col_name (str): The name of the column for which outliers are being calculated.
    - q1 (float, optional): The first quartile value. Default is 0.25.
    - q3 (float, optional): The third quartile value. Default is 0.75.

    Returns:
    - tuple: A tuple containing the lower and upper outlier thresholds.
    """

    q1_value, q3_value = dataframe[col_name].quantile([q1, q3])
    iqr = q3_value - q1_value
    return float(q1_value - 1.5 * iqr), float(q3_value + 1.5 * iqr)

def replace_outliers_with_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Replace outliers in a column of a DataFrame with the lower and upper thresholds.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the column with outliers.
    variable : str
        The name of the column with outliers.
    q1 : float, optional
        The first quartile value. Default is 0.25.
    q3 : float, optional
        The third quartile value. Default is 0.75.

    Returns
    -------
    None
    """
    low_limit, up_limit = calculate_outlier_thresholds(dataframe, col_name, q1, q3)
    dataframe.loc[dataframe[col_name] < float(low_limit), col_name] = int(low_limit)
    dataframe.loc[dataframe[col_name] > float(up_limit), col_name] = int(up_limit)

for col in num_cols:
    replace_outliers_with_thresholds(df, col, 0.05, 0.95)

# Missing value query
df.isnull().any()
df.isna().sum()


# Correlation analysis
num_and_target_columns = num_cols + ["RainToday", "RainTomorrow"]
corr_matrix = df[num_and_target_columns].corr()
sns.heatmap(corr_matrix, cmap='PuOr')

# Create new variable
# Get meteo data with scrapper function
"""
for num in range(49):
    scrapper(num)
    time.sleep(10)
"""

# Create dataframe meteo_api and csv file
"""
vertical_concat = pd.concat(df_list_loc, axis=0, ignore_index=True)
df_api = pd.DataFrame(vertical_concat)
df_api = df_api.rename(columns={"date": "Date"})
df_api = df_api.drop("Unnamed: 0", axis=1)
df_api.to_csv('meteo_api.csv')
"""

# Read meteo_api csv file
df_api = pd.read_csv('meteo_api.csv')



# Merge DataFrames based on Date and Location using the intersection method
merged_df = pd.merge(df, df_api, how='inner', left_on=['Date', 'Location'], right_on=['Date', 'Location'])

merged_df.head()

# Identify missing values
# Fill NaN values with values retrieved from an external API
# Drop columns as it's no longer needed after filling NaN values

merged_df["MinTemp"] = merged_df["MinTemp"].fillna(merged_df["temperature_2m_min"])
merged_df = merged_df.drop("temperature_2m_min", axis=1)

merged_df["MaxTemp"] = merged_df["MaxTemp"].fillna(merged_df["temperature_2m_max"])
merged_df = merged_df.drop("temperature_2m_max", axis=1)

merged_df["Rainfall"] = merged_df["Rainfall"].fillna(merged_df["rain_sum"])
merged_df = merged_df.drop("precipitation_sum", axis=1)
merged_df = merged_df.drop("rain_sum", axis=1)

merged_df["Evaporation"] = merged_df["Evaporation"].fillna(merged_df["et0_fao_evapotranspiration"])
merged_df = merged_df.drop("et0_fao_evapotranspiration", axis=1)

merged_df["WindGustSpeed"] = merged_df["WindGustSpeed"].fillna(merged_df["wind_gusts_10m_max"])
merged_df = merged_df.drop("wind_gusts_10m_max", axis=1)

merged_df["WindSpeed9am"] = merged_df["WindSpeed9am"].fillna(merged_df["wind_speed_10m_max"])
merged_df["WindSpeed3pm"] = merged_df["WindSpeed3pm"].fillna(merged_df["wind_speed_10m_max"])
merged_df = merged_df.drop("wind_speed_10m_max", axis=1)

# Fill NaN values with mean value of variable

merged_df["Pressure9am"] = merged_df["Pressure9am"].fillna(merged_df["Pressure9am"].mean())
merged_df["Pressure3pm"] = merged_df["Pressure3pm"].fillna(merged_df["Pressure3pm"].mean())

merged_df["Humidity3pm"] = merged_df["Humidity3pm"].fillna(merged_df["Humidity3pm"].mean())
merged_df["Humidity9am"] = merged_df["Humidity9am"].fillna(merged_df["Humidity9am"].mean())

merged_df["Cloud9am"] = merged_df["Cloud9am"].fillna(merged_df["Cloud9am"].mean())
merged_df["Cloud3pm"] = merged_df["Cloud3pm"].fillna(merged_df["Cloud3pm"].mean())

merged_df["Temp9am"] = merged_df["Temp9am"].fillna(merged_df["Temp9am"].mean())
merged_df["Temp3pm"] = merged_df["Temp3pm"].fillna(merged_df["Temp3pm"].mean())

merged_df["Sunshine"] = merged_df["Sunshine"].fillna(merged_df["Sunshine"].mean())

# Fill NaN values in the "RainToday" column with the "RainTomorrow" value from the preceding row

if merged_df["RainToday"].isna().any():
    merged_df["RainToday"] = merged_df["RainToday"].fillna(merged_df["RainTomorrow"].shift(1))

# Convert angle to directional name for wind direction variables using function

def angle_to_direction(angle):
    # Define directional sectors and their corresponding names
    sectors = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
               'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    # Calculate the index of the nearest directional sector
    index = round(angle / (360. / len(sectors))) % len(sectors)

    # Return the corresponding directional name
    return sectors[index]

merged_df["WindDirDom"] = merged_df["wind_direction_10m_dominant"].apply(angle_to_direction)
merged_df = merged_df.drop(["wind_direction_10m_dominant", "WindGustDir", "WindDir9am", "WindDir3pm"], axis=1)


# Create Meteorological Seasons variable
merged_df['Seasons'] = pd.to_datetime(df['Date']).dt.month.apply(lambda x: 'Spring' if 9 <= x <= 11 else
                                                            'Winter' if 6 <= x <= 8 else
                                                            'Autumn' if 3 <= x <= 5 else 'Summer')

# Create Astronomical Season variable
# Convert 'Date' column to datetime type
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
def get_astronomical_season(date):
    month = date.month
    day = date.day
    if (month == 9 and day >= 23) or (month == 10) or (month == 11) or (month == 12 and day < 21):
        return "Spring"
    elif (month == 12 and day >= 21) or (month == 1) or (month == 2) or (month == 3 and day < 21):
        return "Summer"
    elif (month == 3 and day >= 21) or (month == 4) or (month == 5) or (month == 6 and day < 21):
        return "Fall"
    else:
        return "Winter"

merged_df['Astron_Season'] = merged_df['Date'].apply(get_astronomical_season)

# Create Altitude and Latitude variables
loc = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
       'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
       'AliceSprings', 'Darwin', 'Katherine', 'Uluru']

lat = [-36.0748, -33.8907, -31.4967, -30.2963, -29.4628,
       -32.9295, -33.281544, -29.0333, -33.75, -41.3333,
       -33.8678, -33.9399228, -28.3339, -32.8064,
       -34.424, -35.2835, -35.4568, -35.5307, -37.5662,
       -36.7582, -38.111, -37.667111, -37.814, -34.1855,
       -36.3333, -38.354, -37.7167, -37.9222, -27.4679, -16.9237,
       -28.0003, -19.2664, -34.9287, -37.8318, -34.4682,
       -31.1998, -35.0269, -34.0333, -31.667663996, -31.9321,
       -31.9522, -32.9833, -34.976, -42.8794, -41.4388,
       -23.6975, -12.4611, -14.4652, -25.3457]

long = [146.924, 150.7426, 145.8344, 153.1135, 149.8416,
        151.7801, 151.579147, 167.95, 150.7, 173.1833,
        151.2073, 151.17527640000003, 116.9352, 151.8436,
        150.8935, 149.1281, 149.1099, 148.7713, 143.8496,
        144.2802, 147.068, 144.833480766796, 144.9633, 142.1625,
        141.65, 141.574, 145.0833, 141.2749, 153.0281, 145.7661,
        153.4309, 146.8057, 138.5986, 140.7792, 138.9977,
        136.8326, 117.8837, 115.1, 116.008999964, 115.9564,
        115.8614, 121.6333, 116.7302, 147.3294, 147.1347,
        133.8836, 130.8418, 132.2635, 131.0367]

# Create a dictionary mapping locations to latitude
location_to_lat = dict(zip(loc, lat))

# Assign latitude values to the Latitude column based on location
merged_df["Latitude"] = merged_df["Location"].map(location_to_lat)

# Create a dictionary mapping locations to longitude
location_to_long = dict(zip(loc, long))

# Assign latitude values to the Latitude column based on location
merged_df["Longitude"] = merged_df["Location"].map(location_to_long)

# Define the API endpoint
url = "https://api.open-meteo.com/v1/elevation"

# Fetch altitudes for each location
altitudes = []

for x, y in zip(lat, long):
    params = {
        "latitude": x,
        "longitude": y
    }
    response = requests.get(url, params=params)
    data = response.json()
    altitude = data.get('elevation', None)
    altitudes.append(altitude)

altitudes = [float(alt[0]) for alt in altitudes]

# Create a dictionary mapping locations to altitudes
location_to_altitude = dict(zip(loc, altitudes))

# Assign altitude values to the Altitude column based on location
merged_df["Altitude"] = merged_df["Location"].map(location_to_altitude)

def get_climate_zone(latitude):
    if latitude <= -23.5:
        return "Tropical"
    elif latitude < -23.5 and latitude >= -40:
        return "Subtropical"
    elif latitude < -40 and latitude <= -60:
        return "Temperate"
    else:
        return "Cold"

# Apply the function to create the 'Climate_Zone' column
merged_df['Climate_Zone'] = merged_df.apply(lambda row: get_climate_zone(row['Latitude']), axis=1)
merged_df.head()

# One-Hot Encoding & create merged dataframe csv file
merged_df = pd.get_dummies(merged_df, columns=["WindDirDom", "Seasons", "Astron_Season", "Climate_Zone"], dtype="int", drop_first=True)

merged_df.head()
merged_df = merged_df.drop("Unnamed: 0", axis=1)

merged_df.to_csv('merged_df.csv')


merged_df = pd.read_csv('merged_df.csv')

cat_cols, num_cols, cat_but_car = grab_col_names(merged_df)


# Standardization
rs = RobustScaler()
merged_df[num_cols] = rs.fit_transform(merged_df[num_cols])
merged_df.describe().T
merged_df.info()

# Modeling
y = merged_df["RainTomorrow"]
X = merged_df.drop(["RainTomorrow", "Date", "Location"], axis=1)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_log_reg))

#Logistic Regression:
"""
Logistic Regression:
              precision    recall  f1-score   support
           0       0.90      0.96      0.93     22672
           1       0.80      0.61      0.69      6420
    accuracy                           0.88     29092
   macro avg       0.85      0.78      0.81     29092
weighted avg       0.87      0.88      0.87     29092

"""

# Model 2: Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
print("\nRandom Forest Classifier:")
print(classification_report(y_test, y_pred_rf))

"""
Random Forest Classifier:
              precision    recall  f1-score   support
           0       0.91      0.95      0.93     22672
           1       0.80      0.68      0.73      6420
    accuracy                           0.89     29092
   macro avg       0.86      0.82      0.83     29092
weighted avg       0.89      0.89      0.89     29092

"""

# Model 3: LightGBM Classifier
lgb_classifier = lgb.LGBMClassifier()
lgb_classifier.fit(X_train, y_train)
y_pred_lgb = lgb_classifier.predict(X_test)
print("\nLightGBM Classifier:")
print(classification_report(y_test, y_pred_lgb))

"""
LightGBM Classifier:
              precision    recall  f1-score   support
           0       0.92      0.95      0.93     22672
           1       0.79      0.70      0.74      6420
    accuracy                           0.89     29092
   macro avg       0.85      0.82      0.84     29092
weighted avg       0.89      0.89      0.89     29092
"""


def plot_importance(model, features, num=len(X)):
    if hasattr(model, 'feature_importances_'):
        feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    else:
        feature_imp = pd.DataFrame({'Value': np.abs(model.coef_[0]), 'Feature': features.columns})

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()

plot_importance(log_reg, X)

plot_importance(rf_classifier, X)

plot_importance(lgb_classifier, X)

