import numpy as np
import pandas as pd
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
def grab_col_names(dataframe, cat_th=3, car_th=20):
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
for col in df:
    plt.figure()
    sns.boxplot(x=df[col])


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


# Identify missing values
# Fill NaN values with values retrieved from an external API
# Drop columns as it's no longer needed after filling NaN values
"""
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
"""
# Fill NaN values with mean value of variable
"""
merged_df["Pressure9am"] = merged_df["Pressure9am"].fillna(merged_df["Pressure9am"].mean())
merged_df["Pressure3pm"] = merged_df["Pressure3pm"].fillna(merged_df["Pressure3pm"].mean())

merged_df["Humidity3pm"] = merged_df["Humidity3pm"].fillna(merged_df["Humidity3pm"].mean())
merged_df["Humidity9am"] = merged_df["Humidity9am"].fillna(merged_df["Humidity9am"].mean())

merged_df["Cloud9am"] = merged_df["Cloud9am"].fillna(merged_df["Cloud9am"].mean())
merged_df["Cloud3pm"] = merged_df["Cloud3pm"].fillna(merged_df["Cloud3pm"].mean())

merged_df["Temp9am"] = merged_df["Temp9am"].fillna(merged_df["Temp9am"].mean())
merged_df["Temp3pm"] = merged_df["Temp3pm"].fillna(merged_df["Temp3pm"].mean())

merged_df["Sunshine"] = merged_df["Sunshine"].fillna(merged_df["Sunshine"].mean())
"""
# Fill NaN values in the "RainToday" column with the "RainTomorrow" value from the preceding row
"""
if merged_df["RainToday"].isna().any():
    merged_df["RainToday"] = merged_df["RainToday"].fillna(merged_df["RainTomorrow"].shift(1))
"""
# Convert angle to directional name for wind direction variables using function
"""
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
"""


# One-Hot Encoding & create merged dataframe csv file
"""
merged_df = pd.get_dummies(merged_df, columns=["WindDirDom"], dtype="float", drop_first=True)
merged_df.to_csv('merged_df.csv')
"""

merged_df = pd.read_csv('merged_df.csv')
merged_df.describe()
merged_df.shape
cat_cols, num_cols, cat_but_car = grab_col_names(merged_df)


# Standardization
rs = RobustScaler()

merged_df[num_cols] = rs.fit_transform(merged_df[num_cols])
merged_df.describe().T
merged_df.info()

# Creating a model
# Encoding categorical variables
label_encoder = LabelEncoder()
merged_df['Date'] = label_encoder.fit_transform(merged_df['Date'])
merged_df['Location'] = label_encoder.fit_transform(merged_df['Location'])

y = merged_df["RainTomorrow"]
X = merged_df.drop(["RainTomorrow"], axis=1)

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
              precision    recall  f1-score   support
           0       0.89      0.96      0.92     22672
           1       0.79      0.58      0.67      6420
    accuracy                           0.87     29092
   macro avg       0.84      0.77      0.80     29092
weighted avg       0.87      0.87      0.87     29092
"""

# Model 2: Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
print("\nRandom Forest Classifier:")
print(classification_report(y_test, y_pred_rf))
#Random Forest Classifier:
"""
              precision    recall  f1-score   support
           0       0.91      0.95      0.93     22672
           1       0.79      0.67      0.72      6420
    accuracy                           0.89     29092
   macro avg       0.85      0.81      0.83     29092
weighted avg       0.88      0.89      0.88     29092
"""

# Model 3: LightGBM Classifier
lgb_classifier = lgb.LGBMClassifier()
lgb_classifier.fit(X_train, y_train)
y_pred_lgb = lgb_classifier.predict(X_test)
print("\nLightGBM Classifier:")
print(classification_report(y_test, y_pred_lgb))
#LightGBM Classifier:
"""
              precision    recall  f1-score   support
           0       0.92      0.95      0.93     22672
           1       0.79      0.69      0.74      6420
    accuracy                           0.89     29092
   macro avg       0.85      0.82      0.83     29092
weighted avg       0.89      0.89      0.89     29092
"""
def plot_importance(model, features, num=len(X), save=True):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(f'{model}_importances.png')


plot_importance(rf_classifier, X)

plot_importance(lgb_classifier, X)

