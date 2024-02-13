import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

df = pd.read_csv('weatherAUS.csv')
df_list_loc = []

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

def scrapper(i):
    """
    meteo historical weather API function
    :param i: index
    :return: df_list_loc: list of dataframe with respect to location
    """

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat[i],
        "longitude": long[i],
        "start_date": df[df["Location"] == loc[i]]["Date"].min(),
        "end_date": df[df["Location"] == loc[i]]["Date"].max(),
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "et0_fao_evapotranspiration"]
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(3).ValuesAsNumpy()
    daily_precipitation_hours = daily.Variables(4).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(5).ValuesAsNumpy()
    daily_wind_gusts_10m_max = daily.Variables(6).ValuesAsNumpy()
    daily_wind_direction_10m_dominant = daily.Variables(7).ValuesAsNumpy()
    daily_et0_fao_evapotranspiration = daily.Variables(8).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s"),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["Location"] = loc[i]
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["rain_sum"] = daily_rain_sum
    daily_data["precipitation_hours"] = daily_precipitation_hours
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
    daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
    daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration

    daily_dataframe = pd.DataFrame(data = daily_data)
    return df_list_loc.append(daily_dataframe)


