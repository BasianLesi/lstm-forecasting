import pandas as pd
import numpy as np
from datetime import date,datetime
import requests
import json
from config import *

day = 60*60*24
today = date.today()
seconds = int(datetime.today().timestamp())
tomorrow = seconds + day;
yesterday = seconds - day;
today = datetime.fromtimestamp(seconds).strftime("%d-%m-%Y %H:%M")

api_key = "8af40bfbe568da6eecfc0b905b468c42"
lat = "55.1449" #Bornholm Latitude
lon = "14.9170" #Bornholm Longitude
yesterday_url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={yesterday}&appid={api_key}&units=metric"
today_url     = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={seconds}&appid={api_key}&units=metric"
two_day_forecast_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=alerts&appid={api_key}&units=metric"

def generate_url(_seconds):
    return f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={_seconds}&appid={api_key}&units=metric"

def get_forecasting_data():
    df = weather_api_call(two_day_forecast_url)
    df1 = pd.read_csv("data/weather/forecast_data.csv")
    df_concat = pd.concat([df, df1])
    df_concat = df_concat.sort_values(by='Time')
    df_concat.drop_duplicates(subset = ['Time'], keep = 'first', inplace = True) # Remove duplicates
    df_concat.to_csv(f"data/weather/forecast_data.csv", index=False)

def get_historical_5days_data():
    days = []
    seconds = int(datetime.today().timestamp())
    for i in range(1,6):
        days.append(seconds-i*day)
    df_list = []
    for sec in days:
        df_list.append(weather_api_call(generate_url(sec)))   
    df_concat = pd.concat(df_list)
    df_concat = df_concat.sort_values(by='Time')
    df_concat.drop_duplicates(subset = ['Time'], keep = 'first', inplace = True) # Remove duplicates
    df = pd.read_csv("data/weather/historical_data.csv")
    df = pd.concat([df, df_concat])
    df.drop_duplicates(subset = ['Time'], keep = 'first', inplace = True) # Remove duplicates
    df = df.sort_values(by='Time')
    seconds = int(datetime.today().timestamp())
    now = datetime.fromtimestamp(seconds).strftime("%d-%m-%Y %H:%M")
    mask = (df['Time'] < now) 
    df = df.loc[mask]
    df.to_csv(f"data/weather/historical_data.csv", index=False)

def normalize_column(df_forecast:pd.DataFrame, col:int = 1, a:int=0, b:int=1):
    df = pd.read_csv(processed_data_dir + "merged.csv")
    col_name = df_forecast.columns[col]
    max = df[col_name].max()
    min = df[col_name].min()
    df_forecast[col_name] = (df_forecast[col_name] - min)/(max - min)
    df_forecast[col_name] = (b-a)*df_forecast[col_name]+a 
    return df_forecast 

def concatenate_dataframes(df1, df2, df3):
    frames = [df1, df2, df3] 
    df =  pd.concat(frames) 
    df.drop_duplicates(subset = ['Time'], keep = 'first', inplace = True) # Remove duplicates
    df.to_csv('.csv', index=False)
    return df
    

def weather_api_call(url):
    response = requests.get(url)
    forecast = json.loads(response.text)
    time = []
    temperature = []
    uvi = []
    wind = []
    power = []

    for i in range(0, len(forecast["hourly"])):
        ts = forecast["hourly"][i]["dt"]
        date_time = datetime.utcfromtimestamp(ts).strftime('%d-%m-%Y %H:%M')
        time.append(date_time)
        temperature.append(forecast["hourly"][i]["temp"])
        uvi.append(forecast["hourly"][i]["uvi"]*100)
        wind.append(forecast["hourly"][i]["wind_speed"])
        power.append(0)

    df = pd.DataFrame(data={"Time":time, "Temperature":temperature, "PV power":power, "Solar radiation":uvi, "Wind power":power, "Wind speed":wind})
    return df



# df_yesterday = weather_api_call(yesterday_url)
# df_today = weather_api_call(today_url)
# df_two_day_forecast = weather_api_call(two_day_forecast_url)

# conc_df = concatenate_dataframes(df_yesterday, df_today, df_two_day_forecast)
# conc_df.to_csv("data/processed/forecast_data.csv", index=False)

# for i in range (1, len(conc_df.columns)):
#         conc_df = normalize_column(conc_df, i)

# conc_df.to_csv("data/processed/forecast_data_normalized.csv", index=False)

get_historical_5days_data()
get_forecasting_data()



























