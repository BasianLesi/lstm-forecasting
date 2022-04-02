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

def update_global_variables(): # a function that updates global variables
    global today, seconds, tomorrow,yesterday, today_url, yesterday_url, two_day_forecast_url
    today = date.today()
    seconds = int(datetime.today().timestamp())
    tomorrow = seconds + day;
    yesterday = seconds - day;
    today = datetime.fromtimestamp(seconds).strftime("%d-%m-%Y %H:%M")

    yesterday_url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={yesterday}&appid={api_key}&units=metric"
    today_url     = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={seconds}&appid={api_key}&units=metric"
    two_day_forecast_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=alerts&appid={api_key}&units=metric"


def generate_url(_seconds):
    return f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={_seconds}&appid={api_key}&units=metric"

def get_today_weather():
    update_global_variables()
    df = weather_api_call(today_url)
    df.to_csv(f"data/weather/present.csv", index=False)

def get_forecasting_data():
    day = 24 #hours
    df = weather_api_call(two_day_forecast_url)
    df1 = pd.read_csv("data/weather/future.csv")
    df_concat = pd.concat([df[:day+2], df1])
    df_concat.index = pd.to_datetime(df_concat['Time'], format='%d-%m-%Y %H:%M')
    df_concat['Seconds'] = df_concat.index.map(pd.Timestamp.timestamp)
    df_concat.drop_duplicates(subset = ['Seconds'], keep = 'first', inplace = True)
    df_concat = df_concat.sort_values(by='Seconds')
    df = df_concat.drop('Seconds', axis=1)
    df.to_csv(f"data/weather/future.csv", index=False)

def get_historical_data():
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
    df = pd.read_csv("data/weather/past.csv")
    df_concat = pd.concat([df, df_concat])
    df_concat.index = pd.to_datetime(df_concat['Time'], format='%d-%m-%Y %H:%M')
    df_concat['Seconds'] = df_concat.index.map(pd.Timestamp.timestamp)
    df_concat.drop_duplicates(subset = ['Seconds'], keep = 'first', inplace = True)
    df_concat = df_concat.sort_values(by='Seconds')
    df = df_concat.drop('Seconds', axis=1)
    df.drop_duplicates(subset = ['Time'], keep = 'first', inplace = True) # Remove duplicates
    df.to_csv(f"data/weather/past.csv", index=False)

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

def update_data():
    get_historical_data()
    get_forecasting_data()
    get_today_weather()

def make_forecasting_data():
    update_data()
    df_past = pd.read_csv("data/weather/past.csv")
    df_present = pd.read_csv("data/weather/present.csv")
    df_future = pd.read_csv("data/weather/future.csv")
    df_past_present = pd.concat([df_past, df_present])
    x = len(df_past_present)
    df = df_past_present[x-24:]
    df_pv = pd.read_csv("data/raw/PV_power_gen_2703.csv")
    df_wp = pd.read_csv("data/raw/wind_power_gen_2703.csv")
    df_pv.rename(columns = {'Photovoltaic':'PV power'}, inplace = True)
    df_wp.rename(columns = {'Wind':'Wind power'}, inplace = True)
    df = df.drop('PV power', axis=1)
    df = df.drop('Wind power', axis=1)
    df = df.merge(df_pv,on="Time", how="left")
    df = df.merge(df_wp,on="Time", how="left")
    df.to_csv("data/processed/preprocessed.csv", index=False)

    df.index = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')
    df['Seconds'] = df.index.map(pd.Timestamp.timestamp)
    day = 60*60*24
    df['Day sin']  = np.sin(df['Seconds'] * (2 * np.pi / day))
    df['Day cos']  = np.cos(df['Seconds'] * (2 * np.pi / day))
    df = df.drop('Seconds', axis=1)
    for i in range (1,len(df.columns)):
        df = normalize_column(df, i)
    df.to_csv("data/processed/norm.csv", index=False)





















