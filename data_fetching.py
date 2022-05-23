import pandas as pd
import numpy as np
from datetime import date,datetime
import requests
import json
from config import *

# global time variables in seconds to be used for api calls
##TODO: Update system time before script runnning!!
day = 60*60*24  
today = date.today()  
seconds = int(datetime.today().timestamp()) 
tomorrow = seconds + day;   
yesterday = seconds - day;  
today = datetime.fromtimestamp(seconds).strftime("%d-%m-%Y %H:%M")

# global api variables and url requests
api_key = "8af40bfbe568da6eecfc0b905b468c42"
lat = "55.1449" #Bornholm Latitude
lon = "14.9170" #Bornholm Longitude
yesterday_url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={yesterday}&appid={api_key}&units=metric"
today_url     = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={seconds}&appid={api_key}&units=metric"
two_day_forecast_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=alerts&appid={api_key}&units=metric"

# Update global variables
def update_global_variables(): 
    global today, seconds, tomorrow,yesterday, today_url, yesterday_url, two_day_forecast_url
    today = date.today()
    seconds = int(datetime.today().timestamp())
    tomorrow = seconds + day;
    yesterday = seconds - day;
    today = datetime.fromtimestamp(seconds).strftime("%d-%m-%Y %H:%M")

    yesterday_url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={yesterday}&appid={api_key}&units=metric"
    today_url     = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={seconds}&appid={api_key}&units=metric"
    two_day_forecast_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=alerts&appid={api_key}&units=metric"

# Returns a url string for the specified seconds
def generate_url(_seconds:int)->str:
    return f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={_seconds}&appid={api_key}&units=metric"

# Collects the actual weather for today until the present hour.
# saved as data/weather/present.csv
def get_today_weather():
    update_global_variables()
    df = weather_api_call(today_url)
    df.to_csv(f"data/weather/present.csv", index=False)

# Get the hourly forecasting weather for the next 24 hours
# saved as data/weather/future.csv
def get_forecasting_data():
    day = 48 #hours
    df = weather_api_call(two_day_forecast_url)
    df1 = pd.read_csv("data/weather/future.csv")
    df_concat = pd.concat([df[:day], df1])
    df_concat.index = pd.to_datetime(df_concat['Time'], format='%d-%m-%Y %H:%M')
    df_concat['Seconds'] = df_concat.index.map(pd.Timestamp.timestamp)
    df_concat.drop_duplicates(subset = ['Seconds'], keep = 'first', inplace = True)
    df_concat = df_concat.sort_values(by='Seconds')
    df = df_concat.drop('Seconds', axis=1)
    df.to_csv(f"data/weather/future.csv", index=False)

# Get the historical data from the api for the past 5 days
# save it as data/weather/past.csv
def get_historical_data():
    days = []   # List of days
    seconds = int(datetime.today().timestamp()) # Current seconds
    for i in range(1,6):    # Get the past 5 days
        days.append(seconds-i*day)  # Add the seconds to the list
    df_list = []    # List of dataframes
    for sec in days:    # For each day
        df_list.append(weather_api_call(generate_url(sec)))     # Get the dataframe for that day
    df_concat = pd.concat(df_list)  # Concatenate the dataframes
    df_concat = df_concat.sort_values(by='Time')    # Sort the dataframe by time
    df_concat.drop_duplicates(subset = ['Time'], keep = 'first', inplace = True) # Remove duplicates
    df = pd.read_csv("data/weather/past.csv")   # Read the past dataframe
    df_concat = pd.concat([df, df_concat])  # Concatenate the dataframes
    df_concat.index = pd.to_datetime(df_concat['Time'], format='%d-%m-%Y %H:%M')    # Set the index to time
    df_concat['Seconds'] = df_concat.index.map(pd.Timestamp.timestamp)  # Set the seconds column
    df_concat.drop_duplicates(subset = ['Seconds'], keep = 'first', inplace = True) # Remove duplicates
    df_concat = df_concat.sort_values(by='Seconds') # Sort by seconds
    df = df_concat.drop('Seconds', axis=1)  # Drop the seconds column
    df.drop_duplicates(subset = ['Time'], keep = 'first', inplace = True) # Remove duplicates
    df.to_csv(f"data/weather/past.csv", index=False)    # Save the dataframe

# Normalize dataframe columns data based on the large dataset that we used for training the lstm model
# Return the normalized dataframe
def normalize_column(df_forecast:pd.DataFrame, col:int = 1, a:int=0, b:int=1)->pd.DataFrame:
    df = pd.read_csv(processed_data_dir + "merged.csv")
    col_name = df_forecast.columns[col]
    max = df[col_name].max()
    min = df[col_name].min()
    df_forecast[col_name] = (df_forecast[col_name] - min)/(max - min)
    df_forecast[col_name] = (b-a)*df_forecast[col_name]+a 
    return df_forecast 

# Concatenate dataframes and remove drop_duplicates
# Return the concatenated dataframe
def concatenate_dataframes(df1, df2, df3)->pd.DataFrame:
    frames = [df1, df2, df3] 
    df =  pd.concat(frames) 
    df.drop_duplicates(subset = ['Time'], keep = 'first', inplace = True) # Remove duplicates
    df.to_csv('.csv', index=False)
    return df
    
# Make a call to the api and return the dataframe
def weather_api_call(url:str)->pd.DataFrame:    
    response = requests.get(url)    # Make a call to the api
    forecast = json.loads(response.text)    # Convert the response to json
    time = []   # Create a list to store the time
    temperature = []    # Create a list to store the temperature
    uvi = []    # Create a list to store uv index
    wind = []   # Create a list to store wind
    power = []  # Create a list to store power

    for i in range(0, len(forecast["hourly"])):   # Loop through the json response
        ts = forecast["hourly"][i]["dt"]    # Get the time
        date_time = datetime.utcfromtimestamp(ts).strftime('%d-%m-%Y %H:%M')    # Convert the time to a string
        time.append(date_time)  # Append the time to the list
        temperature.append(forecast["hourly"][i]["temp"])   # Append the temperature to the list
        uvi.append(forecast["hourly"][i]["uvi"]*100)    # Append the uvi to the list
        wind.append(forecast["hourly"][i]["wind_speed"])    # Append the wind to the list
        power.append(0)   # Append the power to the list

    df = pd.DataFrame(data={"Time":time, "Temperature":temperature, "PV power":power, "Solar radiation":uvi, "Wind power":power, "Wind speed":wind})
    return df

# Update data
def update_data():  
    get_historical_data()   
    get_forecasting_data()
    get_today_weather()
    log("data updated successfully")

# Merge the past 24 hours of actual weather data, pv power and wind power generation
# Save the data as data/processed/preprocessed.csv and data/processed/norm.csv for the normalized dataframe
# These data will be used for the prediction of the pv and windo power generation for the next 24 hours
def make_forecasting_data():
    update_data()
    df_past = pd.read_csv("data/weather/past.csv")  # Past 24 hours of weather data
    df_present = pd.read_csv("data/weather/present.csv")    # Today's weather data
    df_future = pd.read_csv("data/weather/future.csv")  # Future 24 hours of weather data
    df_past_present = pd.concat([df_past, df_present])  # Concatenate past and present data
    x = len(df_past_present)    # Number of rows in the past 24 hours data
    df = df_past_present[x-24:] # Select the last 24 rows of the past 24 hours data
    df_pv = pd.read_csv("data/raw/PV_power_gen_2703.csv")   # PV power generation data
    df_wp = pd.read_csv("data/raw/wind_power_gen_2703.csv") # Wind power generation data
    df_pv.rename(columns = {'Photovoltaic':'PV power'}, inplace = True) # Rename column
    df_wp.rename(columns = {'Wind':'Wind power'}, inplace = True)   # Rename column
    df = df.drop('PV power', axis=1)    # Drop the column
    df = df.drop('Wind power', axis=1)  # Drop the column
    df = df.merge(df_pv,on="Time", how="left")  # Merge the dataframes
    df = df.merge(df_wp,on="Time", how="left")  # Merge the dataframes
    df.to_csv("data/processed/preprocessed.csv", index=False)   # Save the dataframe as data/processed/preprocessed.csv

    df.index = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')  # Convert the time column to datetime
    df['Seconds'] = df.index.map(pd.Timestamp.timestamp)    # Convert the datetime to seconds
    day = 60*60*24  # Number of seconds in a day
    df['Day sin']  = np.sin(df['Seconds'] * (2 * np.pi / day))  # Sin of the seconds
    df['Day cos']  = np.cos(df['Seconds'] * (2 * np.pi / day))  # Cos of the seconds
    df = df.drop('Seconds', axis=1) # Drop the column
    for i in range (1,len(df.columns)): # Normalize the dataframe
        df = normalize_column(df, i) 
    df.to_csv("data/processed/norm.csv", index=False)   # Save the dataframe as data/processed/norm.csv

def preprocess_test_data():
    df = pd.read_csv(raw_data_dir + "bornholm_data.csv")    # Read the raw data

    for i, t in enumerate(df['timestamp']):            # Convert timestamp to datetime
        t1 = t.replace('T', ' ')                   # Replace T with space
        t1 = t1.replace('Z', '')               # Replace Z with nothing
        t1 = t1[:-7]                       # Remove the last 7 characters
        date_time_str = t1               # Convert to string
        t1 = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M') # Convert to datetime
        t1 = t1.replace(minute=00)     # Set the minute to 00
        t1 = t1.strftime('%d-%m-%Y %H:%M')  # Convert to string
        df.at[i, 'timestamp'] = t1   # Replace the timestamp with the new datetime
            
    df.index = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M') # Convert timestamp to datetime
    df['Seconds'] = df.index.map(pd.Timestamp.timestamp)         # Convert datetime to seconds
    df.rename(columns = {'timestamp':'Time'}, inplace = True)   # rename the column
    df["freq"] = 1                                            # Add a frequency column
    df = df.sort_values(by='Seconds')   # Sort by seconds
    df = df.groupby(['Time']).sum()     # Sum the values for each timestamp     
    df.reset_index(inplace=True)        # Reset the index
    df['Time'] =  pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')   # Convert to datetime
    df = df.sort_values(by='Time',ascending=True)   # Sort by timestamp
    df['Time'] = df['Time'].dt.strftime('%d-%m-%Y %H:%M')   # Convert to string
    
    df = df.drop('Seconds', axis=1) # Drop the seconds column
    df["Power"] = df["Measurement"]/df["freq"]
    df.to_csv(processed_data_dir + "bornholm_data.csv", index=False)     # Save the dataframe
    
def seconds_to_datetime(df:pd.DataFrame, col_name:str="Time")->pd.DataFrame:
    df['col_name'] = df['col_name'].apply(lambda x: datetime.fromtimestamp(x).strftime("%d-%m-%Y %H:00"))
    return df

def make_predictions_data():
    update_data()
    df_past = pd.read_csv("data/weather/past.csv")
    df_present = pd.read_csv("data/weather/present.csv")
    df_future = pd.read_csv("data/weather/future.csv")
    df_past_present = pd.concat([df_past, df_present])
    x = len(df_past_present)
    df = df_past_present[x-24:]
    # df_pv = pd.read_csv("data/raw/PV_power_gen_2703.csv")
    # df_wp = pd.read_csv("data/raw/wind_power_gen_2703.csv")
    # df_pv.rename(columns = {'Photovoltaic':'PV power'}, inplace = True)
    # df_wp.rename(columns = {'Wind':'Wind power'}, inplace = True)
    # df = df.drop('PV power', axis=1)
    # df = df.drop('Wind power', axis=1)
    # df = df.merge(df_pv,on="Time", how="left")
    # df = df.merge(df_wp,on="Time", how="left")
    df.to_csv("data/processed/preprocessed.csv", index=False)

    df.index = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')
    df['Seconds'] = df.index.map(pd.Timestamp.timestamp)
    day = 60*60*24
    df['Day sin']  = np.sin(df['Seconds'] * (2 * np.pi / day))
    df['Day cos']  = np.cos(df['Seconds'] * (2 * np.pi / day))
    df = df.drop('Seconds', axis=1)
    for i in range (1,len(df.columns)):
        df = normalize_column(df, i)
    df.to_csv("data/processed/make_predictions.csv", index=False)



# preprocess_test_data()
# update_data()
# make_predictions_data()

















