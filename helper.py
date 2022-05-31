import sys
import os
from config import *
from data_fetching import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
import math
# import click
# import logging
from pathlib import Path
# import matplotlib.pyplot as plt
# from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import mean_squared_error as mse

from keras.models import Sequential, save_model, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
#import matplotlib.pyplot as plt


def df_to_X_y3(df:pd.DataFrame, look_back:int=7, pred_col_name:str = "PV power"):
  y_index = df.columns.get_loc(pred_col_name)
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-look_back):
    row = [r for r in df_as_np[i:i+look_back]]
    X.append(row)
    label = df_as_np[i+look_back][y_index]
    y.append(label)
  return np.array(X), np.array(y)

def normalize_column(df:pd.DataFrame, col:int = 1, a:int=0, b:int=1):
    col_name = df.columns[col]
    df[col_name] = (df[col_name] - df[col_name].min())/(df[col_name].max() - df[col_name].min())
    df[col_name] = (b-a)*df[col_name]+a 
    return df 

def reverse_normalize(df:pd.DataFrame, col_name:str, a:int=0, b:int=1):
    a = pd.read_csv(processed_data_dir + "merged.csv")
    x_min = a[col_name].min()
    x_max = a[col_name].max()
    # print("xmin: ", x_min)
    # print("xmax: ", x_max)
    df[col_name] = df[col_name]*(x_max-x_min) + x_min
    return df
  
def train_model(df:pd.DataFrame, model_name:str = "lstm_model_v1", look_back:int=24, num_of_epochs:int=20, pred_col_name:str="PV power"):
    X, y = df_to_X_y3(df,look_back, pred_col_name)
    data_size = X.shape[0]
    train_size = math.floor(data_size*0.8)
    val_size = math.floor(data_size*0.1)
    X_train1, y_train1 = X[:train_size], y[:train_size]
    X_val1, y_val1 = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    # X_test1, y_test1 = X[train_size+val_size:], y[train_size+val_size:]
    global model_dir 
    model_dir = model_dir + model_name
    create_directory_if_missing(model_dir)

    model = Sequential()
    model.add(InputLayer((look_back, len(df.columns))))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'relu'))
    model.summary()
    checkpoint = ModelCheckpoint(model_dir, save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    history = model.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=num_of_epochs, callbacks=[checkpoint])
    # plot_model_history(history, model_name)
      
def create_directory_if_missing(path:str):
  print("create_directory")
  exists = os.path.exists(path)
  if not exists:
    try:
      os.makedirs(path)
      print("directory created successfully: " + path)
    except:
      print("Unable to create directory: " + path)
  else:
    print("directory already exists") 
  
def predict(df:pd.DataFrame, model, pred_col_name:str): 
  pred:str = "Predicted_"+pred_col_name
  actual:str = "Actual_"+pred_col_name
  look_back = 24
  X, y = df_to_X_y3(df,look_back, pred_col_name)
  data_size = X.shape[0]
  train_size = math.floor(data_size*0.8)
  val_size = math.floor(data_size*0.1)
  X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
  predictions = model.predict(X_test).flatten()
  df_pred = pd.DataFrame(data={pred:predictions, actual:y_test})
  df_pred = reverse_normalize(df_pred, pred_col_name)
  # plot_predictions(df_pred, pred_col_name)
  # df.to_csv(df_pred, pred_col_name + "_pred_2203.csv")
  # print("mse = " + str(mse(y, predictions)))
  
def forecast(df:pd.DataFrame, model, pred_col_name:str, look_back:int=24):
  df_as_np = df.to_numpy()
  pred:str = "Predicted_"+pred_col_name
  actual:str = "Predicted_"+pred_col_name
  X = []
  for i in range(len(df_as_np)-look_back):
    row = [r for r in df_as_np[i:i+look_back]]
    X.append(row)
  X = np.array(X)
  predictions = model.predict(X).flatten()
  df_pred = pd.DataFrame(data={pred:predictions, pred:predictions})
  df_pred = reverse_normalize(df_pred, pred_col_name)
  df_pred.to_csv(processed_data_dir + pred_col_name + "_prediction22_03.csv")
  # plot_forecasting(df_pred, pred_col_name)
  
  
def convert_model_to_tflite(model_dir:str, model_name:str):
  import tensorflow as tf
  converter = tf.lite.TFLiteConverter.from_saved_model(model_dir + model_name)
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops. 
  ]
  tflite_model = converter.convert()
  with open(model_dir + model_name, 'wb') as f:
    f.write(tflite_model)
    
def predict_next_hour(df:pd.DataFrame, model, look_back=24, pred_col_name="PV power"):
  y_index = df.columns.get_loc(pred_col_name)
  df_as_np = df.to_numpy()
  X = []
  y = []
  i = len(df_as_np)-look_back
  row = [r for r in df_as_np[i:i+look_back]]
  X.append(row)
  label = df_as_np[i+look_back-1][y_index]
  y.append(label)
  X = np.array(X)
  y = np.array(y)
  pred = model.predict(X).flatten()
  return pred

def add_day_sin_cos_and_normalize(df:pd.DataFrame):
  df.index = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')
  df['Seconds'] = df.index.map(pd.Timestamp.timestamp)
  day = 60*60*24
  df['Day sin']  = np.sin(df['Seconds'] * (2 * np.pi / day))
  df['Day cos']  = np.cos(df['Seconds'] * (2 * np.pi / day))
  df = df.drop('Seconds', axis=1)
  df = df.drop("Time", axis=1)
  for i in range (0,len(df.columns)):
      df = normalize_column(df, i)
  return df

def keep_only_next_24_hours_data(df, seconds):
    df.index = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')
    df['Seconds'] = df.index.map(pd.Timestamp.timestamp)
    df= df.loc[df["Seconds"] >= seconds]
    df = df.sort_values(by='Seconds')
    df = df.drop('Seconds', axis=1)
    return df

def predict_pv_power(df:pd.DataFrame, model, look_back=24, pred_col_name="PV power"):
  df_pv = df.drop(columns=["Wind speed", "Wind power", "Time"])
  df_future = pd.read_csv("data/weather/future.csv")
  # df_future = df_future.loc[df_future["Time"] >= t]
  df_future = keep_only_next_24_hours_data(df_future, seconds)
  
  pv_future = df_future.drop(columns=["Wind speed", "Wind power"])
  pv_future_norm = add_day_sin_cos_and_normalize(pv_future)
  
  try: 
    for i in range(0, len(pv_future_norm)):
      pv_future_norm[pred_col_name][i] = predict_next_hour(df_pv, model, look_back, pred_col_name)
      df_pv = df_pv.append(pv_future_norm.iloc[i])
  except:
    print("PV power prediction process failed")
    
  df_predicted = reverse_normalize(pv_future_norm, pred_col_name)
  pv_future[pred_col_name] = df_predicted[pred_col_name]
  pv_future = pv_future.drop(columns=["Seconds","Day sin","Day cos"])
  
  try:
    pv_future.to_csv("data/predictions/predicted.csv", index=False)
    print("Predicted data saved to ./data/predictions/predicted.csv")
  except:
    print("Failed to save predicted data")

  
def get_days_change_location(x):
  xposition = []
  n = x.iloc[0][:2]
  location = 0
  for i in x:
      if n != i[:2]:
          print(n)
          print("location = ", + location)
          xposition.append(location)
      n = i[:2]
      location +=1
  return xposition
  