import sys
import os
from config import *  
from data_fetching import * 

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to run on CPU

import pandas as pd
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from keras.models import Sequential, save_model, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# transform data from dataframe to 3D array X and 1D array y
def df_to_X_y3(df:pd.DataFrame, look_back:int=7, pred_col_name:str = "PV power"):
  y_index = df.columns.get_loc(pred_col_name) # get the index of the column
  df_as_np = df.to_numpy()  # convert to numpy array
  X = []  # list of input sequences
  y = []  # list of output targets
  for i in range(len(df_as_np)-look_back):  # for each sequence
    row = [r for r in df_as_np[i:i+look_back]]  # get the sequence
    X.append(row) # add to input sequences
    label = df_as_np[i+look_back][y_index]  # get the label
    y.append(label) # add to output targets
  return np.array(X), np.array(y) # convert to numpy arrays

# normalize data by subtracting the mean and dividing by the standard deviation
#scale data to [a, b]: by default scaling to [0, 1]
def normalize_column(df:pd.DataFrame, col:int = 1, a:int=0, b:int=1): 
    col_name = df.columns[col]  # get the column name
    df[col_name] = (df[col_name] - df[col_name].min())/(df[col_name].max() - df[col_name].min())  # normalize
    df[col_name] = (b-a)*df[col_name]+a  # scale to [a,b]
    return df

# reverse the normalization
def reverse_normalize(df:pd.DataFrame, col_name:str, a:int=0, b:int=1):
    a = pd.read_csv(processed_data_dir + "merged.csv")  # read the dataframe
    x_min = a[col_name].min() # get the min value
    x_max = a[col_name].max() # get the max value
    df[col_name] = df[col_name]*(x_max-x_min) + x_min # reverse normalize
    return df
  
# train model and save it
def train_model(df:pd.DataFrame, model_name:str = "lstm_model_v1", look_back:int=24, num_of_epochs:int=20, pred_col_name:str="PV power"):
    X, y = df_to_X_y3(df,look_back, pred_col_name)  # transform data to X and y
    data_size = X.shape[0]  # get the size of the data
    train_size = math.floor(data_size*0.8)  # get the size of the training data
    val_size = math.floor(data_size*0.1)  # get the size of the validation data
    X_train1, y_train1 = X[:train_size], y[:train_size] # get the training data
    X_val1, y_val1 = X[train_size:train_size+val_size], y[train_size:train_size+val_size] # get the validation data
    global model_dir  # get the model directory
    model_dir = model_dir + model_name      # get the model directory
    create_directory_if_missing(model_dir)  # create the model directory

    model = Sequential()  # create the model
    model.add(InputLayer((look_back, len(df.columns)))) # add the input layer
    model.add(LSTM(64)) # add the LSTM layer
    model.add(Dense(8, 'relu')) # add the dense layer
    model.add(Dense(1, 'relu')) # add the output layer
    model.summary() # print the model summary
    checkpoint = ModelCheckpoint(model_dir, save_best_only=True)  # create the checkpoint
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])  # compile the model
    history = model.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=num_of_epochs, callbacks=[checkpoint]) # train the model
    plot_model_history(history, model_name) # plot the model history
      
# creates a direcotry if missing
def create_directory_if_missing(path:str):
  log("create_directory")
  exists = os.path.exists(path)
  if not exists:
    try:
      os.makedirs(path)
      log("directory created successfully: " + path)
    except:
      log("Unable to create directory: " + path)
  else:
    log("directory already exists")
    
# plot the model history
def plot_model_history(history, model_name:str):  # plot the model history
  plt.plot(history.history['loss'], 'g', label='Training loss') # plot the training loss
  plt.plot(history.history['val_loss'], 'b', label='Validation loss') # plot the validation loss
  plt.title('Training and Validation loss') # set the title
  plt.ylabel('Loss')  # set the y label
  plt.xlabel('Epochs')  # set the x label
  plt.legend(loc='upper right') # set the legend
  plt.savefig(figures_dir + model_name + '_loss.pdf', format='pdf', bbox_inches='tight')  # save the figure
  plt.show()  # show the figure

  plt.plot(history.history['root_mean_squared_error'], 'g', label='Training RMSE')  # plot the training RMSE
  plt.plot(history.history['val_root_mean_squared_error'],'b', label='Validation RMSE') # plot the validation RMSE
  plt.title('Training and Validation RMSE') # set the title
  plt.ylabel('RMSE')  # set the y label
  plt.xlabel('Epochs')  # set the x label
  plt.legend(loc='upper right') # set the legend
  plt.savefig(figures_dir + model_name + '_RMSE.pdf', format='pdf', bbox_inches='tight')  # save the figure
  plt.show()  
  
def plot_predictions(df_pred:pd.DataFrame, pred_col_name:str, start:int=0, end:int=500):
  pred:str = "Predicted_"+pred_col_name
  actual:str = "Actual_"+pred_col_name
  plt.plot(df_pred[pred][start:end], "-b", label=pred)
  plt.plot(df_pred[actual][start:end], "-r", label=actual)
  plt.legend(loc="upper right")
  plt.savefig(figures_dir + pred_col_name + '_Prediction_vs_Actual.pdf', format='pdf', bbox_inches='tight')
  plt.show() 
  
def plot_forecasting(df_pred:pd.DataFrame, pred_col_name:str, start:int=0, end:int=500):
  pred:str = "Predicted_"+pred_col_name
  actual:str = "Actual_"+pred_col_name
  plt.plot(df_pred[pred][start:end], "-b", label=pred)
  plt.plot(df_pred[pred][start:end], "-r", label=actual)
  plt.legend(loc="upper right")
  plt.savefig(figures_dir + pred_col_name + '_forecasting.pdf', format='pdf', bbox_inches='tight')
  if DEBUG: plt.show()

# make predictions using the model
def predict(df:pd.DataFrame, model, pred_col_name:str): 
  pred:str = "Predicted_"+pred_col_name # get the predicted column name
  actual:str = "Actual_"+pred_col_name  # get the actual column name
  look_back = 24  # set look_back time to 24 hours
  X, y = df_to_X_y3(df,look_back, pred_col_name)  # transform data to X and y
  data_size = X.shape[0]  # get the size of the data
  train_size = math.floor(data_size*0.8)  # get the size of the training data
  val_size = math.floor(data_size*0.1)  # get the size of the validation data
  X_test, y_test = X[train_size+val_size:], y[train_size+val_size:] # get the test data
  predictions = model.predict(X_test).flatten() # make the predictions
  df_pred = pd.DataFrame(data={pred:predictions, actual:y_test})  # create the dataframe
  df_pred = reverse_normalize(df_pred, pred_col_name) # reverse the normalization
  plot_predictions(df_pred, pred_col_name)  # plot the predictions
  # df.to_csv(df_pred, pred_col_name + "_pred_2203.csv")
  # log("mse = " + str(mse(y, predictions)))
  
def forecast(df:pd.DataFrame, model, pred_col_name:str, look_back:int=24):
  df_as_np = df.to_numpy()  # convert the dataframe to a numpy array
  pred:str = "Predicted_"+pred_col_name # get the predicted column name
  actual:str = "Predicted_"+pred_col_name # get the actual column name
  X = [] 
  for i in range(len(df_as_np)-look_back):  # iterate over the data
    row = [r for r in df_as_np[i:i+look_back]]  # get the row
    X.append(row) # add the row to the X array
  X = np.array(X) # convert the X array to a numpy array
  predictions = model.predict(X).flatten()  # make the predictions
  df_pred = pd.DataFrame(data={pred:predictions, pred:predictions}) # create the dataframe
  df_pred = reverse_normalize(df_pred, pred_col_name) # reverse the normalization
  df_pred.to_csv(processed_data_dir + pred_col_name + "_prediction22_03.csv") # save the dataframe
  plot_forecasting(df_pred, pred_col_name)  # plot the predictions
  
# convert model to tflite
# not used in our implementation may be useful in future development  
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
    
# make next hour prediction
def predict_next_hour(df:pd.DataFrame, model, look_back=24, pred_col_name="PV power"):
  y_index = df.columns.get_loc(pred_col_name) # get the index of the y column
  df_as_np = df.to_numpy()  # convert the dataframe to a numpy array
  X = []  # create the X array
  y = []  # create the y array
  i = len(df_as_np)-look_back # iterate over the data
  row = [r for r in df_as_np[i:i+look_back]]  # get the row
  X.append(row) # add the row to the X array
  label = df_as_np[i+look_back-1][y_index]  # get the label
  y.append(label) # add the label to the y array
  X = np.array(X) # convert the X array to a numpy array
  y = np.array(y) # convert the y array to a numpy array
  pred = model.predict(X).flatten() # make the prediction
  return pred # return the prediction

# Add sin and cos to corresponding seconds of the day
# normalize dataframe
def add_day_sin_cos_and_normalize(df:pd.DataFrame):
  df.index = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')  # convert the index to datetime
  df['Seconds'] = df.index.map(pd.Timestamp.timestamp)  # convert the index to seconds
  day = 60*60*24  # get the number of seconds in a day
  df['Day sin']  = np.sin(df['Seconds'] * (2 * np.pi / day))  # add the sin of the seconds
  df['Day cos']  = np.cos(df['Seconds'] * (2 * np.pi / day))  # add the cos of the seconds
  df = df.drop('Seconds', axis=1) # remove the seconds column
  df = df.drop("Time", axis=1)  # remove the time column
  for i in range (0,len(df.columns)): # iterate over the columns
      df = normalize_column(df, i)  # normalize the column
  return df   # return the normalized dataframe

# predictions for the PV power
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
    pv_future.to_csv(f"data/predictions/pv_predicted_{today[:-6]}.csv", index=False)
  except:
    print("Failed to save predicted data")
  

# predictions for the Wind power
def predict_wp_power(df:pd.DataFrame, model, look_back=24, pred_col_name="Wind power"):
  t = df["Time"].iloc[-1] # get the last time
  df_wp = df.drop(columns=["PV power","Solar radiation", "Time"]) # get the dataframe without the PV power, solar radiation and time columns
  df_future = pd.read_csv("data/weather/future.csv")  # get the future dataframe
  df_future = df_future.loc[df_future["Time"] >= t] # get the future dataframe after the last time
  
  wp_future = df_future.drop(columns=["PV power", "Solar radiation"])   # get the future dataframe without the PV power and solar radiation columns
  wp_future_norm = add_day_sin_cos_and_normalize(wp_future) # normalize the future dataframe
  
  for i in range(0, len(wp_future_norm)): # iterate over the future dataframe
    wp_future_norm[pred_col_name][i] = predict_next_hour(df_wp, model, look_back, pred_col_name)  # make next hour predictions and append it to the dataframe
    df_wp = df_wp.append(wp_future_norm.iloc[i])  # add the prediction to the dataframe
    
  df_predicted = reverse_normalize(wp_future_norm, pred_col_name) # reverse the normalization
  wp_future[pred_col_name] = df_predicted[pred_col_name]  # add the prediction to the future dataframe
  wp_future = wp_future.drop(columns=["Seconds","Day sin","Day cos"]) # remove the seconds, day sin and day cos columns
  wp_future.to_csv("data/predictions/wp_predicted.csv", index=False)  # save the future dataframe
  
def plot_predicted_data():
  pred_col_name="PV power"
  past = pd.read_csv(processed_data_dir + "preprocessed.csv")
  predictions = pd.read_csv("data/predictions/predicted.csv")
  y1 = past[pred_col_name]
  y2 = predictions[pred_col_name]
  x1 = past["Time"].str[0:2] +"-"+ past["Time"].str[-5:-3]
  x2 = predictions["Time"].str[0:2] +"-"+ predictions["Time"].str[-5:-3]
  plt.plot(x1, y1, label = "past")
  plt.plot(x2, y2, label = "predictions")
  x = x1.append(x2)
  xposition = get_days_change_location(x)
  for xc in xposition:
      plt.axvline(x=xc, color='k', linestyle='--')
  x_label = []
  for t in x:
      x_label.append(t[3:5])
  plt.xticks(ticks=x[0::3], labels=x_label[0::3])
  plt.xlabel('Time - Day-Hour')
  plt.ylabel('PV Power')
  plt.title('PV power predictions')
  plt.legend()
  plt.savefig(figures_dir + pred_col_name + 'prediction_30_March.png', format='png')
  if DEBUG: plt.show()

#get the dataframe iloc where the day changes meaning
def get_days_change_location(x):  # x is a list of strings
  xposition = []  # list of x locations where the day changes
  n = x.iloc[0][:2] # first day
  location = 0  # location of the first day
  for i in x: # for each day
      if n != i[:2]:  # if the day changes
          xposition.append(location)  # add the location to the list
      n = i[:2] # update the day
      location +=1  # update the location
  return xposition  # return the list of locations
  