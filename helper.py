import sys
import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import numpy as np
import math
import click
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import mean_squared_error as mse

from keras.models import Sequential, save_model, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from global_variables import config as g

# ROOT_DIR = os.path.dirname(os.path.abspath("top_level_file.txt"))

ROOT_DIR = g.ROOT_DIR
raw_data_dir = g.raw_data_dir
processed_data_dir = g.processed_data_dir
model_dir = g.model_dir
figures_dir = g.figures_dir

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
    print("xmin: ", x_min)
    print("xmax: ", x_max)
    for col_name in df.columns:
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

    model_dir = g.model_dir + model_name
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
    plot_model_history(history, model_name)
      
def create_directory_if_missing(path:str):
  logger = logging.getLogger("create_directory")
  exists = os.path.exists(path)
  if not exists:
    try:
      os.makedirs(path)
      logger.info("directory created successfully: " + path)
    except:
      logger.info("Unable to create directory: " + path)
  else:
    logger.info("directory already exists")
    
def plot_model_history(history, model_name:str):
  plt.plot(history.history['loss'], 'g', label='Training loss')
  plt.plot(history.history['val_loss'], 'b', label='Validation loss')
  plt.title('Training and Validation loss')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend(loc='upper right')
  plt.savefig(figures_dir + model_name + '_loss.pdf', format='pdf', bbox_inches='tight')
  plt.show()
  
  plt.plot(history.history['root_mean_squared_error'], 'g', label='Training RMSE')
  plt.plot(history.history['val_root_mean_squared_error'],'b', label='Validation RMSE')
  plt.title('Training and Validation RMSE')
  plt.ylabel('RMSE')
  plt.xlabel('Epochs')
  plt.legend(loc='upper right')
  plt.savefig(figures_dir + model_name + '_RMSE.pdf', format='pdf', bbox_inches='tight')
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
  plt.show()
  
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
  plot_predictions(df_pred, pred_col_name)
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
  plot_forecasting(df_pred, pred_col_name)
  
  
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