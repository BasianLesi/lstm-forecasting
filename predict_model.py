# -*- coding: utf-8 -*-
import h5py
import pip
import subprocess,sys
# import numpy as np

from helper import *
from keras.models import model_from_json
from upload_to_cloud import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def load_model_from_json(model_name:str="model"):   # load model from json file
    json_file = open(model_dir + model_name + '.json', 'r') # open json file
    loaded_model_json = json_file.read()             # read json file
    json_file.close()                           # close json file
    log(f"Loaded model from {model_dir + model_name + '.json'}")    # log model name
    # load json and create model
    model = model_from_json(loaded_model_json)  # load json file
    # load weights into model
    model.load_weights(model_dir + model_name + ".h5")
    log(f"Loaded weights from {model_dir + model_name + '.h5'}")
    return model
    

def forecast_PV_power():
    print(f"ROOT DIR = {ROOT_DIR}")

    try:
        df_predict = pd.read_csv(processed_data_dir + "make_predictions.csv")
        log("norm loaded")
    except:
        log(f"Unable to load {processed_data_dir}/make_predictions.csv")
        sys.exit(1)

    try:
        pv_model = load_model_from_json("pv_model")
        log(f"pv_model loaded")
    except:
        log("unable to load pv_model")
        sys.exit(1)

    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("model.h5")

    predict_pv_power(df_predict, pv_model, look_back=24, pred_col_name="PV power")
    
    
forecast_PV_power()   
