# -*- coding: utf-8 -*-
import h5py
import pip
import subprocess,sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.20.1"])
# import numpy as np

from helper import *
from keras.models import model_from_json
from upload_to_cloud import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def forecast_and_upload_PV_power():
    print(f"ROOT DIR = {ROOT_DIR}")
    data_directory = ROOT_DIR+"/data/raw/"

    df_predict = pd.read_csv(processed_data_dir + "make_predictions.csv")

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")

    predict_pv_power(df_predict, loaded_model, look_back=24, pred_col_name="PV power")
    
    
    
