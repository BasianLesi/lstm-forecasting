# -*- coding: utf-8 -*-
from helper import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


print(f"ROOT DIR = {ROOT_DIR}")
data_directory = ROOT_DIR+"/data/raw/"

try:
    df_pv = pd.read_csv(processed_data_dir + 'pv_norm.csv')
    pv_model = load_model(model_dir + "pv_model/")
    pv_forecast = pd.read_csv(processed_data_dir + "PV_predict_data.csv")
except:
    print("unable to load photovoltaic df and model")
    sys.exit(1)

try:
    df_wp = pd.read_csv(processed_data_dir + 'wp_norm.csv')
    wp_model = load_model(model_dir + "wp_model/")
    wp_forecast = pd.read_csv(processed_data_dir + "PV_predict_data.csv")
except:
    print("unable to load wind generation df and model") 
    sys.exit(1)
    
# predict(df_pv, pv_model, "PV power")
# predict(df_wp, wp_model, "Wind power")
forecast(pv_forecast, pv_model, "PV power")
forecast(wp_forecast, wp_model, "Wind power")
