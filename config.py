import os
if os.name == 'nt':
    ROOT_DIR = "c:\\Users\\roni1\Thesis\\lstm-forecasting"
else:
    ROOT_DIR = "/home/pi/lstm-forecasting"

raw_data_dir = ROOT_DIR+"/data/raw/"
processed_data_dir = ROOT_DIR+"/data/processed/"
model_dir = ROOT_DIR+"/models/"
figures_dir = ROOT_DIR+"/reports/figures/"
prediction_dir = ROOT_DIR+"/data/predictions/"
powerlab_dir = ROOT_DIR+"/data/powerlab/"

DEBUG = True

def log(s):
    if DEBUG:
        print(s)



































































































