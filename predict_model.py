# -*- coding: utf-8 -*-
from helper import *
from keras.models import model_from_json
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


print(f"ROOT DIR = {ROOT_DIR}")
data_directory = ROOT_DIR+"/data/raw/"

# df_pv = pd.read_csv(processed_data_dir + 'pv_norm.csv')
# train_model(df_pv, model_name="pv_model",   num_of_epochs = 20, pred_col_name="PV power")

def make_prediction():
    ##TODO: create a function that predicts data
    raise NotImplementedError()

def load_model_from_json(model_name:str="model")->tf.keras.Model:   # load model from json file
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
    
    
try:
    norm = pd.read_csv(processed_data_dir + "norm.csv")
    log("norm loaded")
except:
    log("Unable to load data")
    sys.exit(1)
    
try:
    pv_model = load_model_from_json("pv_model")
    wp_model = load_model_from_json("wp_model")
    log("models loaded")
except:
    log("unable to load models")
    sys.exit(1)


predict_pv_power(norm, pv_model, look_back=24, pred_col_name="PV power")
# predict_wp_power(norm, wp_model, look_back=24, pred_col_name="WP power")

plot_predicted_data()

# pred_col_name="PV power"
# past = pd.read_csv(processed_data_dir + "preprocessed.csv")
# predictions = pd.read_csv("data/predictions/predicted.csv")
# y1 = past[pred_col_name]
# y2 = predictions[pred_col_name]
# x1 = past["Time"].str[0:2] +"-"+ past["Time"].str[-5:-3]
# x2 = predictions["Time"].str[0:2] +"-"+ predictions["Time"].str[-5:-3]
# plt.plot(x1, y1, label = "past")
# plt.plot(x2, y2, label = "predictions")
# x = x1.append(x2)
# xposition = get_days_change_location(x)
# for xc in xposition:
#     plt.axvline(x=xc, color='k', linestyle='--')
# x_label = []
# for t in x:
#     x_label.append(t[3:5])
# plt.xticks(ticks=x[0::3], labels=x_label[0::3])
# plt.xlabel('Time - Day-Hour')
# plt.ylabel('PV Power')
# plt.title('PV power predictions')
# plt.legend()
# plt.savefig(figures_dir + pred_col_name + 'prediction_30_March.png', format='png')
# if DEBUG: plt.show()

