# -*- coding: utf-8 -*-
from helper import *
from keras.models import model_from_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


print(f"ROOT DIR = {ROOT_DIR}")
data_directory = ROOT_DIR+"/data/raw/"

# df_pv = pd.read_csv(processed_data_dir + 'pv_norm.csv')
# train_model(df_pv, model_name="pv_model",   num_of_epochs = 20, pred_col_name="PV power")

def make_prediction():
    ##TODO: create a function that predicts data
    raise NotImplementedError()


try:
    df_pv = pd.read_csv(processed_data_dir + 'pv_norm.csv')
    wp_model = load_model(model_dir + "wp_model/")
    # pv_forecast = pd.read_csv(processed_data_dir + "PV_predict_data.csv")
    norm = pd.read_csv(processed_data_dir + "norm.csv")
except:
    print("unable to load photovoltaic df and model")
    sys.exit(1)

if(len(sys.argv) > 1):
    print("arguments: ", sys.argv)
    if(sys.argv[1] == "h5"):
        print("loading h5 model")
        json_file = open(model_dir + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        pv_model = model_from_json(loaded_model_json)
        # load weights into new model
        pv_model.load_weights(model_dir + "model.h5")
        

predict_pv_power(norm, pv_model, look_back=24, pred_col_name="PV power")
predict_wp_power(norm, wp_model, look_back=24, pred_col_name="PV power")

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
plt.show()

