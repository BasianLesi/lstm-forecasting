# -*- coding: utf-8 -*-
from helper import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = g.ROOT_DIR
raw_data_dir = g.raw_data_dir
processed_data_dir = g.processed_data_dir

print(f"ROOT DIR = {ROOT_DIR}")
data_directory = ROOT_DIR+"/data/raw/"

def main(data_dir, model_dir):
    """ Runs the script for model training """
    
    try:
        df_pv = pd.read_csv(processed_data_dir + 'pv_norm.csv')
        pv_model = load_model(model_dir + "pv_model/")
        pv_forecast = pd.read_csv(processed_data_dir + "PV_predict_data.csv")
    except:
        logger.error("Unalbe loading df or model dir: " + processed_data_dir)
        sys.exit(1)
    
    try:
        df_wp = pd.read_csv(processed_data_dir + 'wp_norm.csv')
        wp_model = load_model(model_dir + "wp_model/")
        wp_forecast = pd.read_csv(processed_data_dir + "PV_predict_data.csv")

        
        
    except:
        logger.error("Unalbe loading df or model from dir: " + processed_data_dir)
        sys.exit(1)
        
    # predict(df_pv, pv_model, "PV power")
    # predict(df_wp, wp_model, "Wind power")
    forecast(pv_forecast, pv_model, "PV power")
    forecast(wp_forecast, wp_model, "Wind power")
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()