from config import *
from data_fetching import *
from predict_model import *
import pandas as pd
import numpy as np 

import time


if __name__ == '__main__':
    starttime = time.time()

    while True:
        log("task scheduled to run every hour")
        update_data()
        make_predictions_data()
        forecast_PV_power()
        upload_to_google_sheets()
        log("sleep for an hour")
        time.sleep(3600.0 - ((time.time() - starttime) % 3600.0))
    