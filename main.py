from config import *
from data_fetching import *
from upload_to_cloud import *
from predict_model import *

import time

seconds = 60
minutes = 60
hour = float(seconds*minutes)
# data is updated every hour
if __name__ == '__main__':
    starttime = time.time()

    while True:
        log("task scheduled to run every hour")
        update_data()
        make_predictions_data()
        forecast_PV_power()
        upload_to_google_sheets()
        log("sleep for an hour")
        time.sleep(hour- ((time.time() - starttime) % hour))
    