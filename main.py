from config import *
from data_fetching import *
from upload_to_cloud import *
import pandas as pd
import numpy as np 

import time


# data is updated every hour
if __name__ == '__main__':
    starttime = time.time()

    while True:
        update_data()
        # make_forecasting_data()
        ##TODO: make prediction data
        # make_prediciton()
        # update_spreadsheet()
        time.sleep(3600.0 - ((time.time() - starttime) % 3600.0))   # sleep for the remainder of an hour
           
    