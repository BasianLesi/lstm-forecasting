from config import *
from data_fetching import *
import pandas as pd
import numpy as np 

import time


if __name__ == '__main__':
    starttime = time.time()

    while True:
        update_data()
        time.sleep(60.0 - ((time.time() - starttime) % 60.0))
    