from config import *
from data_fetching import *
import pandas as pd
import numpy as np 
import requests
import json
from datetime import datetime
import os
import sys


if __name__ == '__main__':
    df = fetch_weather_forecast()
    merge_forecast_with_actual_data()
    