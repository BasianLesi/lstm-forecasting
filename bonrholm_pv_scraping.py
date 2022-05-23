from selenium import webdriver
import pandas as pd
import numpy as np
from datetime import datetime


import time


def fetch_actual_pv_data():
    driver = webdriver.PhantomJS(executable_path='C:\\phantomjs-2.1.1-windows\\bin\\phantomjs.exe')
    driver.get("https://bornholm.powerlab.dk/")

    # This will get the html after on-load javascript
    html2 = driver.execute_script("return document.documentElement.innerHTML;")

    i = html2.index("sub_solar_cells")

    solar_data_html = html2[i:i+500]

    end = solar_data_html.index("MW")

    for i in range(end-10, end):
        if solar_data_html[i] == '>':
            start = i + 1
    power = float(solar_data_html[start:end])


    time = int(datetime.today().timestamp()) 

    metrics = pd.read_csv("actual_pv_power.csv")
    metrics = metrics.append({"Time":time, "PV power":power}, ignore_index=True)
    metrics.to_csv("actual_pv_power.csv", index=False)



if __name__ == '__main__':
    starttime = time.time()

    while True:
        #fetch solar power data form https://bornholm.powerlab.dk every minute
        fetch_actual_pv_data()
        time.sleep(60.0 - ((time.time() - starttime) % 60.0))
        

