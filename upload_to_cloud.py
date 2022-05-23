import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
import pandas as pd
from config import *


def upload_to_google_sheets(hours:int=24):
    scope = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
    ]  
    creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)  
    client = gspread.authorize(creds)
    try:
        sheet = client.open("future").sheet1 
    except:
        log("unable to open google sheet")
        exit(1)

    df = pd.read_csv(prediction_dir + 'predicted.csv')
    for i in range(len(df)-1, len(df)-hours-1, -1):
        insertRow = df.iloc[i].values.flatten().tolist()
        for s in range(1, len(insertRow)):
            insertRow[s] = float(insertRow[s])
        
        sheet.insert_row(insertRow, 2)
    pass        
    sheet.delete_rows(50, 50+hours)
    log("updated google sheet successfully")

# upload_to_google_sheets()