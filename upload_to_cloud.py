import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
import pandas as pd
from config import *


def update_spreadsheet(hours:int=24):
    scope = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
    ]  
    creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)  
    client = gspread.authorize(creds)
    sheet = client.open("future").sheet1 

    df = pd.read_csv('predicted.csv')
    for i in range(len(df)-1, len(df)-hours-1, -1):
        insertRow = df.iloc[i].values.flatten().tolist()
        for s in range(1, len(insertRow)):
            insertRow[s] = float(insertRow[s])
        
        sheet.insert_row(insertRow, 2)
    pass        
    sheet.delete_rows(26, 26+hours)

update_spreadsheet()


         
         
         
        
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         