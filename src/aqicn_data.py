import requests
import json
import csv
import pandas as pd


def get_data(file_path):
    count = 0
    df = pd.read_csv(file_path)
    lat_list = df["lat"].tolist()
    long_list = df["long"].tolist()
    for lat, long in zip(lat_list, long_list):
        print(lat,long)
        count = count + 1
        r = requests.get(
            f"""https://api.waqi.info/feed/geo:{lat};{long}/?token=12f1f5758ff0027428dd7a64bbe1c50c7b10206b""")
        print(r.json())
        data = r.json()
        with open('Database/JSONS/API_data.json', 'w') as f:
            json.dump(data, f)
        print("Finished " + str(count))






get_data("Database/CSVS/Universities_No_Dupes.csv")
'''
  
'''