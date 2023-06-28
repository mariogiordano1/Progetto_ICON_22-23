import requests
import json
import csv
import pandas as pd
from pathlib2 import Path
import json
import numpy as np

def get_data(file_path):
    count = 0
    df = pd.read_csv(file_path)
    lat_list = df["Lat"].tolist()
    long_list = df["Lng"].tolist()
    with open(f"""Database/JSONS/API_Dataset_completo.json""", 'a') as f:
        for lat, long in zip(lat_list, long_list):
                print(lat, long)
                count = count + 1
                r = requests.get(
                        f"""https://api.waqi.info/feed/geo:{lat};{long}/?token=12f1f5758ff0027428dd7a64bbe1c50c7b10206b""")
                print(r.json())
                #break
                data = r.json()
                f.write("\n")
                json.dump(data, f)
                print("Finished " + str(count))
        f.write("_")


def remove_blank_spaces_txt():
    with open("Database/JSONS/API_Dataset_completo.json", 'r') as r, open("Database/JSONS/API_Dataset_completo.json", 'w') as o:
        for line in r:
            # strip() function
            if line.strip():
                o.write(line)


def format_file(file_path):
    file = Path(file_path)
    data = file.read_text()
    data = data.replace("}}}", '}}},')
    data = data.replace("}}},_", '}}}]')
    data = data.replace('{\"status\"', '[{\"status\"', 1)
    data = data.replace('{"status": "nope", "data": "can not connect"}', "")
    file.write_text(data)
    #remove_blank_spaces_txt()

def pm10_from_Datset_Completo():
    with open('Database/JSONS/API_Dataset_completo.json') as json_file:
        data = json.load(json_file)
        count = 0
        with open ("Database/CSVS/pm10_from_complete_ds.csv", "w", encoding="utf-8") as file:
            for line in data:               
                pm10 = data[count]["data"]["forecast"]["daily"]['pm10'][0]['avg']
                print(pm10)
                #print(name)
                count = count +1
                string = ""
                string = str(pm10) 
                file.write(string + '\n')

def concat_with_complete_ds():
        data1 = pd.read_csv('Database/CSVS/Dataset_completato.csv')
        data2 = pd.read_csv('Database/CSVS/pm10_from_complete_ds.csv', names=["PM10"])

        output1 = data1.merge(data2, left_index=True, right_index=True, how='inner')
        output1.to_csv("Database/CSVS/Dataset_comp_pm10.csv", index=False)
        
def create_final_dataset():
        df = pd.read_csv("Database/CSVS/Nominatim_data_complete.csv")
        cols = ["COUNTRY", "CITY", "AQI_Value", "CO_AQI_Value", "Ozone_AQI_Value", "NO2_AQI_Value", "PM2_5_AQI_Value", "Lat", "Lng", "PM10"]
        df = df[cols]
        df = df.rename(columns={'COUNTRY': 'Country', 'CITY': 'City'})
        df.to_csv("Database/CSVS/Nominatim_data_complete.csv")

        files = ['Database/CSVS/Dataset_comp_pm10.csv', 'Database/CSVS/Nominatim_data_complete.csv']
        df = pd.DataFrame()
        for file in files:
                data = pd.read_csv(file)
                df = pd.concat([df, data], axis=0)
                df.to_csv('Database/CSVS/merged_files.csv', index=False)
                
def fill_categories():
    df = pd.read_csv("Database/CSVS/merged_files.csv")
    df["PM10_Category"] = pd.cut(df['PM10'], bins=[0.0,50.9,100.9,250.9,350.9,430.9,np.inf], labels=["Good","Satisfactory", "Moderate", "Poor", "Very poor", "Severe"])
    
    df["AQI_Category"] = pd.cut(df['AQI_Value'], bins=[0.0,50.9,100.9,150.9,200.9,300.9,np.inf], labels=["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"])
    
    df["CO_AQI_Category"] = pd.cut(df['CO_AQI_Value'], bins=[-np.inf,29.9,199.9,799.9,np.inf], labels=["Good", "Moderate", "Hazardous", "Death"])
    
    df["Ozone_AQI_Category"] = pd.cut(df['Ozone_AQI_Value'], bins=[-np.inf,119.9,179.9,np.inf], labels=["Good", "Unhealthy for Sensitive Groups", "Hazardous"])
    
    df["NO2_AQI_Category"] = pd.cut(df['NO2_AQI_Value'], bins=[-np.inf,39.9,399.9,np.inf], labels=["Good", "Unhealthy", "Hazardous"])
    
    df["PM2_5_AQI_Category"] = pd.cut(df['PM2_5_AQI_Value'], bins=[-np.inf,29.9,59.9,89.9,119.9,249.9,np.inf], labels=["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])
    
    df.drop(columns=["Unnamed: 0"])
    df.to_csv("Database/CSVS/merged_files.csv", index=False)




#get_data("Dataset_Completato.csv")
#format_file("API_Dataset_Completo.json")
#pm10_from_Datset_Completo()
#concat_with_complete_ds()
#create_final_dataset()
#fill_categories()





            #sort = pd.read_csv("merged_files.csv", index_col=False)
#sort.replace('', np.nan, inplace=True)
#sort.replace('na', np.nan, inplace=True)
#sort.drop(columns=['Unnamed: 0.1'], inplace=True, axis=1)
            #sort.drop(columns=['Unnamed: 0'], inplace=True, axis=1)
            #sort.to_csv("merged_files.csv")
            #print(sort)
            

