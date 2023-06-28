import json
import pandas as pd
from pathlib2 import Path 
from geopy.geocoders import Nominatim


def prendi_dati():
    with open('Database/JSONS/API_Data_Complete.json') as json_file:
        data = json.load(json_file)
        count = 0
        with open ("Database/TXTS/API_data2.txt", "w", encoding="utf-8") as file:
            for line in data:
                try:
                    aqi = data[count]["data"]["aqi"]
                    o3 = data[count]["data"]["forecast"]["daily"]['o3'][0]['avg'] #o3, pm1count, pm25
                    no2 = data[count]["data"]["iaqi"]['no2']['v']
                    co = data[count]["data"]["iaqi"]['co']['v']
                    pm10 = data[count]["data"]["forecast"]["daily"]['pm10'][0]['avg']
                    pm25 = data[count]["data"]["forecast"]["daily"]['pm25'][0]['avg']
                    coordinates = data[count]["data"]["city"]["geo"]
                    #name = data[count]["data"]["city"]["name"]
                    print(aqi)
                    print(co)
                    print(o3)
                    print(no2)
                    print(pm25)
                    print(coordinates)
                    print(pm10)
                    #print(name)
                    count = count +1
                    string = ""
                    string = str(aqi) + "," + str(co) + "," + str(o3) + "," + str(no2) + ","+ str(pm25) + "," + str(coordinates) + "," + str(pm10)
                    file.write(string + '\n')
                    print(string)
                except KeyError:
                    aqi = data[count]["data"]["aqi"]
                    o3 = data[count]["data"]["forecast"]["daily"]['o3'][0]['avg'] #o3, pm1count, pm25
                    no2 = "na"
                    co = "na"
                    pm10 = data[count]["data"]["forecast"]["daily"]['pm10'][0]['avg']
                    pm25 = data[count]["data"]["forecast"]["daily"]['pm25'][0]['avg']
                    coordinates = data[count]["data"]["city"]["geo"]
                    #name = data[count]["data"]["city"]["name"]
                    print(aqi)
                    print(co)
                    print(o3)
                    print(no2)
                    print(pm25)
                    print(coordinates)
                    print(pm10)
                    #print(name)
                    count = count +1
                    string = ""
                    string = str(aqi) + "," + str(co) + "," + str(o3) + "," + str(no2) + ","+ str(pm25) + "," + str(coordinates) + "," + str(pm10)
                    #file.write(string + '\n')
                    print(string)
                    file.write(string + '\n')
                    
                
            
            
            
def from_txt_to_csv():
    df = pd.read_csv("Database/TXTS/API_data2.txt", sep=",", index_col=False, header=None, names=["AQI_Value", "CO_AQI_Value", "Ozone_AQI_Value", "NO2_AQI_Value", "PM2_5_AQI_Value", "Lat", "Lng", "PM10"])
    df.to_csv("Database/CSVS/API_Info.csv")

    repl = ""
    with open("Database/CSVS/API_Info.csv") as file:
        repl = file.read().replace("[", "").replace("]", "")

    with open("Database/CSVS/API_Info_CORRECT.csv", "w") as file2:
        file2.write(repl)


def city_from_coordinates():
    df = pd.read_csv("Database/CSVS/API_Info_CORRECT.csv", index_col=False)
    LAT = df['LAT']
    LON = df['LON']
    with open ("Database/TXTS/Coordinates.txt", "w", encoding="utf-8") as file:
        for x, y in zip(LAT, LON):
         data = str(f"{x},{y}")
         file.write(data)
         file.write('\n')
         
    #df = pd.read_csv("Database/TXTS/Coordinates.txt", header=None, names=["LAT", "LON"])
    #df.to_csv("Database/CSVS/Solo_Coordinate.csv", index=False)
    
    
    
    
    
def nominatim_data_complete():
    file = open("Database/TXTS/Coordinates.txt")
    file_list = file.read().splitlines()
    with open ("Database/TXTS/Nominatim_api_data.txt", "w", encoding="utf-8") as file2:
        count = 0      
        for row in file_list:
            count = count +1
            print(row)
            geolocator = Nominatim(user_agent="geoapiExercises")
            #Latitude = "34.00585" 48.7759
            #Longitude = "71.53775" 2.37577
            #Language = "en-US"
            location = geolocator.reverse(row, language='en-US')
            address = location.raw['address']
            
            city = address.get('city', '')
            country = address.get('country', '')
            village = address.get('village', '')
            town = address.get('town', '')
            
            cc = (country + "," + city+ "," + village+ "," + town + ";")
            print(cc)
            file2.write(cc)
            file2.write('\n')
            
    file = Path("Database/TXTS/Nominatim_api_data.txt")
    data = file.read_text()
    data = data.replace(",,,", ",")
    data = data.replace(",,", ",")
    data = data.replace(",;", "")
    data = data.replace(";", "")
    data = data.replace("Rwanda", "Rwanda,Kigali")
    data = data.replace("Australia", "Australia,Canberra")
    data = data.replace("Ethiopia", "Ethiopia,Addiss Abeba")
    data = data.replace("Nigeria", "Nigeria,Abuja")
    data = data.replace("Chile", "Chile,Santiago")
    data = data.replace("India", "India,New Delhi")
    data = data.replace("Jordan", "Jordan,Amman")
    data = data.replace("Ecuador", "Ecuador,Quito")
    data = data.replace("Thailand", "Thailand,Bangkok")
    
    file.write_text(data)
    
    df = pd.read_csv("Database/TXTS/Nominatim_api_data.txt", header=None, names=["COUNTRY", "CITY", "TOWN"])
    df = df.drop(columns=["TOWN"])
    df.to_csv("Database/CSVS/nominatim_city_country.csv")
    
    data1 = pd.read_csv('Database/CSVS/API_Info_CORRECT.csv')
    data2 = pd.read_csv('Database/CSVS/nominatim_city_country.csv')

    output1 = data1.merge(data2, left_index=True, right_index=True, how='inner')
    output1 = output1.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"])
    output1.to_csv("Database/CSVS/Nominatim_data_complete.csv", index=False)
            

#prendi_dati()
#from_txt_to_csv()
#city_from_coordinates()
#nominatim_data_complete()







