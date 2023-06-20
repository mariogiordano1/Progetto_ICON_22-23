import requests
import json
import csv
import pandas as pd
from pathlib2 import Path
import json



def get_data(file_path):
    count = 0
    df = pd.read_csv(file_path, encoding="utf-16")
    lat_list = df["lat"].tolist()
    long_list = df["long"].tolist()
    with open(f"""Database/JSONS/API_data_TESTING.json""", 'a', encoding="utf-16") as f:
        for lat, long in zip(lat_list, long_list):
                print(lat, long)
                count = count + 1
                r = requests.get(
                        f"""https://api.waqi.info/feed/geo:{lat};{long}/?token=12f1f5758ff0027428dd7a64bbe1c50c7b10206b""")
                print(r.json())
                data = r.json()
                f.write("\n")
                json.dump(data, f)
                print("Finished " + str(count))
        f.write("_")



def change_encoding(file_path):
    with open(file_path, 'rb') as source_file:
        with open("Database/JSONS/API_data_TESTING_decoded.json", 'w+b') as dest_file:
            contents = source_file.read()
            dest_file.write(contents.decode('utf-16').encode('windows-1252', errors='replace'))

def format_file(file_path):
    file = Path(file_path)
    data = file.read_text()
    data = data.replace("}}}", '}}},')
    data = data.replace("}}},_", '}}}]')
    data = data.replace('{\"status\"', '[{\"status\"', 1)
    data = data.replace('{"status": "nope", "data": "can not connect"}', "")
    file.write_text(data)
    remove_blank_spaces_txt()



def remove_blank_spaces_txt():
    with open("Database/JSONS/API_Data_TESTING.json", 'r') as r, open("Database/JSONS/API_Data_Complete.json", 'w') as o:
        for line in r:
            # strip() function
            if line.strip():
                o.write(line)

def append_parent():
    file = open("Database/JSONS/API_Data_Complete.txt", mode='a+', encoding='windows-1252')
    file.write("}")
    file.close()
    p = Path("Database/JSONS/API_Data_Complete.txt", encoding='windows-1252')
    p.rename(p.with_suffix('.json'))


#get_data("Database/CSVS/Universities_No_Dupes.csv")
#change_encoding("Database/JSONS/API_data_TESTING.json")
format_file("Database/JSONS/API_data_TESTING.json")


'''
  
'''
