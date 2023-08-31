import requests
import json
import datetime
import pandas as pd
import random 
from rdfpandas.graph import to_dataframe
from rdflib import Graph, Namespace, RDFS, QB, URIRef, RDF, XSD, Literal, DCTERMS
from cow_csvw.csvw_tool import COW
import os


from pathlib2 import Path


def json_extract(obj, key):
    arr = []
    def extract(obj, arr, key):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))



def get_data(lat, lon, include_aqi= False):
    r = requests.get(f"""http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid=8177dd5523cebe3df4f66dec1673fcc2""")
    #file = open("data.json", 'w')
    #json.dump(r.json(), file)
    #file.close()


    lat = json_extract(r.json(), "lat")
    lon = json_extract(r.json(), "lon")
    co = json_extract(r.json(), "co")
    no2 = json_extract(r.json(), "no2")
    o3 =  json_extract(r.json(), "o3")
    pm25 = json_extract(r.json(), "pm2_5")
    pm10 = json_extract(r.json(), "pm10")
    date_time = json_extract(r.json(), "dt")
    if include_aqi == True:
        aqi = json_extract(r.json(), "aqi")
    

    dt = datetime.datetime.fromtimestamp(date_time[0])
    dt = dt.strftime("%d/%m/%Y,%H:%M:%S")
    print(dt)

    if include_aqi ==False :
        data_dict = {"lat": float(lat[0]),
                    "lon": float(lon[0]),
                    "co": float(co[0]),
                    "no2": float(no2[0]),
                    "o3": float(o3[0]),
                    "pm25": float(pm25[0]),
                    "pm10": float(pm10[0]),
                    #"aqi": float(aqi[0]),
                    "dt": dt
        }
    elif include_aqi== True:
        data_dict = {"lat": float(lat[0]),
                    "lon": float(lon[0]),
                    "co": float(co[0]),
                    "no2": float(no2[0]),
                    "o3": float(o3[0]),
                    "pm25": float(pm25[0]),
                    "pm10": float(pm10[0]),
                    "aqi": float(aqi[0]),
                    "dt": dt
        }

    json_string = json.dumps(data_dict, default=json_serial)
    #df = pd.read_json(json_string, typ='series')
    #df.to_frame('count')
   # df.to_csv("test.csv")
    print(json_string)
    #file = open("data2.json", 'w')
    #file.write(json_string)
    #file.close()
    return data_dict

def create_dataset_randomized(data_size, aqi=False):
    '''
    Creates CSV and RDF Datasets from random Coordinates, takes as input number of observations to collect
    '''
    
    g = Graph()
    aqi = Namespace("http://aqi.com/")
    geo = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")


    g.bind("geo", geo) 
    g.bind("aqi", aqi)
    g.bind("dcterms", DCTERMS)
    
    for i in range(0,data_size):
        data_dict = get_data(random.uniform(-90, 90),random.uniform(-180, 180))
        id_str = f"""http://aqi.com/id/{data_dict["lat"]};{data_dict["lon"]};{data_dict["dt"]}"""
        id_str = str(id_str).replace("T", "/")
        observation = URIRef(id_str)
        g.add((observation, RDF.type, aqi.Observation))
        g.add((observation, geo.lat,  Literal(data_dict["lat"],datatype=XSD.float)))
        g.add((observation, geo.long, Literal(data_dict["lon"],datatype=XSD.float)))
        g.add((observation, aqi.PM25Concentration, Literal(data_dict["pm25"],datatype=XSD.float)))
        g.add((observation, aqi.PM10Concentration, Literal(data_dict["pm10"],datatype=XSD.float)))
        g.add((observation, aqi.NO2Concentration, Literal(data_dict["no2"],datatype=XSD.float)))
        g.add((observation, aqi.OzoneConcentration, Literal(data_dict["o3"],datatype=XSD.float)))
        g.add((observation, aqi.COConcentration, Literal(data_dict["co"], datatype=XSD.float)))
        g.add((observation, DCTERMS.date, Literal(data_dict["dt"], datatype=XSD.dateTime)))
        if aqi == True:
            g.add((observation, aqi.AQIValue, Literal(data_dict["aqi"], datatype=XSD.float)))


    
    g.serialize("graph2rdf_rand.ttl")

    df = to_dataframe(g)
    df.drop(columns=["rdf:type{URIRef}"], inplace=True)
    df.rename(columns={"aqi:COConcentration{Literal}(xsd:float)" :"CO", "aqi:NO2Concentration{Literal}(xsd:float)": "NO2", "aqi:PM25Concentration{Literal}(xsd:float)" : "PM25", "aqi:OzoneConcentration{Literal}(xsd:float)" : "Ozone", "aqi:PM10Concentration{Literal}(xsd:float)": "PM10", "dcterms:date{Literal}(xsd:dateTime)" : "Datetime", "geo1:lat{Literal}(xsd:float)" : "Lat", "geo1:long{Literal}(xsd:float)" :"Lon"}, inplace=True)
    df.to_csv("graph2csv_rand.csv", index=False)
    COW(mode='build', files=[os.path.join("C:/Users/Patrick Clark/Desktop/TESTS/", "graph2csv_rand.csv")],base='http://aqi.com/', delimiter=',', quotechar='\"')

   
    
    file = Path(r"graph2csv_rand.csv-metadata.json")
  
    # Reading and storing the content of the file in
    # a data variable
    data = file.read_text()
    
        # Replacing the text using the replace function
    data = data.replace('"@id": "http://aqi.com/graph2csv_rand.csv/column/Lat"','"propertyUrl" : "http://www.w3.org/2003/01/geo/wgs84_pos#lat"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_rand.csv/column/Lon"', '"propertyUrl" : "http://www.w3.org/2003/01/geo/wgs84_pos#long"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_rand.csv/column/CO"', '"propertyUrl" : "http://aqi.com/COConcentration"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_rand.csv/column/NO2"', '"propertyUrl" : "http://aqi.com/NO2Concentration"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_rand.csv/column/PM25"', '"propertyUrl" : "http://aqi.com/PM25Concentration"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_rand.csv/column/PM10"', '"propertyUrl" : "http://aqi.com/PM10Concentration"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_rand.csv/column/Ozone"', '"propertyUrl" : "http://aqi.com/OzoneConcentration"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_rand.csv/column/Datetime"', '"propertyUrl" : "http://purl.org/dc/terms/date"')



    
        # Writing the replaced data
        # in the text file
    file.write_text(data)


def create_dataset_from_coords(file_path):
    '''
    Creates CSV and RDF Datasets from a file of Coordinates, takes as input number of observations to collect
    '''
    
    g = Graph()
    aqi = Namespace("http://aqi.com/")
    geo = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")


    g.bind("geo", geo)
    g.bind("aqi", aqi)
    g.bind("dcterms", DCTERMS)
    df = pd.read_csv(file_path, encoding="utf-16")
    lat_list = df["lat"].tolist()
    long_list = df["long"].tolist()
    for lat, lon in zip(lat_list, long_list):
        data_dict = get_data(lat, lon)
        observation = URIRef(f"""http://aqi.com/id/{data_dict["lat"]};{data_dict["lon"]};{data_dict["dt"]}""")
        g.add((observation, RDF.type, aqi.Observation))
        g.add((observation, geo.lat,  Literal(data_dict["lat"],datatype=XSD.float)))
        g.add((observation, geo.long, Literal(data_dict["lon"],datatype=XSD.float)))
        g.add((observation, aqi.PM25Concentration, Literal(data_dict["pm25"],datatype=XSD.float)))
        g.add((observation, aqi.PM10Concentration, Literal(data_dict["pm10"],datatype=XSD.float)))
        g.add((observation, aqi.NO2Concentration, Literal(data_dict["no2"],datatype=XSD.float)))
        g.add((observation, aqi.OzoneConcentration, Literal(data_dict["o3"],datatype=XSD.float)))
        g.add((observation, aqi.COConcentration, Literal(data_dict["co"], datatype=XSD.float)))
        g.add((observation, DCTERMS.date, Literal(data_dict["dt"], datatype=XSD.dateTime)))
        if aqi == True:
            g.add((observation, aqi.AQIValue, Literal(data_dict["aqi"], datatype=XSD.float)))

    
    
    g.serialize("graph2rdf_from_file.ttl")
    df = to_dataframe(g)
    df.drop(columns=["rdf:type{URIRef}"], inplace=True)
    df.rename(columns={"aqi:COConcentration{Literal}(xsd:float)" :"CO", "aqi:NO2Concentration{Literal}(xsd:float)": "NO2", "aqi:PM25Concentration{Literal}(xsd:float)" : "PM25", "aqi:OzoneConcentration{Literal}(xsd:float)" : "Ozone", "aqi:PM10Concentration{Literal}(xsd:float)": "PM10", "dcterms:date{Literal}(xsd:dateTime)" : "Datetime", "geo1:lat{Literal}(xsd:float)" : "Lat", "geo1:long{Literal}(xsd:float)" :"Lon"}, inplace=True)
    df.to_csv("graph2csv_from_file.csv", index=False)
    g.serialize("graph2rdf_from_file.ttl")
    COW(mode='build', files=[os.path.join("C:/Users/Patrick Clark/Desktop/TESTS/", "graph2csv_from_file.csv")],base='http://aqi.com/', delimiter=',', quotechar='\"')
    
    file = Path(r"graph2csv_from_file.csv-metadata.json")
  
    # Reading and storing the content of the file in
    # a data variable
    data = file.read_text()
    
        # Replacing the text using the replace function
    data = data.replace('"@id": "http://aqi.com/graph2csv_from_file.csv/column/Lat"','"propertyUrl" : "http://www.w3.org/2003/01/geo/wgs84_pos#lat"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_from_file.csv/column/Lon"', '"propertyUrl" : "http://www.w3.org/2003/01/geo/wgs84_pos#long"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_from_file.csv/column/CO"', '"propertyUrl" : "http://aqi.com/COConcentration"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_from_file.csv/column/NO2"', '"propertyUrl" : "http://aqi.com/NO2Concentration"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_from_file.csv/column/PM25"', '"propertyUrl" : "http://aqi.com/PM25Concentration"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_from_file.csv/column/PM10"', '"propertyUrl" : "http://aqi.com/PM10Concentration"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_from_file.csv/column/Ozone"', '"propertyUrl" : "http://aqi.com/OzoneConcentration"')
    data = data.replace('"@id": "http://aqi.com/graph2csv_from_file.csv/column/Datetime"', '"propertyUrl" : "http://purl.org/dc/terms/date"')



    
        # Writing the replaced data
        # in the text file
    file.write_text(data)



    
def queries(g : Graph):
    data_query = """
    SELECT DISTINCT ?pm25
    WHERE {
        ?a rdf:type aqi:Observation.
        ?a aqi:PM25Concentration ?pm25.
    }"""
    qres = g.query(data_query)
    for row in qres:
        print(row.pm25)


    
create_dataset_randomized(500)
#create_dataset_from_coords("Lat_Long.csv")
g = Graph()

g.parse("rdf_ttl_random.ttl")
data_query = """
    SELECT DISTINCT ?pm25 ?pm10 ?co
    WHERE {
        ?a rdf:type aqi:Observation.
        ?a geo1:lat '-36.8942'^^xsd:float.
        ?a aqi:PM25Concentration ?pm25.
        ?a aqi:PM10Concentration ?pm10.
        ?a aqi:COConcentration ?co.
    }"""
qres = g.query(data_query)
for row in qres:
    print(row.pm25, row.pm10, row.co)