import sys

import pandas as pd
from pathlib2 import Path
import csv

from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE, RDFXML, RDF, TSV, CSV

file = open("Database/TXTS/Countries_No_Duplicate.txt", 'r', newline="\n")

file_to_list = file.read().splitlines()


def sparql_query_for_capitals():
    for countryname in file_to_list:
        endpoint_url = "https://dbpedia.org/sparql"
        file_output = open(f"""Database/CSVS/Capitals.csv""", 'a', newline="\n", encoding="utf-8")
        print(file.readlines())
        query = f"""SELECT ?citylabel
        WHERE
        {{
            ?country rdf:type dbo:Country.
            ?country rdfs:label "{countryname}" @en.
            ?country dbo:capital ?city.
            ?city rdfs:label ?citylabel.
            FILTER(LANG(?citylabel) = "en").
        }}"""

        user_agent = "botpython/0.1 /%s.%s" % (sys.version_info[0], sys.version_info[1])
        sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(CSV)

        file_output.write(str(sparql.query().convert()) + "\n")


def format_file(file_path):
    remove_blank_spaces("Database/CSVS/Capitals.csv",file_path)
    file = Path(file_path)
    data = file.read_text()
    data = data.replace("b'", "")
    data = data.replace("citylabel", "")
    data = data.replace('\\n', "")
    data = data.replace('\'', "")
    data = data.replace('\"', "")
    data = data.replace('\\xc3\\xad', "í")
    data = data.replace('\\xc3\\xa9', "é")
    data = data.replace('\\xc3\\xa1', "á")
    data = data.replace('\\xc3\\xb3', "ó")
    data = data.replace("GitegaBujumbura", "Gitega")
    data = data.replace("Moroni, Comoros", "Moroni")
    data = data.replace("San José, Costa Rica", "San_José")
    data = data.replace("Ciudad de la PazMalabo", "Malabo")
    data = data.replace("Georgetown, Guyana", "Georgetown")
    data = data.replace("JerusalemStatus of Jerusalem", "Jerusalem")
    data = data.replace("PutrajayaKuala Lumpur", "Kuala_Lumpur")
    data = data.replace("Kingston, Jamaica", "Kingston")
    data = data.replace("Tripoli, Libya", "Tripoli")
    data = data.replace("Monaco CityCity-state", "Monaco City")
    data = data.replace("Muscat, Oman", "Muscat")
    data = data.replace("Tripoli, Libya", "Tripoli")
    data = data.replace("Monaco CityCity-state", "Monaco City")
    data = data.replace("ManilaMetro Manila", "Manila")
    data = data.replace("Victoria, Seychelles", "Victoria")
    data = data.replace("City-state", "")
    data = data.replace("Cape TownPretoriaBloemfontein", "Cape Town")
    data = data.replace("ColomboSri Jayawardenepura Kotte", "Sri Jayawardenepura Kotte")
    data = data.replace("JerusalemRamallah", "Ramallah")
    data = data.replace("Nassau, Bahamas", "Nassau")
    data = data.replace("BernDe jure", "Bern")
    data = data.replace("St. John\s, Antigua and Barbuda", "St._John_s")
    data = data.replace('\\', "_")
    data = data.replace(" ", "_")

    file.write_text(data)
    remove_blank_spaces(file_path, "Database/CSVS/Capitals-Complete.csv")

file_path = r"Database/CSVS/Capitals.csv"

def remove_blank_spaces(file_path_original, file_path_output):
    with open(file_path_original, newline='',errors='ignore', encoding="windows-1252") as in_file:
        with open(file_path_output, 'w', newline='', encoding="windows-1252") as out_file:
            writer = csv.writer(out_file)
            for row in csv.reader(in_file):
                if row:
                    writer.writerow(row)



file_path_formatted = r"Database/CSVS/Capitals-No-Spaces.csv"
# sparql_query_for_capitals()
format_file(file_path_formatted)
