import sys

import pandas as pd
from pathlib2 import Path
import csv
from SPARQLWrapper import SPARQLWrapper, CSV


def sparql_query():
    with open('Database/CSVS/Capitals-Complete.csv', 'r', encoding="windows-1252") as f_in, open(
            "Database/TXTS/Capitals-Complete.txt", "w", encoding="windows-1252") as f_out:
        content = f_in.read().replace(',', ' ')
        f_out.write(content)

    file = open("Database/TXTS/Capitals-Complete.txt", 'r', newline="\n")

    file_to_list = file.read().splitlines()

    for cityname in file_to_list:
        endpoint_url = "https://dbpedia.org/sparql"
        file_output = open(f"""Database/CSVS/Universities_OnlyCoords.csv""", 'a', newline="\n")
        print(file.readlines())
        query = f"""SELECT DISTINCT ?lat ?lon
            WHERE
            {{
              ?univ rdf:type dbo:University.
              ?univ dbo:city dbr:{cityname}.
              ?univ geo:lat ?lat.
              ?univ geo:long ?lon.
              ?univ dbo:city ?city.
              ?city dbo:country ?country.
              ?country rdfs:label ?countrylabel.
              ?city rdfs:label ?citylabel.
              ?univ rdfs:label ?unilabel
              FILTER(LANG(?countrylabel) = "en").
              FILTER(LANG(?citylabel) = "en").
              FILTER(LANG(?unilabel) = "en").
            }}"""

        user_agent = "GeoDataBotPython/0.1 /%s.%s" % (sys.version_info[0], sys.version_info[1])

        sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
        sparql.setReturnFormat(CSV)
        sparql.setQuery(query)

        file_output.write(str(sparql.query().convert()) + "\n")


def define_rows(file_path):
    file = Path(file_path)
    data = file.read_text()
    data = data.replace("\\n", "\n")
    data = data.replace("b\'\"lat\",\"lon\"", "")
    data = data.replace('\"b\'\"\"lat\"\"\",lon', "")
    data = data.replace("\'", "")
    file.write_text(data)
    file_output = "Database/CSVS/Universities_Cleaned.csv"
    remove_blank_spaces(file_path, file_output)


def remove_blank_spaces(file_path_original, file_path_output):
    with open(file_path_original, newline='', errors='ignore', encoding="windows-1252") as in_file:
        with open(file_path_output, 'w', newline='', encoding="windows-1252") as out_file:
            writer = csv.writer(out_file)
            for row in csv.reader(in_file):
                if row:
                    writer.writerow(row)


def remove_duplicates(file_path):
    df = pd.read_csv(file_path, header=None, names=["lat", "long"], encoding="windows-1252")
    df = df.drop_duplicates()
    df.to_csv("Database/CSVS/Universities_No_Dupes.csv", index=False)


#sparql_query()
define_rows("Database/CSVS/Universities_OnlyCoords.csv")
remove_duplicates("Database/CSVS/Universities_Cleaned.csv")
