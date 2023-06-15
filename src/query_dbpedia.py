import sys

import pandas as pd
from pathlib2 import Path

from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE, RDFXML, RDF, TSV, CSV

file = open("Database/TXTS/Countries_No_Duplicate.txt", 'r', newline="\n")

file_to_list = file.read().splitlines()


def sparql_query_for_capitals():
    for countryname in file_to_list:
        endpoint_url = "https://dbpedia.org/sparql"
        file_output = open(f"""Database/CSVS/Capitals.csv""", 'a', newline="\n", encoding="UTF-8")
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
    file = Path(file_path)
    data = file.read_text()
    data = data.replace("b'", "")
    data = data.replace("citylabel", "")
    data = data.replace('\\n', "")
    data = data.replace('\'', "")
    file.write_text(data)


filePath = r"Database/CSVS/Capitals.csv"
#sparql_query_for_capitals()
format_file(filePath)
