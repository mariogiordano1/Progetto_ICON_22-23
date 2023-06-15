import sys

import pandas as pd
from pathlib2 import Path
from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE, RDFXML, RDF, TSV, CSV


def sparql_query():
    file = open("Database/TXTS/Countries_No_Duplicate.txt", 'r', newline="\n")

    file_to_list = file.read().splitlines()

    for cityname in file_to_list:
        endpoint_url = "https://dbpedia.org/sparql"
        file_output = open(f"""Database/CSVS/Database.csv""", 'a', newline="\n")
        print(file.readlines())
        query = f"""SELECT DISTINCT ?unilabel?lat ?lon ?countrylabel ?citylabel
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




sparql_query()