import sys

import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE, RDFXML, RDF, TSV
file = open("Database/city.txt", 'r', newline="\n")
file_to_list = file.read().splitlines()


for cityname in file_to_list:
    endpoint_url = "https://query.wikidata.org/sparql"
    file_output = open("Database/tester.txt", 'a', newline="\n")
    print(file.readlines())
    query = f"""SELECT ?label_en
        WHERE
        {{
          ?city wdt:P1566 "{cityname}".
          ?city wdt:P17 ?countryLabel .
          ?countryLabel rdfs:label ?label_en filter (lang(?label_en) = "en").
        }}""" + """ -H "Accept: text/csv" """
    user_agent = "GeoDataBotPython/0.1 /%s.%s" % (sys.version_info[0], sys.version_info[1])

    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)

    file_output.write(str(sparql.query().convert()) + "\n")
