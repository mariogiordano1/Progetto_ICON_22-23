'''
Import SPARQL e CSV per restituire i risultati delle query in formato CSV
'''
from SPARQLWrapper import SPARQLWrapper, CSV, XML, JSON
import pandas as pd
import geocoder as gc

'''
Endpoint servizio per le query
'''
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
'''
Query da eseguire
'''
sparql.setQuery(
    """
SELECT ?item ?itemLabel
WHERE
{
    ?item wdt:P31 wd:Q146 .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
"""
)

'''
Setta il risultato delle query in CSV
'''
sparql.setReturnFormat(JSON)

results = sparql.query().convert()

#print(results)
g = gc.geonames('Taranto', key='Giovanni') # ti amo

print(g.geonames_id)

results_df = pd.json_normalize(results['results']['bindings'])
print(results_df[['item.value', 'itemLabel.value']].head())
'''
import sys
from SPARQLWrapper import SPARQLWrapper, JSON

endpoint_url = "https://query.wikidata.org/sparql"

query = """SELECT ?city ?label_en
WHERE
{
  ?city wdt:P1566 "3165926".
  ?city wdt:P17 ?countryLabel .
  ?countryLabel rdfs:label ?label_en filter (lang(?label_en) = "en").

}
"""


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


results = get_results(endpoint_url, query)

for result in results["results"]["bindings"]:
    print(result)

'''