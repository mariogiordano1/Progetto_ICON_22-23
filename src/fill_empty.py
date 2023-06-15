'''
Import SPARQL e CSV per restituire i risultati delle query in formato CSV
'''
from SPARQLWrapper import SPARQLWrapper, CSV, XML, JSON
import pandas as pd
import geocoder as gc
from geocoder import geonames
file = open('Database/Null-Solo-Città.txt','r')
with open('Database/city.txt', 'w') as f:
    for i in file.readlines():
        f.write(str(geonames(i, key='GiovanniSecondo').geonames_id))
        f.write("\n")
        print(i)

topology_list = file.readlines()
print(topology_list)

'''
Prende i geodata id from geodata database
'''
g = gc.geonames('Granville', key='GiovanniSecondo') # ti amo



