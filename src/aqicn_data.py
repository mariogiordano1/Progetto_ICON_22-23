from collections import namedtuple

import aqilib.aqicn
import aqilib.utils
from src.aqilib import aqicn, utils

api = aqicn.AqicnApi(secret="12f1f5758ff0027428dd7a64bbe1c50c7b10206b")
ip_based_data = api.get_feed()
utils.scrap_data_from_website()
Coordinate = namedtuple('Coordinate', ['lat', 'lng'])
api.get_location_feed(Coordinate(lat=10.3, lng=20.7))