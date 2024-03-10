import pandas as pd
import geopandas
import geoplot
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import warnings


def get_dummy_europe():
    polygon = Polygon([(-25,35), (40,35), (40,75),(-25,75)])
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    europe=world[world.continent=="Europe"]
    europe=geopandas.clip(europe, polygon)
    return europe