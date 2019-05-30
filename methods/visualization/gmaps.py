import pandas as pd
import csv
from gmplot import gmplot

from neural.single.preprocess import read_dataset, read_paris_dataset, read_london_dataset
from neural.single.grid import read_grid_cells

API_KEY = ''

##### Cities #####

manhattan = {
    'name': 'Manhattan, NY', 'lat': 40.730610,
    'lon': -73.935242, 'file_name': 'manhattan_tweets.html'
}

los_angeles = {
    'name': 'Los Angeles, CA', 'lat': 34.052235,
    'lon': -118.243683, 'file_name': 'los_angeles.html'
}

london = {
    'name': 'London', 'lat': 51.509865,
    'lon': -0.118092, 'file_name': 'london.html'
}

paris = {
    'name': 'Paris', 'lat': 48.864716,
    'lon': 2.349014, 'file_name': 'paris.html'
}

city = london

# Read data to visualize

data = read_london_dataset('./datasets/london/tweets.tsv', grid=False)
grid_cells = read_grid_cells('./datasets/london/grid.tsv')

latitudes = data.latitude.values
longitudes = data.longitude.values


gmap = gmplot.GoogleMapPlotter(city['lat'], city['lon'], 10, API_KEY)

# Place grid cells
for cell in grid_cells:
    cell_lat = [cell.nw()[1], cell.ne[1], cell.se()[1], cell.sw[1]]
    cell_lon = [cell.nw()[0], cell.ne[0], cell.se()[0], cell.sw[0]]
    gmap.polygon(cell_lat, cell_lon, color='#4d94ff')

# Place tweets
gmap.scatter(latitudes, longitudes, '#3B0B39', size=10, marker=False)
gmap.heatmap(latitudes, longitudes, radius=40)

gmap.draw(f'visualization/maps/{city['file_name']}')
