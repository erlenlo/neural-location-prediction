import numpy as np
import gmplot

from neural.single.grid import read_grid_cells

grid_cells = read_grid_cells('./datasets/london/grid.tsv')

API_KEY = ''

gmap = gmplot.GoogleMapPlotter(grid_cells[0].sw[1], grid_cells[0].sw[0], 10, API_KEY)

for cell in grid_cells:
    latitudes = [cell.nw()[1], cell.ne[1], cell.se()[1], cell.sw[1]]
    longitudes = [cell.nw()[0], cell.ne[0], cell.se()[0], cell.sw[0]]
    gmap.polygon(latitudes, longitudes, color='#4d94ff')

gmap.draw(f"visualization/maps/london_grid.html")
