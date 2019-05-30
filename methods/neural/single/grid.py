import pandas as pd
import numpy as np
import csv
import requests
import math

from scipy.spatial import cKDTree

from neural.single.preprocess import timer, read_dataset, read_paris_dataset, read_london_dataset, custom_train_test_split

##### cKDTree throws RecursionError #####
import sys
sys.setrecursionlimit(10000)

ONE_DEGREE_IN_KM = 111

class GridCell():
    __last_id = 0

    def __init__(self, south_west, north_east, gcid, coordinates_count=None):
        self.gcid = gcid  # gcid - grid cell id

        self.sw = south_west
        self.ne = north_east

        self.coordinates_count = coordinates_count
        GridCell.__last_id += 1

    def contains_point(self, long, lat):
        return long >= self.sw[0] and long <= self.ne[0] and lat >= self.sw[1] and lat <= self.ne[1]

    def center(self):
        return (self.sw[1] + self.ne[1]) / 2, (self.sw[0] + self.ne[0]) / 2

    def se(self):
        return (self.ne[0], self.sw[1])

    def nw(self):
        return (self.sw[0], self.ne[1])

    def __str__(self):
        return '{self.gcid}, {self.sw}, {self.ne}, {self.coordinates_count}'.format(**locals())


class KDTreeGrid():
    def __init__(self, coordinates, leafsize=400, tweets_limit=10):
        self.kd_tree = cKDTree(coordinates, leafsize=leafsize, balanced_tree=False)
        self.tree = self.kd_tree.tree
        self.tweets_limit = tweets_limit
        self.leaf_nodes = []
        self.gcid = 0
        self.get_leaves(self.leaf_nodes, self.tree)

    def get_leaves(self, leaf_nodes, tree):
        if tree.greater is None and tree.lesser is None:
            if tree.children >= self.tweets_limit:
                leaf_nodes.append(
                    GridCell(*bounding_box(tree.data_points), self.gcid, tree.children))
                self.gcid += 1

        else:
            if tree.greater:
                self.get_leaves(leaf_nodes, tree.greater)
            if tree.lesser:
                self.get_leaves(leaf_nodes, tree.lesser)


def bounding_box(points):
    lat, long = zip(*points)
    return [(min(long), min(lat)), (max(long), max(lat))]


def bounding_box_from_points(latitudes, longitudes):
    return [min(latitudes), max(latitudes), min(longitudes), max(longitudes)]


def bounding_box_from_query(query):  # query should typically be '<name>, <county>, <country>'
    result = requests.get(
        url='https://nominatim.openstreetmap.org/search?q={query}&format=json&polygon=0&addressdetails=0'.format(**locals())).json()[0]
    bbox = list(map(float, result['boundingbox']))
    return bbox


def create_grid(bbox, density_in_km):
    #### bbox = [south Latitude, north Latitude, west Longitude, east Longitude] ######
    latitude_len = ONE_DEGREE_IN_KM * abs(bbox[1] - bbox[0])
    longitude_len = ONE_DEGREE_IN_KM * \
        math.cos(math.radians(bbox[0] + (bbox[1] - bbox[0]))) * abs(bbox[3] - bbox[2])

    nx = round(longitude_len/density_in_km)
    ny = round(latitude_len/density_in_km)

    x = np.linspace(bbox[2], bbox[3], nx if nx > 1 else 2)
    y = np.linspace(bbox[0], bbox[1], ny if ny > 1 else 2)

    grid = []
    gcid = 0
    for i in range(0, len(x) - 1):
        for j in range(0, len(y) - 1):
            cell = GridCell((x[i], y[j]), (x[i+1], y[j+1]), gcid)
            grid.append(cell)
            gcid += 1

    return grid


def write_grid_file(path, grid):
    with open(path, mode='w', newline='') as grid_file:
        grid_file_writer = csv.writer(grid_file, delimiter='\t', quotechar='"')
        for cell in grid:
            grid_file_writer.writerow(
                [int(cell.gcid), cell.sw[1], cell.sw[0], cell.ne[1], cell.ne[0]])


def cell_tweet_count(dataset, cell):
    count = 0
    for index, row in dataset.iterrows():
        if cell.contains_point(row['longitude'], row['latitude']):
            count += 1
    return count


@timer
def remove_empty_cells(dataset, grid):
    vfind_gcid = np.vectorize(find_gcid)
    gcids = vfind_gcid([[cell] for cell in grid], dataset['longitude'].values, dataset['latitude'].values)

    indexes_to_remove = []
    new_gcid = 0
    for index, gcid in enumerate(gcids):
        if gcid.tolist().count(None) != len(gcid):
            dataset.loc[dataset.gcid == gcid, 'gcid'] = new_gcid
            grid[index].gcid = new_gcid
            new_gcid += 1
        else:
            indexes_to_remove.append(index)

    for i in reversed(indexes_to_remove):
        del grid[i]

    return dataset, grid


@timer
def generate_gcid_dataset(data, grid):
    vfind_gcid = np.vectorize(find_gcid)
    gcids = vfind_gcid([[cell] for cell in grid], data['longitude'].values, data['latitude'].values)  
    gcids = np.transpose(gcids, (1, 0))
    result = []
    for gcid in gcids:
        if gcid.tolist().count(None) == len(gcid):
            result.append(None)
        else:
            result.append(gcid[gcid != np.array(None)][0])

    data.insert(0, 'gcid', result)

    data_none_nan = data.dropna(subset=['gcid']).copy()
    data_none_nan.loc[:, 'gcid'] = data_none_nan.loc[:, 'gcid'].astype(int)

    print('Data count:', len(data.values.tolist()))
    print('Data none nan count:', len(data_none_nan.values.tolist()))

    return data_none_nan


def find_gcid(cell, long, lat):
    if cell.contains_point(long, lat):
        return cell.gcid


def read_grid_cells(filename):
    columns = [
        "gcid",
        "sw_lat",
        "sw_long",
        "ne_lat",
        "ne_long",
    ]
    cells = pd.read_csv(filename, sep="\t", names=columns)
    grid = []
    for cell in cells.values:
        grid.append(GridCell((cell[2], cell[1]), (cell[4], cell[3]), cell[0]))
    return grid


ADAPTIVE = True
LEAF_SIZE = 200
TWEETS_LIMIT = 10

CITY = 'london'
DENSITY = 1 # in km


if __name__ == '__main__':
    dataset = read_london_dataset('datasets/london.tsv')
    save_path = f'datasets/grid/adaptive/{CITY}'

    if ADAPTIVE:
        tree_grid = KDTreeGrid(dataset[['latitude', 'longitude']].values, leafsize=LEAF_SIZE, tweets_limit=TWEETS_LIMIT)
        grid = tree_grid.leaf_nodes
    else:
        bbox = bounding_box_from_query('London, England')
        grid = create_grid(bbox, DENSITY)

    dataset_gcid = generate_gcid_dataset(dataset.copy(), grid)
    # dataset_gcid, grid = remove_empty_cells(dataset_gcid.copy(), grid)

    ### Generate training and test set ###
    tweets_train, tweets_test = custom_train_test_split(dataset_gcid.copy())

    ### Write files ###
    tweets_train.to_csv(f'{save_path}/tweets.tsv', index=False, header=False, sep='\t')
    tweets_test.to_csv(f'{save_path}/tweets_test.tsv', index=False, header=False, sep='\t')
    write_grid_file(f'{save_path}/grid.tsv', grid)
            