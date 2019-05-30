import matplotlib.pyplot as plt

import pandas as pd
import re

def plot(data, xlabel='x label', ylabel='y label', fig_name='figure'):

    for line in data:
        plt.plot(line[0], line[1], label=line[2])
        plt.plot(line[0], line[1], 'xk', label='_nolegend_')
    
    plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(f'visualization/figures/{fig_name}.png')
    plt.show()


GRID_TYPE = 'uniform'
CITIES = [('manhattan', 'Manhattan, NY'), ('la', 'Los Angeles, CA'), ('paris', 'Paris')]

if __name__ == '__main__':
    # data = []
    # xlabel = 'Cell size in kilometers'
    # fig_name='threshold_test/uniform'

    # for city in CITIES:
    #     df = pd.read_csv(f'neural/output/{GRID_TYPE}/{city[0]}.csv')
        
    #     x = [float(re.search('density-(.+?)-', item[1].modelName).group(1)) for item in df.iterrows()]
    #     y = [item[1].medianErrorDistanceInMeters for item in df.iterrows()]

    #     data.append([x,y,city[1]])

    df = pd.read_csv(f'neural/output/adaptive/paris.csv')
    data = []
    xlabel='Leafsize of each bucket'
    fig_name='threshold_test/adaptive/paris'

    for row, item in df.iterrows():
        tweets_limit = int(re.search('leafsize-(.+?)-tweetslimit', item.modelName).group(1))
        leafsize = int(re.search('classifier-(.+?)-leafsize', item.modelName).group(1))
        
        label = f'Tweets limit {tweets_limit}'

        index = [index for index, line in enumerate(data) if line[2] == label]
        if not index:
            data.append([[],[],label])
            index = len(data) - 1
        else:
            index = index[0]

        data[index][0].append(leafsize)
        data[index][1].append(item.medianErrorDistanceInMeters)

    plot(data, xlabel=xlabel, ylabel='Median error distance in meters', fig_name=fig_name)