import numpy as np

data = np.load('./data/netflix_shows.csv')

def decode(idx):
    return data['train'][idx]