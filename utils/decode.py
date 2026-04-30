import pandas as pd

data = pd.read_csv("./data/netflix_shows.csv")

def decode(idx):
    return data.iloc[idx]

if __name__ == '__main__':
    print(data)
    idx = int(input("index: "))
    print(decode(idx))