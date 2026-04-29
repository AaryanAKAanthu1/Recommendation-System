from utils import similarity, encode, decode
import numpy as np

class Recommender:
    def __init__(self):
        pass

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def mean_weight(self, movies):
        weights = np.ones(len(movies))
        weights = self.softmax(weights)
        embeddings = np.mean(movies, axis=0) 

        return weights * embeddings
    
    def recommend(self, movies_watched):
        watched_embeddings = encode.encode(movies_watched)
        watched_embeddings = self.mean_weight(watched_embeddings)
        recommendations = similarity.scan(watched_embeddings)
        recommendations = [decode.decode(recommendations[i][0]) for i in range(0, len(recommendations))]

        return recommendations

        

