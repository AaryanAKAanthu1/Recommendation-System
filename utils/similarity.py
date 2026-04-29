import numpy as np


path = './data/shows_embeddings.npy'
data = np.load(path)

def similarity(embeddings1, embeddings2):
    return np.dot(embeddings1, embeddings2)/ (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))

def sort(inpList):
    SortedData = sorted(zip(inpList['idx'], inpList['score']), key=lambda x: x[1],reverse=True)

def scan(embeddings1, threshold=0.60):
    recommendations = {'idx':[], 'score':[]}

    for idx in range(0, len(data)):
        score = similarity(embeddings1, data[idx])

        if score >= threshold:
            recommendations['idx'].append(idx)
            recommendations['score'].append(score)

    return sort(recommendations)


   

if __name__ == '__main__':
    print(data.shape)