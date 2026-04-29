from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import numpy as np


model = SentenceTransformer('BAAI/bge-large-en-v1.5')
data = load_dataset('hugginglearners/netflix-shows')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# format data
data = data['train']

def format_show(row):
    def clean(val):
        return str(val).strip() if val is not None else ""

    parts = []
    if row.get('title'):       parts.append(f"Title: {clean(row['title'])}")
    if row.get('type'):        parts.append(f"Type: {clean(row['type'])}")
    if row.get('listed_in'):   parts.append(f"Genre: {clean(row['listed_in'])}")
    if row.get('description'): parts.append(f"Description: {clean(row['description'])}")
    if row.get('director'):    parts.append(f"Director: {clean(row['director'])}")
    if row.get('cast'):        parts.append(f"Cast: {clean(row['cast'])}")
    if row.get('country'):     parts.append(f"Country: {clean(row['country'])}")
    if row.get('release_year'):parts.append(f"Release Year: {clean(row['release_year'])}")
    if row.get('rating'):      parts.append(f"Rating: {clean(row['rating'])}")
    if row.get('duration'):    parts.append(f"Duration: {clean(row['duration'])}")

    return ". ".join(parts)


def encode(data):
    texts = [format_show(row) for row in data]
    instruction = "Represent this movie for semantic search:"
    embeddings = model.encode(
        texts,
        prompt= instruction,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        device= device

    )
    return embeddings

def save_encodings(encodings):
    np.save('./data/shows_embeddings.py')
    

if __name__ == '__main__':
    encodings = encode(data=data)
    save_encodings(encodings=encodings)

