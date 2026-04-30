A Recommendation system based on Cosine Similarity Search<br><br>
logic:<br>
  watched_movies_embeddings = softmax(1xn ones matrix) * watched_movies_embeddings<br>
  similarity_score = cosine(watched_movies_embeddings, movie_emb) for movie_emb in total_movies_embeddings<br>
<br>
  recommend movie if similarity_score >= threshold
  
