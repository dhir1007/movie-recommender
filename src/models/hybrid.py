import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path

from api.models_loader import collab_model, content_embeddings, faiss_index, movie_df, item_codes_map

PROJECT_ROOT = Path('/Users/dhirkatre/code/movie-recommender')  
def get_hybrid_recommendations(user_id: int, n: int = 10, alpha: float = 0.6, collab_model=None, content_embeddings=None, faiss_index=None, movie_df=None, item_codes_map=None):
    """
    Hybrid score = alpha * collaborative (ALS) + (1-alpha) * content (embeddings)
    Returns list of (movie_id, hybrid_score) tuples, sorted descending.
    """
    if any(x is None for x in [collab_model, content_embeddings, faiss_index, movie_df, item_codes_map]):
        raise ValueError("Models not loaded")
    
    ratings = pd.read_csv(PROJECT_ROOT / 'data/raw/ratings.csv')
    user_rated = ratings[ratings['userId'] == user_id]

    # --- Collaborative scores (implicit ALS) ---
    if len(user_rated) == 0:
        # Cold-start: fallback to popular movies
        popular = ratings.groupby('movieId')['rating'].mean().nlargest(100)
        collab_scores = {mid: 0.5 for mid in popular.index}
    else:
        # Get internal user ID
        user_internal = user_rated['userId'].astype('category').cat.codes.iloc[0]
        
        # Get user's row (approximate - in real app, cache this sparse matrix)
        # For simplicity, recommend many items
        als_items, als_scores = collab_model.recommend(
            user_internal,
            csr_matrix(([], ([], [])), shape=(1, len(item_codes_map))),  # dummy row
            N=n*10,
            filter_already_liked_items=True
        )
        collab_scores = {item_codes_map[iid]: score for iid, score in zip(als_items, als_scores)}

    # --- Content-based scores ---
    if len(user_rated) == 0:
        content_scores = {mid: 0.5 for mid in movie_df['movieId'].sample(100)}
    else:
        top_rated_ids = user_rated.nlargest(5, 'rating')['movieId'].values
        top_indices = movie_df[movie_df['movieId'].isin(top_rated_ids)].index
        user_emb = np.mean(content_embeddings[top_indices], axis=0).reshape(1, -1).astype('float32')
        distances, indices = faiss_index.search(user_emb, n*10)
        content_scores = {movie_df.iloc[idx]['movieId']: dist for idx, dist in zip(indices[0], distances[0])}

    # --- Blend ---
    all_movie_ids = set(collab_scores.keys()) | set(content_scores.keys())
    hybrid_scores = {}
    for mid in all_movie_ids:
        c_score = collab_scores.get(mid, 0)
        cont_score = content_scores.get(mid, 0)
        hybrid_scores[mid] = alpha * c_score + (1 - alpha) * cont_score

    # Sort descending by score
    sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return sorted_recs