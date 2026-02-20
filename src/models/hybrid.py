import numpy as np
import pandas as pd
from api.main import collab_model, content_embeddings, faiss_index, movie_df, item_codes_map
from implicit.als import AlternatingLeastSquares

def get_hybrid_recommendations(user_id: int, n: int = 10, alpha: float = 0.6):
    """
    Hybrid recommendation:
    alpha * collaborative_score (implicit ALS) + (1-alpha) * content_score (embeddings)
    """
    # --- 1. Collaborative scores (ALS) ---
    # Get internal user ID
    ratings = pd.read_csv('../data/raw/ml-latest-small/ratings.csv')  # load only when needed
    user_mask = ratings['userId'] == user_id
    if not user_mask.any():
        # Cold-start user → fallback to popular
        popular = ratings.groupby('movieId')['rating'].mean().nlargest(50)
        collab_scores = {mid: 0.5 for mid in popular.index}
    else:
        user_internal = ratings[user_mask]['userId'].astype('category').cat.codes.iloc[0]
        user_row = csr_matrix(([], ([], [])), shape=(len(ratings['userId'].unique()), len(ratings['movieId'].unique())))
        # Note: in real code, you'd cache this sparse matrix
        # Here we approximate by recommending many items
        als_items, als_scores = collab_model.recommend(
            user_internal, user_row[user_internal],
            N=n*10, filter_already_liked_items=True
        )
        collab_scores = {item_codes_map[iid]: score for iid, score in zip(als_items, als_scores)}
    
    # --- 2. Content-based scores ---
    user_rated = ratings[user_mask]
    if len(user_rated) == 0:
        # Cold-start content: popular + random
        content_scores = {mid: 0.5 for mid in movie_df['movieId'].sample(50)}
    else:
        top_rated_ids = user_rated.nlargest(5, 'rating')['movieId'].values
        top_indices = movie_df[movie_df['movieId'].isin(top_rated_ids)].index
        user_emb = np.mean(content_embeddings[top_indices], axis=0).reshape(1, -1).astype('float32')
        distances, indices = faiss_index.search(user_emb, n*10)
        content_scores = {movie_df.iloc[idx]['movieId']: dist for idx, dist in zip(indices[0], distances[0])}
    
    # --- 3. Blend ---
    all_movie_ids = set(collab_scores.keys()) | set(content_scores.keys())
    hybrid_scores = {}
    for mid in all_movie_ids:
        c_score = collab_scores.get(mid, 0)
        cont_score = content_scores.get(mid, 0)
        hybrid_scores[mid] = alpha * c_score + (1 - alpha) * cont_score
    
    # Sort & return top N
    sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return sorted_recs  # list of (movieId, hybrid_score)