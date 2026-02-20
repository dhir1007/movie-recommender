from fastapi import APIRouter, Query, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List, Dict
import numpy as np

from api.main import collab_model, content_embeddings, faiss_index, movie_df, item_codes_map
from src.models.hybrid import get_hybrid_recommendations

router = APIRouter(prefix="/api", tags=["recommendations"])

limiter = Limiter(key_func=get_remote_address)

@router.get("/recommend")
@limiter.limit("30/minute")
async def recommend(
    user_id: int = Query(..., ge=1, description="User ID from MovieLens"),
    n: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """
    Get hybrid personalized recommendations for a user.
    Combines implicit ALS collaborative scores + content-based embeddings.
    """
    try:
        recs = get_hybrid_recommendations(user_id, n=n)
        
        result = []
        for movie_id, hybrid_score in recs:
            movie = movie_df[movie_df['movieId'] == movie_id]
            if movie.empty:
                continue
            movie = movie.iloc[0]
            result.append({
                "movieId": int(movie_id),
                "title": movie['title'],
                "genres": movie['genres'],
                "hybrid_score": round(hybrid_score, 4)
            })
        
        return {"user_id": user_id, "recommendations": result}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/similar")
@limiter.limit("30/minute")
async def similar(
    movie_id: int = Query(..., description="Movie ID to find similar movies for"),
    n: int = Query(10, ge=1, le=50, description="Number of similar movies")
):
    """
    Get content-based similar movies using embeddings + FAISS.
    """
    try:
        idx = movie_df[movie_df['movieId'] == movie_id].index
        if idx.empty:
            raise ValueError(f"Movie ID {movie_id} not found")
        idx = idx[0]
        
        query_vec = content_embeddings[idx].reshape(1, -1).astype('float32')
        distances, indices = faiss_index.search(query_vec, n + 1)
        
        # Skip self (first result)
        distances = distances[0][1:]
        indices = indices[0][1:]
        
        result = []
        for idx, score in zip(indices, distances):
            movie = movie_df.iloc[idx]
            result.append({
                "movieId": int(movie['movieId']),
                "title": movie['title'],
                "genres": movie['genres'],
                "similarity_score": round(score, 4)
            })
        
        return {"movie_id": movie_id, "similar_movies": result}
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/health")
async def health():
    """Check if API and models are loaded."""
    loaded = {
        "collab_model": collab_model is not None,
        "content_embeddings": content_embeddings is not None,
        "faiss_index": faiss_index is not None,
        "movie_df": movie_df is not None
    }
    status = all(loaded.values())
    return {"status": "healthy" if status else "degraded", "models": loaded}