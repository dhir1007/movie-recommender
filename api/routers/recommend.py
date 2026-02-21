from fastapi import APIRouter, Query, HTTPException, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List, Dict

from api.models_loader import load_all_models

from src.models.hybrid import get_hybrid_recommendations

router = APIRouter(prefix="/api", tags=["recommendations"])

limiter = Limiter(key_func=get_remote_address)

@router.get("/recommend")
@limiter.limit("30/minute")
async def recommend(
    request: Request,
    user_id: int = Query(..., ge=1, description="MovieLens user ID"),
    n: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    models: dict = Depends(load_all_models)
):
    collab_model = models['collab_model']
    content_embeddings = models['content_embeddings']
    faiss_index = models['faiss_index']
    movie_df = models['movie_df']
    item_codes_map = models['item_codes_map']
    if collab_model is None or content_embeddings is None or faiss_index is None or movie_df is None or item_codes_map is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    print(f"Request received for user {user_id} with {n} recommendations")
    try:
        recs = get_hybrid_recommendations(user_id=user_id, n=n, collab_model=collab_model, content_embeddings=content_embeddings, faiss_index=faiss_index, movie_df=movie_df, item_codes_map=item_codes_map)
        print(f"Hybrid recommendations generated: ", len(recs))
        
        result = []
        for movie_id, hybrid_score in recs:
            print(f"Movie ID: {movie_id}, Hybrid score: {hybrid_score}")
            movie = movie_df[movie_df['movieId'] == movie_id]
            if movie.empty:
                print(f"Movie ID {movie_id} not found")
                continue
            movie = movie.iloc[0]
            result.append({
                "movieId": int(movie_id),
                "title": movie['title'],
                "genres": movie['genres'],
                "hybrid_score": float(round(hybrid_score, 4))
            })
        
        return {
            "user_id": user_id,
            "n_requested": n,
            "recommendations": result
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error generating recommendations in /recommend: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar")
@limiter.limit("30/minute")
async def similar(
    request: Request,
    movie_id: int = Query(..., description="MovieLens movie ID"),
    n: int = Query(10, ge=1, le=50, description="Number of similar movies"),
    models: dict = Depends(load_all_models)
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
        
        # Skip self
        distances = distances[0][1:]
        indices = indices[0][1:]
        
        result = []
        for idx, score in zip(indices, distances):
            movie = movie_df.iloc[idx]
            result.append({
                "movieId": int(movie['movieId']),
                "title": movie['title'],
                "genres": movie['genres'],
                "similarity_score": float(round(score, 4))
            })
        
        return {
            "movie_id": movie_id,
            "similar_movies": result
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/health")
async def health(models: dict = Depends(load_all_models)):
    """Health check - confirms models are loaded."""
    status = {
        "status": "healthy",
        "models_loaded": {
            "collaborative": models['collab_model'] is not None,
            "content_embeddings": models['content_embeddings'] is not None,
            "faiss_index": models['faiss_index'] is not None,
            "movie_df": models['movie_df'] is not None,
            "item_codes_map": models['item_codes_map'] is not None
        }
    }
    return status

# ... existing imports at top ...
from fastapi import Depends
from sqlalchemy.orm import Session
from src.db import get_db, Rating
from pydantic import BaseModel

class RatingCreate(BaseModel):
    user_id: int
    movie_id: int
    rating: float

@router.post("/rate")
@limiter.limit("10/minute")
async def rate_movie(
    rating: RatingCreate,
    db: Session = Depends(get_db),
    request: Request = None  # for limiter
):
    """
    Submit a new rating (saved to SQLite DB).
    """
    if not 0.5 <= rating.rating <= 5.0:
        raise HTTPException(status_code=400, detail="Rating must be between 0.5 and 5.0")

    db_rating = Rating(
        user_id=rating.user_id,
        movie_id=rating.movie_id,
        rating=rating.rating
    )
    db.add(db_rating)
    db.commit()
    db.refresh(db_rating)

    return {
        "message": "Rating saved successfully",
        "rating_id": db_rating.id,
        "user_id": db_rating.user_id,
        "movie_id": db_rating.movie_id,
        "rating": db_rating.rating
    }