from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path
import joblib
import faiss
import numpy as np
import pandas as pd

from api.routers.recommend import router as recommend_router

MODELS_DIR = Path('../models')

# Global variables (loaded once at startup)
collab_model = None
content_embeddings = None
faiss_index = None
movie_df = None
item_codes_map = None  # for implicit ALS mapping

@asynccontextmanager
async def lifespan(app: FastAPI):
    global collab_model, content_embeddings, faiss_index, movie_df, item_codes_map
    
    print("Loading models...")
    
    # Collaborative (implicit ALS)
    collab_model = joblib.load(MODELS_DIR / 'collab_als.joblib')
    
    # Content-based (embeddings + FAISS)
    content_embeddings = np.load(MODELS_DIR / 'content_embeddings.npy')
    faiss_index = faiss.read_index(str(MODELS_DIR / 'content_faiss.index'))
    
    # Movies DataFrame for mapping IDs → titles/genres
    movie_df = pd.read_csv('../data/processed/movies_with_plots.csv')
    
    # ALS item ID mapping (from Day 5)
    ratings = pd.read_csv('../data/raw/ml-latest-small/ratings.csv')
    item_codes = ratings['movieId'].astype('category').cat.categories
    item_codes_map = dict(enumerate(item_codes))
    
    print("All models loaded successfully")
    yield
    
    # Optional cleanup
    print("Shutting down - clearing models")
    collab_model = None
    content_embeddings = None
    faiss_index = None
    movie_df = None
    item_codes_map = None

app = FastAPI(
    title="Hybrid Movie Recommender API",
    description="Personalized movie recommendations using hybrid content + collaborative filtering",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(recommend_router)