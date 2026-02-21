# api/models_loader.py
from pathlib import Path
import joblib
import faiss
import numpy as np
import pandas as pd

PROJECT_ROOT = Path('/Users/dhirkatre/code/movie-recommender')  # ← YOUR FULL PATH HERE
MODELS_DIR = PROJECT_ROOT / 'models'

# Global variables - populated by load_all_models()
collab_model = None
content_embeddings = None
faiss_index = None
movie_df = None
item_codes_map = None

def load_all_models():

    
    print("Loading collab_als.joblib...")
    collab_model = joblib.load(MODELS_DIR / 'collab_als.joblib')
    print("Collab loaded")
    
    print("Loading content_embeddings.npy...")
    content_embeddings = np.load(MODELS_DIR / 'content_embeddings.npy')
    print("Embeddings loaded")
    
    print("Loading content_faiss.index...")
    faiss_index = faiss.read_index(str(MODELS_DIR / 'content_faiss.index'))
    print("FAISS loaded")
    
    print("Loading movie_df...")
    movie_df = pd.read_csv(PROJECT_ROOT / 'data/processed/movies_with_plots.csv')
    print("movie_df loaded")
    
    print("Loading item_codes_map...")
    ratings = pd.read_csv(PROJECT_ROOT / 'data/raw/ratings.csv')
    item_codes = ratings['movieId'].astype('category').cat.categories
    item_codes_map = dict(enumerate(item_codes))
    print("item_codes_map loaded")
    
    print("All models loaded successfully")
    return {
        'collab_model': collab_model,
        'content_embeddings': content_embeddings,
        'faiss_index': faiss_index,
        'movie_df': movie_df,
        'item_codes_map': item_codes_map
    }