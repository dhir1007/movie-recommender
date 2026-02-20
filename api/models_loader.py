# api/models_loader.py
from pathlib import Path
import joblib
import faiss
import numpy as np
import pandas as pd

MODELS_DIR = Path('../models')

# Global variables (loaded once)
collab_model = None
content_embeddings = None
faiss_index = None
movie_df = None
item_codes_map = None

def load_all_models():
    global collab_model, content_embeddings, faiss_index, movie_df, item_codes_map
    
    print("Loading models...")
    
    # Collaborative (implicit ALS)
    collab_model = joblib.load(MODELS_DIR / 'collab_als.joblib')
    
    # Content-based
    content_embeddings = np.load(MODELS_DIR / 'content_embeddings.npy')
    faiss_index = faiss.read_index(str(MODELS_DIR / 'content_faiss.index'))
    
    # Movies DF
    movie_df = pd.read_csv('../data/processed/movies_with_plots.csv')
    
    # ALS item mapping
    ratings = pd.read_csv('../data/raw/ml-latest-small/ratings.csv')
    item_codes = ratings['movieId'].astype('category').cat.categories
    item_codes_map = dict(enumerate(item_codes))
    
    print("All models loaded successfully")