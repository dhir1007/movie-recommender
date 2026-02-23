# api/models_loader.py
import joblib
import faiss
import numpy as np
import pandas as pd



# Global variables - populated by load_all_models()
collab_model = None
content_embeddings = None
faiss_index = None
movie_df = None
item_codes_map = None

def load_all_models():
    global collab_model, content_embeddings, faiss_index, movie_df, item_codes_map
    
    print("Loading collab_als.joblib...")
    collab_model = joblib.load('models/collab_als.joblib')
    print("Collab loaded")
    
    print("Loading content_embeddings.npy...")
    content_embeddings = np.load('models/content_embeddings.npy')
    print("Embeddings loaded")
    
    print("Loading content_faiss.index...")
    faiss_index = faiss.read_index('models/content_faiss.index')
    print("FAISS loaded")
    
    print("Loading movie_df...")
    movie_df = pd.read_csv('data/processed/movies_with_plots.csv')
    movie_df['movieId'] = movie_df['movieId'].astype(int)  
    print("movie_df loaded")
    
    print("Loading item_codes_map...")
    ratings = pd.read_csv('data/raw/ratings.csv')
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