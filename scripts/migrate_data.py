# scripts/migrate_data.py
# Run once: python scripts/migrate_data.py
from src.db import Base, engine, SessionLocal, Movie, Rating
import pandas as pd
from pathlib import Path

print("Creating tables...")
Base.metadata.create_all(bind=engine)
print("Tables created.")

# Paths (relative to project root)
processed_movies = Path('/Users/dhirkatre/code/movie-recommender/data/processed/movies_with_plots.csv')
raw_ratings = Path('/Users/dhirkatre/code/movie-recommender/data/raw/ratings.csv')

print("Loading CSVs...")
movies_df = pd.read_csv(processed_movies)
ratings_df = pd.read_csv(raw_ratings)

db = SessionLocal()
try:
    print("Inserting movies...")
    for _, row in movies_df.iterrows():
        movie = Movie(
            movie_id=int(row['movieId']),
            title=str(row['title']),
            genres=str(row['genres']),
            overview=str(row.get('overview', ''))
        )
        db.add(movie)
    db.commit()
    print(f"Inserted {len(movies_df)} movies")

    print("Inserting ratings...")
    for _, row in ratings_df.iterrows():
        rating = Rating(
            user_id=int(row['userId']),
            movie_id=int(row['movieId']),
            rating=float(row['rating'])
        )
        db.add(rating)
    db.commit()
    print(f"Inserted {len(ratings_df)} ratings")

    print("Migration complete!")
except Exception as e:
    db.rollback()
    print("Migration failed:", str(e))
finally:
    db.close()