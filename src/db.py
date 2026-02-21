# src/db.py
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pathlib import Path

# Get project root (works from any file in src/)
PROJECT_ROOT = Path('/Users/dhirkatre/code/movie-recommender')  # 2 levels up from src/db.py → root

DATABASE_URL = f"sqlite:///{PROJECT_ROOT / 'data' / 'movies.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # needed for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Movie(Base):
    __tablename__ = "movies"

    movie_id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=False)
    genres = Column(String, nullable=True)
    overview = Column(String, nullable=True)

    # Relationship to ratings
    ratings = relationship("Rating", back_populates="movie")

class Rating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.movie_id"), index=True, nullable=False)
    rating = Column(Float, nullable=False)

    # Relationship back to movie
    movie = relationship("Movie", back_populates="ratings")

# src/db.py (add at the end)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()