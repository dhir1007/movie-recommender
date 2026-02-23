# Hybrid Movie Recommender API

A production-ready movie recommendation REST API that combines **content-based filtering** (sentence embeddings + FAISS) with **collaborative filtering** (implicit ALS) into a single hybrid engine. Built end-to-end: data pipeline, ML models, FastAPI service, SQLite persistence, Docker packaging, and CI/CD.

> **Live Swagger UI**: _coming soon on Render_

---

## Architecture

```
┌──────────────┐      ┌──────────────────────────────────────────────┐
│   Client     │─────▶│  FastAPI  (uvicorn, rate-limited via slowapi)│
│  (Swagger /  │      │                                              │
│   cURL /     │◀─────│  /api/recommend   /api/similar   /api/rate   │
│   Frontend)  │      │  /api/health      GET /                      │
└──────────────┘      └──────────┬───────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │    Hybrid Engine         │
                    │  α · collab + (1-α) · content │
                    └─────┬──────────┬────────┘
                          │          │
               ┌──────────┘          └──────────┐
               ▼                                ▼
    ┌─────────────────┐              ┌─────────────────────┐
    │ Collaborative    │              │ Content-Based        │
    │ implicit ALS     │              │ Sentence-Transformers│
    │ (user × item     │              │ all-MiniLM-L6-v2     │
    │  matrix factor.) │              │ + FAISS (L2 ANN)     │
    └─────────────────┘              └─────────────────────┘
               │                                │
               ▼                                ▼
    ┌─────────────────┐              ┌─────────────────────┐
    │ ratings.csv      │              │ content_embeddings   │
    │ (100k ratings)   │              │ .npy + .index        │
    └─────────────────┘              └─────────────────────┘
```

---

## Features

- **Hybrid recommendations** — blends collaborative and content signals with a tunable alpha weight, plus cold-start fallback for new users
- **Content similarity** — find movies similar to any title using semantic embeddings and approximate nearest-neighbor search
- **User ratings** — POST new ratings that persist to a SQLite database (SQLAlchemy ORM)
- **Rate limiting** — per-IP throttling via slowapi (30 req/min for reads, 10 req/min for writes)
- **Health check** — verifies all five model artifacts are loaded and reports status
- **Dockerized** — multi-stage build (builder + slim runtime) with OpenBLAS and OpenMP support
- **CI/CD** — GitHub Actions pipeline: lint (Ruff), test (pytest), Docker build
- **Interactive docs** — auto-generated Swagger UI at `/docs`

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI, Pydantic v2, Uvicorn |
| **ML — Content** | Sentence-Transformers (`all-MiniLM-L6-v2`), FAISS (CPU), scikit-learn TF-IDF |
| **ML — Collaborative** | implicit (ALS matrix factorization) |
| **ML — Hybrid** | Custom weighted blending with cold-start handling |
| **Data** | Pandas, NumPy, SciPy (sparse matrices) |
| **Database** | SQLAlchemy + SQLite (Postgres-ready) |
| **Infrastructure** | Docker (multi-stage), Docker Compose, GitHub Actions |
| **Quality** | pytest, Ruff, slowapi |
| **Data Source** | [MovieLens Latest Small](https://grouplens.org/datasets/movielens/) (~100k ratings, ~9k movies) + TMDB plot overviews |

---

## API Endpoints

### `GET /api/recommend`

Personalized hybrid recommendations for a user.

```bash
curl "http://localhost:8000/api/recommend?user_id=1&n=5"
```

```json
{
  "user_id": 1,
  "n_requested": 5,
  "recommendations": [
    {
      "movieId": 296,
      "title": "Pulp Fiction (1994)",
      "genres": "Comedy|Crime|Drama|Thriller",
      "hybrid_score": 0.8723
    }
  ]
}
```

### `GET /api/similar`

Content-based similar movies via embedding + FAISS lookup.

```bash
curl "http://localhost:8000/api/similar?movie_id=1&n=5"
```

### `POST /api/rate`

Submit a new user rating (persisted to SQLite).

```bash
curl -X POST "http://localhost:8000/api/rate" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "movie_id": 296, "rating": 4.5}'
```

### `GET /api/health`

Health check — confirms all model artifacts are loaded.

---

## How It Works

### 1. Data Pipeline

Raw MovieLens CSVs (`ratings.csv`, `movies.csv`, `links.csv`, `tags.csv`) are enriched with plot overviews fetched from the TMDB API. The cleaned dataset is saved to `data/processed/movies_with_plots.csv`. A migration script loads everything into SQLite for runtime persistence.

### 2. Content-Based Model

Movie overviews are encoded into 384-dimensional vectors using the `all-MiniLM-L6-v2` sentence transformer. These embeddings are indexed in a FAISS `IndexFlatL2` for sub-millisecond approximate nearest-neighbor search. A TF-IDF baseline on genres + overviews is also available for comparison.

### 3. Collaborative Filtering Model

An Alternating Least Squares (ALS) model from the `implicit` library is trained on the user-item interaction matrix (100k+ ratings). It learns latent factors that capture user preferences and item characteristics from rating patterns alone.

### 4. Hybrid Blending

For a given user, both models produce candidate scores:

```
hybrid_score = α × collab_score + (1 − α) × content_score
```

- **α = 0.6** by default (collaborative-leaning)
- **Cold-start users** (no ratings): falls back to popularity-weighted content recommendations
- **Warm users**: top-5 rated movies form a user embedding profile for the content arm; ALS provides the collaborative arm

### 5. Serving

Models are loaded once at startup (`models_loader.py`) and injected into endpoints via FastAPI dependency injection. All five artifacts — ALS model, embeddings array, FAISS index, movie DataFrame, item-code mapping — must be healthy for the service to accept traffic.

---

## Project Structure

```
movie-recommender/
├── api/                        # FastAPI application
│   ├── main.py                 #   App factory, lifespan, router registration
│   ├── models_loader.py        #   Model loading & dependency injection
│   ├── schemas.py              #   Pydantic request/response models
│   ├── dependencies.py         #   Shared dependencies
│   └── routers/
│       └── recommend.py        #   /recommend, /similar, /rate, /health
├── src/                        # Core ML & data logic
│   ├── data.py                 #   Data loading utilities
│   ├── db.py                   #   SQLAlchemy models (Movie, Rating) + engine
│   ├── utils.py                #   Helper functions
│   └── models/
│       ├── content.py          #   TF-IDF content similarity
│       ├── collab.py           #   Implicit ALS collaborative filtering
│       └── hybrid.py           #   Hybrid blending + cold-start logic
├── notebooks/                  # Jupyter exploration & training
│   ├── 01_eda.ipynb            #   Exploratory data analysis
│   ├── 02_content_tfidf.ipynb  #   TF-IDF baseline experiments
│   ├── 03_content_embeddings_faiss.ipynb  # Embedding + FAISS pipeline
│   └── 04_collab_implicit.ipynb           # ALS training & evaluation
├── scripts/
│   └── migrate_data.py         # CSV → SQLite migration
├── models/                     # Serialized model artifacts (git-ignored)
│   ├── collab_als.joblib       #   Trained ALS model
│   ├── content_embeddings.npy  #   384-dim sentence embeddings
│   ├── content_faiss.index     #   FAISS L2 index
│   ├── content_similarity_matrix.joblib   # TF-IDF cosine sim matrix
│   ├── tfidf_vectorizer.joblib #   Fitted TF-IDF vectorizer
│   └── movie_indices.csv       #   movieId ↔ matrix index mapping
├── data/
│   ├── raw/                    # Original MovieLens CSVs
│   └── processed/              # Enriched CSVs (with TMDB plots)
├── tests/
│   └── test_api.py             # Endpoint tests (health, recommend, similar, rate)
├── .github/workflows/
│   └── ci.yml                  # Lint → Test → Docker build pipeline
├── Dockerfile                  # Multi-stage (builder + slim runtime)
├── docker-compose.yml          # Local orchestration
├── pyproject.toml              # Poetry dependency management
└── poetry.lock
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/) for dependency management
- Docker (optional, for containerized runs)

### Local Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender

# Install dependencies
poetry install

# Download MovieLens data into data/raw/
# (ratings.csv, movies.csv, links.csv, tags.csv)

# Run the notebooks in order to train models:
#   01_eda.ipynb → 02_content_tfidf.ipynb → 03_content_embeddings_faiss.ipynb → 04_collab_implicit.ipynb
# This produces all artifacts in the models/ directory.

# Migrate data to SQLite
python scripts/migrate_data.py

# Start the API
uvicorn api.main:app --reload
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

### Docker

```bash
# Build and run
docker compose up --build

# Or standalone
docker build -t movie-recommender .
docker run -p 8000:8000 \
  -v ./models:/app/models:ro \
  -v ./data:/app/data:ro \
  movie-recommender
```

### Run Tests

```bash
poetry run pytest tests/
```

### Lint

```bash
poetry run ruff check .
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Dataset statistics, rating distributions, genre analysis, sparsity |
| `02_content_tfidf.ipynb` | TF-IDF vectorization on genres + plot overviews, cosine similarity baseline |
| `03_content_embeddings_faiss.ipynb` | Sentence-transformer encoding, FAISS index construction, qualitative evaluation |
| `04_collab_implicit.ipynb` | Implicit ALS training on user-item matrix, RMSE evaluation, top-N generation |

---

## License

MIT
