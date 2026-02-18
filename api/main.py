from fastapi import FastAPI

app = FastAPI(title="Movie Recommender API")

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running!"}

@app.get("/")
def root():
    return {"message": "Welcome to the Hybrid Movie Recommendation API"}