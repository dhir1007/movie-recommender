# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert all(data["models_loaded"].values())  # all true

def test_recommend():
    response = client.get("/api/recommend?user_id=1&n=5")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 5
    assert "user_id" in data
    assert data["user_id"] == 1

def test_similar():
    response = client.get("/api/similar?movie_id=603&n=5")
    assert response.status_code in (200, 404)  # 404 if movie missing is ok
    if response.status_code == 200:
        data = response.json()
        assert "similar_movies" in data
        assert len(data["similar_movies"]) <= 5

def test_rate():
    payload = {
        "user_id": 999,
        "movie_id": 296,
        "rating": 4.5
    }
    response = client.post("/api/rate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "rating_id" in data
    assert data["message"] == "Rating saved successfully"