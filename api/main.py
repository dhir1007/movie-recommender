from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends

from api.routers.recommend import router as recommend_router
from api.models_loader import load_all_models

@asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("=== Lifespan startup started ===")
#     try:
#         load_all_models()
#         print("=== Lifespan startup finished successfully ===")
#     except Exception as e:
#         print("=== Lifespan startup FAILED ===")
#         print(str(e))
#         raise  # re-raise so server fails visibly
#     yield
#     print("=== Lifespan shutdown ===")

def get_models():
    return load_all_models()

app = FastAPI(
    title="Hybrid Movie Recommender API",
    description="Personalized movie recommendations using hybrid content-based (embeddings) + collaborative (implicit ALS) filtering",
    version="0.1.0",
)

app.include_router(recommend_router)


@app.get("/")
async def root():
    return {"message": "Welcome to the Hybrid Movie Recommender API. Visit /docs for Swagger UI."}