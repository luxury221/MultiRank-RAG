from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import API_TITLE, CORS_ORIGINS
from backend.routers import analyze, health, jobs


def create_app() -> FastAPI:
    app = FastAPI(title=API_TITLE)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(jobs.router)
    return app


app = create_app()
