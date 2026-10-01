from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

INDEX = Path(__file__).parent / "ui" / "index.html"


def create_router() -> APIRouter:
    """Serves the single-page UI at `/`. It only calls the public /api/v1 endpoints."""
    router = APIRouter(include_in_schema=False)

    @router.get("/")
    def index() -> FileResponse:
        return FileResponse(INDEX)

    return router
