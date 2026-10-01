from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, FastAPI

from app.analysis.api.router import create_router as create_analysis_router
from app.analysis.api.ui import create_router as create_ui_router
from app.analysis.application.run_analysis import RunAnalysis
from app.analysis.infrastructure.container import build_run_analysis, make_video_opener
from app.analysis.infrastructure.job_store import FileJobStore
from app.evaluation.api.router import create_router as create_evaluation_router
from app.ingestion.domain.ports import VideoSource
from app.shared.config import Settings, get_settings

API_PREFIX = "/api/v1"


def create_app(
    settings: Settings | None = None,
    run_analysis: RunAnalysis | None = None,
    open_video: Callable[[Path], VideoSource] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="AgenticVLM", description="Elderly bed-monitoring video analysis")

    api = APIRouter(prefix=API_PREFIX)

    @api.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    store = FileJobStore(settings.outputs_dir)
    run_analysis = run_analysis or build_run_analysis(settings)
    open_video = open_video or make_video_opener(settings)
    api.include_router(create_analysis_router(store, run_analysis, settings, open_video))
    api.include_router(create_evaluation_router(store, settings, open_video))
    app.include_router(api)
    app.include_router(create_ui_router())
    return app


app = create_app()
