import shutil
from collections.abc import Callable
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)

from app.analysis.application.run_analysis import RunAnalysis
from app.analysis.domain.job import AnalysisJob, JobStatus, JobStore
from app.ingestion.domain.ports import VideoSource
from app.shared.config import Settings

# Result files of a finished job, each served as-is: URL name -> (file, media type, Swagger summary).
_JSON = "application/json"
RESULT_FILES = {
    "timeline": ("timeline.txt", "text/plain; charset=utf-8", "State timeline"),
    "summary": ("summary.json", _JSON, "Activity-duration summary"),
    "events": ("events.json", _JSON, "Bed events with decisions"),
    "alerts": ("alerts.json", _JSON, "All alert decisions with evidence"),
    "traces": (
        "agent_traces.json",
        _JSON,
        "Agent traces and VLM usage (empty when the agent is off)",
    ),
    "result": ("raw_result.json", _JSON, "Segments and events in seconds (used by the UI)"),
}


def create_router(
    store: JobStore,
    run_analysis: RunAnalysis,
    settings: Settings,
    open_video: Callable[[Path], VideoSource],
) -> APIRouter:
    router = APIRouter(prefix="/analysis", tags=["analysis"])
    videos_dir, rois_dir = Path(settings.data_dir) / "videos", Path(settings.data_dir) / "rois"

    def inside(base: Path, name: str) -> Path:
        path = (base / name).resolve()
        if not path.is_relative_to(base.resolve()):  # no ../ escapes
            raise HTTPException(400, f"Path must stay inside {base}")
        if not path.is_file():
            raise HTTPException(400, f"File not found: {path.name}")
        return path

    def save(upload: UploadFile, destination: Path) -> Path:
        with destination.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        return destination

    def job_or_404(job_id: str) -> AnalysisJob:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job

    def result(job_id: str, name: str) -> str:
        job = job_or_404(job_id)
        if job.status != JobStatus.DONE:
            raise HTTPException(409, f"Job is {job.status.value}")
        return (job.output_dir / name).read_text(encoding="utf-8")

    @router.post("", status_code=202, summary="Start an analysis")
    def start(
        background: BackgroundTasks,
        video: UploadFile | None = File(None, description="Video upload, or use video_path"),
        roi: UploadFile | None = File(None, description="Bed polygon JSON upload, or use roi_path"),
        video_path: str | None = Form(None, description="File name inside data/videos"),
        roi_path: str | None = Form(
            None, description="File name inside data/rois (default: <video>.json)"
        ),
    ) -> dict[str, str]:
        if (video is None) == (video_path is None):
            raise HTTPException(400, "Give either a video upload or video_path")
        if video is not None and roi is None and roi_path is None:
            raise HTTPException(400, "An uploaded video needs a bed polygon (roi or roi_path)")
        # Validate named files before creating the job so failed requests leave no folder behind.
        known_video = inside(videos_dir, video_path) if video_path else None
        known_roi = None
        if roi is None:
            known_roi = inside(rois_dir, roi_path or f"{known_video.stem}.json")

        job = store.new_job()
        job.video_path = known_video or save(video, job.output_dir / "input.mp4")
        job.roi_path = known_roi or save(roi, job.output_dir / "roi.json")
        background.add_task(run_analysis.run, job)
        return {"job_id": job.job_id}

    @router.get("/sources", summary="Videos and bed polygons available on the server")
    def sources() -> dict[str, list[str]]:
        def names(base: Path, pattern: str) -> list[str]:
            return sorted(p.relative_to(base).as_posix() for p in base.rglob(pattern))

        return {"videos": names(videos_dir, "*.mp4"), "rois": names(rois_dir, "*.json")}

    @router.get("/{job_id}", summary="Status and progress")
    def status(job_id: str) -> dict[str, object]:
        job = job_or_404(job_id)
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": round(job.progress, 3),
            "error": job.error,
        }

    @router.get("/{job_id}/frame", summary="JPEG of the analysed video at time t (seconds)")
    def frame(job_id: str, t: float = Query(0, ge=0)) -> Response:
        # A frame instead of a <video>: stitched files use a codec browsers cannot play.
        try:
            jpeg = open_video(job_or_404(job_id).video_path).frame_jpeg(t, max_side=960)
        except (OSError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(jpeg, media_type="image/jpeg")

    def add_result_route(name: str, file: str, media_type: str, summary: str) -> None:
        def handler(job_id: str) -> Response:
            return Response(result(job_id, file), media_type=media_type)

        router.add_api_route(f"/{{job_id}}/{name}", handler, methods=["GET"], summary=summary)

    for name, (file, media_type, summary) in RESULT_FILES.items():
        add_result_route(name, file, media_type, summary)

    return router
