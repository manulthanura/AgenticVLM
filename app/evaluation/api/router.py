from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.analysis.domain.job import JobStatus, JobStore
from app.evaluation.infrastructure.job_evaluator import evaluate_job_dir
from app.ingestion.domain.ports import VideoSource
from app.shared.config import Settings


def create_router(
    store: JobStore, settings: Settings, open_video: Callable[[Path], VideoSource]
) -> APIRouter:
    router = APIRouter(prefix="/evaluation", tags=["evaluation"])

    @router.post("/{job_id}", summary="Evaluate a finished job against ground truth")
    def evaluate(
        job_id: str,
        ground_truth: UploadFile | None = File(
            None, description="CSV: start_sec,end_sec,state. Default: data/ground_truth/<video>.csv"
        ),
    ) -> dict[str, object]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        if job.status != JobStatus.DONE:
            raise HTTPException(409, f"Job is {job.status.value}")

        if ground_truth is not None:
            raw = ground_truth.file.read()
        else:
            path = Path(settings.data_dir) / "ground_truth" / f"{job.video_path.stem}.csv"
            if not path.is_file():
                raise HTTPException(400, f"Upload a CSV or create {path.as_posix()}")
            raw = path.read_bytes()

        try:  # frames for the failure write-up are a nice-to-have
            video = open_video(job.video_path)
        except OSError:
            video = None
        try:
            return evaluate_job_dir(job.output_dir, raw.decode("utf-8-sig"), video, settings)
        except ValueError as exc:  # bad CSV
            raise HTTPException(422, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(409, "Job has no raw_result.json: run the analysis again") from exc

    return router
