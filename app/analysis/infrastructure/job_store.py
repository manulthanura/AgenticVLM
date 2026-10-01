from pathlib import Path
from uuid import uuid4

from app.analysis.domain.job import AnalysisJob, JobStore


class FileJobStore(JobStore):
    """Results are files under outputs/<job_id>/; the job index itself lives in memory (no database)."""

    def __init__(self, outputs_dir: str | Path) -> None:
        self._outputs_dir = Path(outputs_dir)
        self._jobs: dict[str, AnalysisJob] = {}

    def new_job(self) -> AnalysisJob:
        job_id = uuid4().hex[:8]
        job = AnalysisJob(job_id, self._outputs_dir / job_id)
        job.output_dir.mkdir(parents=True, exist_ok=True)
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> AnalysisJob | None:
        return self._jobs.get(job_id)
