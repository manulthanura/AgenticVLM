from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class AnalysisJob:
    job_id: str
    output_dir: Path  # results and uploaded inputs live here
    video_path: Path | None = None
    roi_path: Path | None = None
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0..1
    error: str | None = None


class JobStore(ABC):
    @abstractmethod
    def new_job(self) -> AnalysisJob: ...

    @abstractmethod
    def get(self, job_id: str) -> AnalysisJob | None: ...
