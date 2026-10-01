import json
from pathlib import Path

import cv2

from app.evaluation.application.evaluate import evaluate_texts, render_failure_cases_md
from app.evaluation.domain.metrics import FailureCase
from app.ingestion.domain.ports import VideoSource
from app.shared.config import Settings


def save_failure_frames(
    video: VideoSource, failures: list[FailureCase], directory: Path
) -> dict[int, str]:
    """One frame from the middle of each failure, for the write-up. Returns case number -> relative path."""
    directory.mkdir(parents=True, exist_ok=True)
    saved = {}
    for i, failure in enumerate(failures, start=1):
        middle = (failure.start_sec + failure.end_sec) / 2
        frames = video.frames_around(middle, 1, 0.0)
        if frames:
            cv2.imwrite(str(directory / f"case_{i}.jpg"), frames[0].image)
            saved[i] = f"{directory.name}/case_{i}.jpg"
    return saved


def evaluate_job_dir(
    job_dir: Path, ground_truth_csv: str, video: VideoSource | None, settings: Settings
) -> dict[str, object]:
    """Evaluate a finished job folder; writes evaluation.json, failure_cases.md and failure_frames/."""
    raw = (job_dir / "raw_result.json").read_text(encoding="utf-8")
    traces_file = job_dir / "agent_traces.json"
    traces = traces_file.read_text(encoding="utf-8") if traces_file.exists() else ""
    report = evaluate_texts(ground_truth_csv, raw, traces, settings)

    frames = (
        save_failure_frames(video, report.failures, job_dir / "failure_frames") if video else {}
    )
    result = report.to_dict()
    (job_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (job_dir / "failure_cases.md").write_text(
        render_failure_cases_md(report, frames), encoding="utf-8"
    )
    return result
