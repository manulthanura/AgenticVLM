import logging
from collections.abc import Callable
from pathlib import Path

from app.alerting.domain.rules import evaluate_alerts
from app.analysis.domain.job import AnalysisJob, JobStatus
from app.events.domain.detector import BedEventDetector
from app.extract.application.feature_extractor import FeatureExtractor
from app.extract.domain.bed import BedRegion
from app.extract.domain.ports import PoseEstimator
from app.ingestion.domain.ports import VideoSource
from app.reasoning.application.reasoner import Reasoner
from app.reporting.application.render_report import render_report
from app.shared.config import Settings
from app.state.application.build_timeline import build_timeline

logger = logging.getLogger(__name__)


class RunAnalysis:
    """The whole pipeline for one job. Concrete adapters are injected by the composition root."""

    def __init__(
        self,
        settings: Settings,
        open_video: Callable[[Path], VideoSource],
        load_bed: Callable[[Path], BedRegion],
        make_estimator: Callable[[], PoseEstimator],  # new per job: trackers keep state
        write_report: Callable[[Path, dict[str, str]], None],
        make_reasoner: Callable[[], Reasoner]
        | None = None,  # new per job: the call cap is per video
    ) -> None:
        self._settings = settings
        self._open_video, self._load_bed = open_video, load_bed
        self._make_estimator, self._write_report = make_estimator, write_report
        self._make_reasoner = make_reasoner

    def run(self, job: AnalysisJob) -> None:
        """Never raises: a background task has nobody to catch, so failures go on the job."""
        job.status = JobStatus.RUNNING
        try:
            source = self._open_video(job.video_path)
            extractor = FeatureExtractor(
                self._make_estimator(), self._load_bed(job.roi_path), self._settings
            )
            features = []
            for frame in source.frames():
                features.append(extractor.extract(frame))
                job.progress = frame.index / max(1, source.metadata.frame_count)
            timeline = build_timeline(features, source.metadata.duration_sec, self._settings)
            events = BedEventDetector(self._settings).detect(timeline)
            traces = []
            if self._make_reasoner:  # optional second pass for ambiguous cases
                result = self._make_reasoner().run(timeline, events, features, source)
                timeline, events, traces = result.timeline, result.events, result.traces
            findings = [t.finding for t in traces]
            decisions = evaluate_alerts(timeline, events, self._settings, findings)
            report = render_report(timeline, events, decisions, traces)
            self._write_report(job.output_dir, report)
            job.progress, job.status = 1.0, JobStatus.DONE
        except Exception as exc:
            logger.exception("Analysis %s failed", job.job_id)
            job.status, job.error = JobStatus.FAILED, str(exc)
