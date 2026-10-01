"""Composition root: the only place that picks concrete adapters."""

from collections.abc import Callable
from pathlib import Path

from app.analysis.application.run_analysis import RunAnalysis
from app.extract.domain.ports import PoseEstimator
from app.extract.infrastructure.roi_loader import load_bed_region
from app.ingestion.domain.ports import VideoSource
from app.ingestion.infrastructure.opencv_source import OpenCVVideoSource
from app.reasoning.application.reasoner import Reasoner
from app.reporting.infrastructure.file_writer import write_files
from app.shared.config import Settings


def azure_configured(settings: Settings) -> bool:
    return all(
        (
            settings.azure_openai_endpoint,
            settings.azure_openai_api_key,
            settings.azure_openai_api_version,
            settings.azure_openai_deployment_fast,
        )
    )


def make_video_opener(settings: Settings) -> Callable[[Path], VideoSource]:
    return lambda path: OpenCVVideoSource(path, settings.sample_fps)


def build_run_analysis(settings: Settings) -> RunAnalysis:
    def make_estimator() -> PoseEstimator:
        # Imported here so the API starts fast and tests never need ultralytics.
        from app.extract.infrastructure.ultralytics_estimator import UltralyticsPoseEstimator

        return UltralyticsPoseEstimator(settings.pose_model, settings.detection_conf_min)

    def make_reasoner() -> Reasoner:
        from openai import AzureOpenAI

        from app.reasoning.infrastructure.azure_vlm import AzureVlm
        from app.reasoning.infrastructure.disk_cache import DiskCache

        client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        cache = DiskCache(Path(settings.outputs_dir) / "cache")
        return Reasoner(AzureVlm(client, settings, cache), settings)

    return RunAnalysis(
        settings,
        open_video=make_video_opener(settings),
        load_bed=lambda path: load_bed_region(path, settings.edge_margin_px),
        make_estimator=make_estimator,
        write_report=write_files,
        # Without Azure settings the agent is skipped and the pipeline runs on rules alone.
        make_reasoner=make_reasoner if azure_configured(settings) else None,
    )
