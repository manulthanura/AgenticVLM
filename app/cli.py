import argparse
import json
from pathlib import Path

import cv2

from app.analysis.domain.job import JobStatus
from app.analysis.infrastructure.container import build_run_analysis, make_video_opener
from app.analysis.infrastructure.job_store import FileJobStore
from app.dataprep.domain.clips import ground_truth_template
from app.dataprep.infrastructure.roi import (
    draw_roi_preview,
    parse_points,
    pick_polygon,
    read_frame,
    save_roi,
)
from app.dataprep.infrastructure.stitcher import stitch_clips
from app.evaluation.infrastructure.job_evaluator import evaluate_job_dir
from app.shared.config import Settings, get_settings


def analyze(args: argparse.Namespace, settings: Settings) -> None:
    job = FileJobStore(settings.outputs_dir).new_job()
    job.video_path, job.roi_path = Path(args.video), Path(args.roi)
    build_run_analysis(settings).run(job)
    if job.status == JobStatus.FAILED:
        raise SystemExit(f"Analysis failed: {job.error}")
    print(f"Done. Results in {job.output_dir}")
    print((job.output_dir / "timeline.txt").read_text(encoding="utf-8"))


def evaluate(args: argparse.Namespace, settings: Settings) -> None:
    video = make_video_opener(settings)(Path(args.video)) if args.video else None
    ground_truth = Path(args.gt).read_text(encoding="utf-8-sig")
    result = evaluate_job_dir(Path(args.job), ground_truth, video, settings)
    print(json.dumps({k: result[k] for k in ("frame_accuracy", "events", "llm")}, indent=2))
    print(f"Full report, failure_cases.md and frames are in {args.job}")


def stitch(args: argparse.Namespace, settings: Settings) -> None:
    out = Path(args.out)
    clips = stitch_clips([Path(c) for c in args.clips], out)
    for clip in clips:
        print(f"{clip.start_sec:7.2f} - {clip.end_sec:7.2f}  {clip.name}")
    template = Path(settings.data_dir) / "ground_truth" / f"{out.stem}.csv"
    if template.exists():
        print(f"Kept existing ground truth: {template}")
    else:
        template.parent.mkdir(parents=True, exist_ok=True)
        template.write_text(ground_truth_template(clips), encoding="utf-8")
        print(f"Wrote ground-truth template (fill in the states): {template}")


def roi(args: argparse.Namespace, settings: Settings) -> None:
    video = Path(args.video)
    frame = read_frame(video, args.time)
    polygon = parse_points(args.points) if args.points else pick_polygon(frame)
    if not polygon:
        raise SystemExit("Cancelled: no polygon saved")
    height, width = frame.shape[:2]
    roi_path = Path(settings.data_dir) / "rois" / f"{video.stem}.json"
    save_roi(roi_path, polygon, (width, height))
    preview = roi_path.with_name(f"{video.stem}_preview.jpg")
    cv2.imwrite(str(preview), draw_roi_preview(frame, polygon))
    print(f"Saved {roi_path}\nCheck the polygon in {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="Analyze a video end-to-end")
    p.add_argument("video")
    p.add_argument("--roi", required=True, help="Bed polygon JSON")
    p.set_defaults(run=analyze)

    p = sub.add_parser("evaluate", help="Evaluate a finished job against ground truth")
    p.add_argument("--job", required=True, help="outputs/<job_id> folder")
    p.add_argument("--gt", required=True, help="Ground-truth CSV")
    p.add_argument("--video", help="The analysed video, to save frames for failure cases")
    p.set_defaults(run=evaluate)

    p = sub.add_parser("stitch", help="Join same-scene clips and write a ground-truth template")
    p.add_argument("out", help="Output mp4, e.g. data/videos/seq1.mp4")
    p.add_argument("clips", nargs="+")
    p.set_defaults(run=stitch)

    p = sub.add_parser("roi", help="Create data/rois/<video>.json (click in a window, or --points)")
    p.add_argument("video")
    p.add_argument("--time", type=float, default=0.0, help="Frame to show, in seconds")
    p.add_argument("--points", help="Skip the window: 'x,y x,y x,y ...'")
    p.set_defaults(run=roi)

    args = parser.parse_args()
    args.run(args, get_settings())


if __name__ == "__main__":
    main()
