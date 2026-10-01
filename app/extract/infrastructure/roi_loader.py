import json
from pathlib import Path

from app.extract.domain.bed import BedRegion


def load_bed_region(path: str | Path, edge_margin_px: float) -> BedRegion:
    """Read `{"bed_polygon": [[x, y], ...], "frame_size": [w, h]}` (pixel coordinates of the video)."""
    data = json.loads(Path(path).read_text())
    return BedRegion([tuple(point) for point in data["bed_polygon"]], edge_margin_px)
