from enum import StrEnum

from shapely.geometry import Point, Polygon, box


class BedZone(StrEnum):
    INSIDE = "inside"
    EDGE = "edge"
    OUTSIDE = "outside"


class BedRegion:
    def __init__(self, polygon: list[tuple[float, float]], edge_margin_px: float) -> None:
        self._polygon = Polygon(polygon)
        self._edge_margin_px = edge_margin_px

    def zone(self, x: float, y: float) -> BedZone:
        point = Point(x, y)
        # Hips of someone sitting on the bed edge often fall just outside the drawn polygon,
        # so anything within the margin of the border counts as EDGE on either side.
        if point.distance(self._polygon.exterior) <= self._edge_margin_px:
            return BedZone.EDGE
        return BedZone.INSIDE if self._polygon.contains(point) else BedZone.OUTSIDE

    def overlap_ratio(self, bbox: tuple[float, float, float, float]) -> float:
        """Share of the person's bounding box that lies on the bed."""
        person = box(*bbox)
        return person.intersection(self._polygon).area / person.area if person.area else 0.0
