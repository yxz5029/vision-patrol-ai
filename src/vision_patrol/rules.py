"""Pure rule logic, separated from model inference for easy testing."""
from __future__ import annotations

from typing import Any


def center_in_zone(box: list[float] | tuple[float, float, float, float],
                   zone: list[float] | tuple[float, float, float, float] | None,
                   frame_width: int,
                   frame_height: int) -> bool:
    """Return whether a pixel-space bounding-box center lies in a normalized zone."""
    if zone is None:
        return False
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    if len(box) != 4 or len(zone) != 4:
        raise ValueError("box and zone must each contain four coordinates.")

    x1, y1, x2, y2 = map(float, box)
    zx1, zy1, zx2, zy2 = map(float, zone)
    if not (0 <= zx1 <= zx2 <= 1 and 0 <= zy1 <= zy2 <= 1):
        raise ValueError("Zone coordinates must be normalized and ordered in [0, 1].")

    cx = (x1 + x2) / 2.0 / frame_width
    cy = (y1 + y2) / 2.0 / frame_height
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def evaluate_detection(detection: dict[str, Any],
                       zone: list[float] | None,
                       alert_classes: set[str],
                       frame_width: int,
                       frame_height: int) -> dict[str, Any]:
    """Enrich a detection with zone and alert flags."""
    in_zone = center_in_zone(detection["bbox"], zone, frame_width, frame_height)
    is_alert = in_zone and detection["class_name"] in alert_classes
    return {**detection, "in_zone": in_zone, "alert": is_alert}
