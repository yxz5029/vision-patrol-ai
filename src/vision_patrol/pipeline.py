"""YOLO inference and media processing."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from .rules import evaluate_detection


def load_config(path: str | Path | None) -> dict[str, Any]:
    defaults = {
        "model": "yolov8n.pt",
        "confidence": 0.35,
        "classes": [],
        "zone": None,
        "alert_classes": [],
    }
    if path is None:
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        user_config = json.load(f)
    defaults.update(user_config)
    if not 0 < float(defaults["confidence"]) <= 1:
        raise ValueError("confidence must be in the interval (0, 1].")
    return defaults


def process_media(source: str | Path, output_dir: str | Path,
                  config: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Input media does not exist: {source_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(config["model"])
    allowed_classes = {str(x).lower() for x in config.get("classes", [])}
    alert_classes = {str(x).lower() for x in config.get("alert_classes", [])}
    zone = config.get("zone")

    video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v"}
    is_video = source_path.suffix.lower() in video_suffixes
    cap = cv2.VideoCapture(str(source_path)) if is_video else None
    if is_video:
        if cap is None or not cap.isOpened():
            raise ValueError(f"Could not open video: {source_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 20.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_path = out_dir / f"annotated_{source_path.stem}.mp4"
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create output video. Check codec and output path.")
    else:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise ValueError("Input must be a readable image or video.")
        height, width = frame.shape[:2]
        total_frames = 1
        output_path = out_dir / f"annotated_{source_path.stem}{source_path.suffix}"
        writer = None

    events: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    frame_index = 0

    try:
        while True:
            if is_video:
                ok, frame = cap.read()
                if not ok:
                    break
            elif frame_index == 0:
                frame = cv2.imread(str(source_path))
            else:
                break

            result = model.predict(frame, conf=float(config["confidence"]), verbose=False)[0]
            detections = []
            if result.boxes is not None:
                for item in result.boxes:
                    cls_id = int(item.cls.item())
                    class_name = str(model.names[cls_id]).lower()
                    confidence = float(item.conf.item())
                    if allowed_classes and class_name not in allowed_classes:
                        continue
                    bbox = [float(v) for v in item.xyxy[0].tolist()]
                    detection = {
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "bbox": [round(v, 2) for v in bbox],
                    }
                    detection = evaluate_detection(
                        detection, zone, alert_classes, width, height
                    )
                    detections.append(detection)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    if detection["alert"]:
                        events.append({
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "frame": frame_index,
                            **detection,
                        })

            annotated = result.plot()
            if zone is not None:
                zx1, zy1, zx2, zy2 = zone
                p1 = (int(zx1 * width), int(zy1 * height))
                p2 = (int(zx2 * width), int(zy2 * height))
                cv2.rectangle(annotated, p1, p2, (0, 165, 255), 2)
                cv2.putText(annotated, "MONITORED ZONE", (p1[0], max(20, p1[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            if is_video:
                writer.write(annotated)
            else:
                cv2.imwrite(str(output_path), annotated)

            frame_index += 1
            if not is_video:
                break
    finally:
        if is_video:
            cap.release()
            writer.release()

    event_path = out_dir / "events.jsonl"
    with event_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    summary = {
        "input": str(source_path),
        "frames_processed": frame_index,
        "detections_by_class": class_counts,
        "alert_event_count": len(events),
        "annotated_output": str(output_path),
        "event_log": str(event_path),
        "model": config["model"],
        "confidence_threshold": config["confidence"],
        "note": "Prototype only; not validated for safety-critical decisions."
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
