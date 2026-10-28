# Vision Patrol AI
### Configurable vision-based obstruction monitoring for industrial inspection

Vision Patrol AI is a small, modular computer-vision pipeline for reviewing images or video from an inspection route. It runs a pretrained YOLO object detector, filters detections by configurable confidence thresholds, evaluates configurable zone-based rules, and exports annotated media plus structured event logs.

> **Scope and limitations:** This is a portfolio prototype, not a certified safety system. The default model is pretrained on COCO and does **not** reliably verify PPE compliance, electrical-panel clearance, energized equipment, or every industrial hazard. Those tasks require suitable domain data, validation, and safety engineering. No custom model training or physical-robot deployment is claimed.

## Features
- Image and video inference using Ultralytics YOLO
- Configurable confidence threshold and monitored object classes
- Optional normalized rectangular inspection zone (`x1,y1,x2,y2`, values from 0 to 1)
- Rule-based alerts when selected objects are detected inside the zone
- Annotated image/video output
- JSONL event log with timestamp, class, confidence, bounding box, and zone status
- Clear CLI, configuration file, and unit tests for rule logic

## Architecture

```text
Image / video
     |
     v
YOLO detector (pretrained weights)
     |
     v
Detection filtering ----> annotated output
     |
     v
Zone/rule evaluator
     |
     +----> JSONL event log
     +----> console summary
```

## Quick start

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/YOUR-USERNAME/vision-patrol-ai.git
cd vision-patrol-ai

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Run on an image:

```bash
python -m vision_patrol.cli --source path/to/image.jpg --output-dir runs/demo
```

Run on a video:

```bash
python -m vision_patrol.cli --source path/to/patrol.mp4 --output-dir runs/demo
```

Use a configuration file:

```bash
python -m vision_patrol.cli --source path/to/patrol.mp4 --config configs/default.json --output-dir runs/demo
```

The first run downloads the pretrained `yolov8n.pt` weights if they are not already cached. Use only media you have permission to process.

## Configuration

`configs/default.json` contains:
- `model`: Ultralytics model weights
- `confidence`: minimum detection confidence
- `classes`: COCO class names to keep (empty list means keep all classes)
- `zone`: optional normalized rectangle; `null` disables zone checks
- `alert_classes`: detected classes that should produce alert events when inside the zone

Example zone coordinates are normalized to the frame size. A zone of `[0.25, 0.20, 0.85, 0.95]` covers the central/right lower area of a frame. This is a simple rectangular region, not a calibrated robot map.

## Outputs

For an input named `patrol.mp4`, the output directory contains:
- `annotated_patrol.mp4` — annotated video
- `events.jsonl` — one JSON event per line
- `summary.json` — aggregate counts and alert totals

For image input, an annotated image is also written.

## Tests

```bash
pip install pytest
pytest -q
```

Tests cover confidence filtering, zone membership, and alert rule behavior without downloading model weights.

## Engineering notes / next steps
1. Replace the rectangular image-space zone with a calibrated region tied to robot pose or a map.
2. Collect and label site-specific data for PPE, panel-clearance, and asset-tag tasks.
3. Evaluate precision/recall and false-alert rates on a held-out dataset before operational use.
4. Export a validated model to ONNX/TensorRT and benchmark latency on target edge hardware.
5. Integrate detections with a ROS 2 node and publish structured inspection events.

## Responsible use
Do not use this prototype as the sole basis for worker-safety decisions, electrical safety decisions, or autonomous robot control. Detection confidence is not a probability of safety, and a missing detection does not establish that an area is safe.
