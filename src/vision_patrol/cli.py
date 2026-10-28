"""Command-line interface."""
from __future__ import annotations

import argparse
import json
import sys

from .pipeline import load_config, process_media


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run configurable YOLO-based visual obstruction monitoring."
    )
    parser.add_argument("--source", required=True, help="Path to an image or video.")
    parser.add_argument("--output-dir", default="runs/inspection", help="Output directory.")
    parser.add_argument("--config", default=None, help="Optional JSON configuration file.")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Override confidence threshold (0, 1].")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
        if args.confidence is not None:
            if not 0 < args.confidence <= 1:
                raise ValueError("--confidence must be in the interval (0, 1].")
            config["confidence"] = args.confidence
        summary = process_media(args.source, args.output_dir, config)
        print(json.dumps(summary, indent=2))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
