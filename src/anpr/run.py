"""Command-line entry point: ``python -m anpr.run --config configs/default.yaml``."""

from __future__ import annotations

import argparse
import json
import logging
import signal

from anpr.config import load_config
from anpr.factory import build_runner


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and print the final metrics as JSON."""
    ap = argparse.ArgumentParser(description="Indian ANPR streaming pipeline")
    ap.add_argument("--config", help="YAML config file")
    ap.add_argument("--source", help="RTSP URL, video file or webcam index")
    ap.add_argument(
        "--live", action="store_true", default=None, help="treat as a live camera"
    )
    ap.add_argument("--no-plates", action="store_true", help="vehicles + tracking only")
    ap.add_argument("--max-frames", type=int, help="stop after N processed frames")
    ap.add_argument("--camera-id")
    ap.add_argument("--jsonl", help="event output file")
    ap.add_argument("--sqlite", help="event database file")
    args = ap.parse_args(argv)

    overrides = {
        "source": args.source,
        "live": args.live,
        "camera_id": args.camera_id,
        "sinks.jsonl": args.jsonl,
        "sinks.sqlite": args.sqlite,
        "plate.enabled": False if args.no_plates else None,
    }
    cfg = load_config(args.config, overrides)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    runner = build_runner(cfg)
    signal.signal(signal.SIGINT, lambda *_: runner.stop())
    signal.signal(signal.SIGTERM, lambda *_: runner.stop())
    snapshot = runner.run(max_frames=args.max_frames)
    print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
