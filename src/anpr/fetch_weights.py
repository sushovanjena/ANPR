"""Command line: download published weights or hash files for the manifest.

Examples:
    python -m anpr.fetch_weights                 # download everything published
    python -m anpr.fetch_weights --name ocr_repsvtr_v2.pt
    python -m anpr.fetch_weights --hash weights/ocr_repsvtr_v2.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from anpr import weights


def main(argv: list[str] | None = None) -> int:
    """Download weights, or print the checksum to record in the manifest."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", action="append", help="file to fetch (repeatable)")
    ap.add_argument("--dir", default=str(weights.DEFAULT_DIR))
    ap.add_argument("--openocr", action="store_true", help="also clone OpenOCR")
    ap.add_argument("--hash", metavar="FILE", help="print SHA-256 and size of FILE")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.hash:
        p = Path(args.hash)
        print(f'"sha256": "{weights.sha256_of(p)}"  # {p.stat().st_size} bytes')
        return 0
    manifest = weights.load_manifest()
    names = args.name or list(manifest["files"])
    missing = 0
    for name in names:
        try:
            print("ok", weights.fetch(name, args.dir, manifest))
        except weights.WeightsUnavailable as exc:
            print("skip", name, "-", exc)
            missing += 1
    if args.openocr:
        print("openocr", weights.ensure_openocr("third_party/OpenOCR"))
    return 1 if missing == len(names) else 0


if __name__ == "__main__":
    raise SystemExit(main())
