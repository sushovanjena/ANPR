"""Download the public datasets used to build and evaluate the ANPR models.

None of these datasets are stored in the repository: they are large and carry
their own licences. This script fetches them from their original sources into
``--root`` (default ``data/``). Check each licence before redistributing.

Sources:
    kaggle      kedarsai/indian-license-plates-with-labels (needs Kaggle API key)
    datacluster Dataclusterlabspvtltd/indian-number-plates-dataset (Hugging Face)
    roadtext    RoadText-1K, IIIT Hyderabad CVIT (direct zip)
    coco        COCO val2017 images + annotations (background images only)

Usage:
    python scripts/download_data.py --only roadtext datacluster
    python scripts/download_data.py            # everything
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROADTEXT_URL = "https://cvit.iiit.ac.in/images/datasets/RoadText-1K/RoadText1k.zip"
COCO_URLS = (
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
)
KAGGLE_SLUG = "kedarsai/indian-license-plates-with-labels"
HF_REPO = "Dataclusterlabspvtltd/indian-number-plates-dataset"


def _download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` unless it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"skip (exists): {dest}")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, part)
    part.rename(dest)
    return dest


def _unzip(zip_path: Path, out_dir: Path) -> None:
    """Extract ``zip_path`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def get_kaggle(root: Path) -> None:
    """Download the Kaggle plate dataset (requires ``~/.kaggle/kaggle.json``)."""
    out = root / "kaggle_plates"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG]
        + ["-p", str(out), "--unzip"],
        check=True,
    )


def get_datacluster(root: Path) -> None:
    """Download the DataCluster plate sample from Hugging Face."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        HF_REPO, repo_type="dataset", local_dir=str(root / "datacluster")
    )


def get_roadtext(root: Path) -> None:
    """Download and unpack RoadText-1K (read its License.txt before use)."""
    zip_path = _download(ROADTEXT_URL, root / "roadtext" / "RoadText1k.zip")
    _unzip(zip_path, root / "roadtext")


def get_coco(root: Path) -> None:
    """Download COCO val2017 images and annotations (used as backgrounds)."""
    for url in COCO_URLS:
        zip_path = _download(url, root / "coco" / url.rsplit("/", 1)[-1])
        _unzip(zip_path, root / "coco")


GETTERS = {
    "kaggle": get_kaggle,
    "datacluster": get_datacluster,
    "roadtext": get_roadtext,
    "coco": get_coco,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data", help="output directory")
    ap.add_argument("--only", nargs="+", choices=sorted(GETTERS), help="subset")
    args = ap.parse_args(argv)
    root = Path(args.root)
    for name in args.only or list(GETTERS):
        print(f"== {name}")
        GETTERS[name](root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
