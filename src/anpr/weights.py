"""Locate and download model weights listed in ``weights/manifest.json``.

Weights are not stored in git. After training they are uploaded as assets of a
GitHub Release (tag ``weights-v1`` by default) and their URL and SHA-256 are
written into the manifest. Until then the manifest entries have ``url: null``
and :func:`resolve` raises :class:`WeightsUnavailable` with instructions, so the
pipeline fails with a clear message instead of a stack trace.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
from pathlib import Path

log = logging.getLogger("anpr.weights")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "weights" / "manifest.json"
DEFAULT_DIR = REPO_ROOT / "weights"


class WeightsUnavailable(FileNotFoundError):
    """Raised when a weight file is absent and cannot be downloaded."""


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict:
    """Read the weights manifest.

    Args:
        path: Manifest JSON file.

    Returns:
        The parsed manifest (``{"repo", "release_tag", "files": {...}}``).
    """
    return json.loads(Path(path).read_text())


def release_url(manifest: dict, name: str) -> str | None:
    """Return the download URL for ``name``.

    An explicit ``url`` in the manifest wins; otherwise a URL is built from the
    release tag, but only once a SHA-256 is recorded, because that is the sign
    that the asset has actually been uploaded.
    """
    entry = manifest["files"].get(name)
    if entry is None:
        return None
    if entry.get("url"):
        return entry["url"]
    if entry.get("sha256"):
        repo, tag = manifest["repo"], manifest["release_tag"]
        return f"https://github.com/{repo}/releases/download/{tag}/{name}"
    return None


def sha256_of(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, dest_dir: Path | str = DEFAULT_DIR, manifest: dict | None = None):
    """Download one manifest file into ``dest_dir`` and verify its checksum.

    Args:
        name: File name as listed in the manifest.
        dest_dir: Directory to save into.
        manifest: Parsed manifest; the repository default is loaded if omitted.

    Returns:
        Path of the downloaded file.

    Raises:
        WeightsUnavailable: If the file has no download URL yet.
        ValueError: If the downloaded file does not match the recorded SHA-256.
    """
    manifest = manifest or load_manifest()
    url = release_url(manifest, name)
    if url is None:
        raise WeightsUnavailable(
            f"'{name}' has not been published yet (no URL in weights/manifest.json). "
            "Place the file in the weights/ directory by hand, or run with "
            "--no-plates for vehicle detection and tracking only."
        )
    dest = Path(dest_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s from %s", name, url)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out)
    expected = manifest["files"][name].get("sha256")
    if expected and sha256_of(tmp) != expected:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {name}; download discarded")
    tmp.replace(dest)
    return dest


def resolve(path: Path | str, auto_fetch: bool = True) -> Path:
    """Return an existing weight file, downloading it first if it is listed.

    Args:
        path: Configured weight path. If it does not exist and its file name is
            in the manifest, it is downloaded to that location.
        auto_fetch: Set False to forbid network access.

    Returns:
        Path to the weight file.

    Raises:
        WeightsUnavailable: If the file is missing and cannot be fetched.
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    if p.exists():
        return p
    manifest = load_manifest()
    if auto_fetch and p.name in manifest["files"]:
        return fetch(p.name, p.parent, manifest)
    raise WeightsUnavailable(
        f"weights not found: {p}. Run `python -m anpr.fetch_weights` or copy the "
        "file there."
    )


def ensure_openocr(target: Path | str) -> Path:
    """Clone OpenOCR (the RepSVTR code base) if it is not present.

    Args:
        target: Directory to clone into.

    Returns:
        The OpenOCR directory.
    """
    import subprocess

    target = Path(target)
    if not target.is_absolute():
        target = REPO_ROOT / target
    if not (target / "configs").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/Topdu/OpenOCR.git",
                str(target),
            ],
            check=True,
        )
    return target
