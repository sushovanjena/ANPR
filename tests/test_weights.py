import hashlib
import json

import pytest

from anpr import weights


def _manifest(tmp_path, **entry):
    return {"repo": "o/r", "release_tag": "t", "files": {"w.pt": entry}}


def test_unpublished_weight_gives_clear_error(tmp_path):
    m = _manifest(tmp_path, url=None, sha256=None)
    with pytest.raises(weights.WeightsUnavailable, match="not been published"):
        weights.fetch("w.pt", tmp_path, m)


def test_release_url_built_only_after_checksum_recorded():
    assert weights.release_url(_manifest(None, url=None, sha256=None), "w.pt") is None
    url = weights.release_url(_manifest(None, url=None, sha256="ab"), "w.pt")
    assert url == "https://github.com/o/r/releases/download/t/w.pt"


def test_fetch_verifies_checksum(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"model-bytes")
    good = hashlib.sha256(b"model-bytes").hexdigest()
    m = _manifest(tmp_path, url=src.as_uri(), sha256=good)
    out = weights.fetch("w.pt", tmp_path / "dl", m)
    assert out.read_bytes() == b"model-bytes"

    m = _manifest(tmp_path, url=src.as_uri(), sha256="0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        weights.fetch("w.pt", tmp_path / "dl2", m)
    assert not list((tmp_path / "dl2").glob("*"))


def test_resolve_returns_existing_file_without_network(tmp_path):
    f = tmp_path / "x.pt"
    f.write_bytes(b"1")
    assert weights.resolve(f, auto_fetch=False) == f


def test_resolve_missing_unlisted_file_raises(tmp_path):
    with pytest.raises(weights.WeightsUnavailable, match="weights not found"):
        weights.resolve(tmp_path / "nope.pt")


def test_repo_manifest_lists_every_default_weight():
    from pathlib import Path

    from anpr.config import Config

    cfg = Config()
    listed = set(weights.load_manifest()["files"])
    for p in (cfg.plate.weights, cfg.ocr.weights, cfg.ocr.dict_path):
        assert Path(p).name in listed
    json.dumps(weights.load_manifest())


def test_fetch_cli_hash(tmp_path, capsys):
    from anpr.fetch_weights import main

    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    assert main(["--hash", str(f)]) == 0
    assert hashlib.sha256(b"abc").hexdigest() in capsys.readouterr().out
