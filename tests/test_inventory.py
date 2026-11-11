from pathlib import Path

from anpr.data.inventory import label_path_for, parse_label_file


def test_parse_valid_and_malformed(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.2 0.1\n0 0.1 0.1 0.2 0.2 0.3 0.3\n")
    rows, bad = parse_label_file(f)
    assert rows == [(0, 0.5, 0.5, 0.2, 0.1)] and bad == 1


def test_missing_label_is_empty(tmp_path):
    assert parse_label_file(tmp_path / "none.txt") == ([], 0)


def test_label_path_mapping():
    p = Path("train/images/x.jpg")
    assert label_path_for(p) == Path("train/labels/x.txt")
