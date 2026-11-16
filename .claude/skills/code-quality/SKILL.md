---
name: code-quality
description: Check and keep this repo's code quality (PEP 8, Google-style docstrings, small modules, tests). Use before committing or when adding or changing code in src/, tests/ or scripts/.
---

# Code quality for this repo

Run these from the repo root, in order, and fix what they report.

1. `ruff check .` : PEP 8 (E, W), pyflakes (F), import order (I) and Google-style
   docstrings (D). Line length is 88. Config is in `pyproject.toml`.
2. `ruff format --check src tests scripts` : formatting. Run `ruff format <path>` to fix.
3. `pytest -q` : all tests must pass on CPU, with no network and no weights.

## Rules

- Every module, public class and public function has a docstring: one summary line,
  then `Args:`, `Returns:` and `Raises:` when they add information.
- One responsibility per module. Pipeline stages live in `src/anpr/pipeline/`, the
  tracker in `track/`, OCR in `ocr/`, live-stream code in `stream/`, configuration in
  `config.py`, weights handling in `weights.py`.
- Stages take their model as a plain function or object so tests can pass a fake.
  Never import a heavy model library at module top level if the unit tests do not need it.
- Config keys are typed dataclasses; unknown keys must raise, not be ignored.
- A new behaviour needs a test in `tests/test_<module>.py`; a bug fix needs a test
  that fails without the fix.
- No weights, datasets or secrets in git. Weights come from `weights/manifest.json`.
- `notebooks/development_log.ipynb` is a record of how the work was done. Do not lint
  or rewrite it.
