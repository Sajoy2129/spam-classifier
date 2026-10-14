# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning project: an SMS spam classifier (TF-IDF + Logistic Regression)
trained on the UCI SMS Spam Collection. `PLAN.md` is the phase-by-phase
checklist; tick its boxes as steps are finished. `README.md` holds the
published results.

## Commands

Use the project virtual environment (Python 3.13): `source .venv/bin/activate`,
or call `.venv/bin/python` directly.

```
python src/download_data.py          # fetch dataset into data/ (needed first)
python src/train.py                  # compare models, save models/spam_pipeline.joblib
python src/evaluate.py               # confusion matrices, P/R/F1, 5-fold CV, mistakes
python src/predict.py "some message" # one message; no argument = interactive loop
pytest                               # all tests
pytest tests/test_evaluate.py::test_score_predictions_known_values  # one test
```

There is no linter configured. Follow PEP 8.

## Architecture

- **Flat imports, no package.** Modules import each other as `from data import ...`,
  not `from src.data import ...`. Scripts work because `src/` is the script's
  own folder; tests work because `pytest.ini` sets `pythonpath = src`. Keep it
  this way.
- **Paths** come from `download_data.PROJECT_ROOT` (`DATA_FILE`) and
  `train.MODEL_FILE`, so scripts run from any folder. `data/` and `models/` are
  gitignored and recreated by the scripts.
- **Data flow:** `load_data()` → `remove_duplicates()` → `split_data()` (80/20,
  stratified, `random_state=42`). Duplicates are dropped *before* the split so
  the same text cannot land in both train and test sets.
- **`train.py` is the source of truth for models:** `MODELS` (name →
  unfitted classifier, including an always-ham `DummyClassifier` baseline)
  and `FINAL_MODEL` (the key that gets saved). `evaluate.py` imports both;
  `predict.py` imports `MODEL_FILE`. To switch the final model, change
  `FINAL_MODEL` and re-run `train.py`.
- **`model.build_pipeline(classifier)`** wraps TF-IDF + a `clone()` of the
  classifier in one sklearn `Pipeline`. The clone stops pipelines from
  sharing the module-level `MODELS` instances. Cross-validation refits the
  whole pipeline per fold, so the vectorizer never sees held-out data.
- The saved model is fitted on the **training split only**, so
  `evaluate.py`'s test-set numbers describe the saved model.
- Labels are the strings `"ham"` / `"spam"`. sklearn metrics need
  `pos_label="spam"`; spam is the class being scored.

## Gotchas

- `load_data` uses `quoting=csv.QUOTE_NONE` (messages contain stray `"`) and
  `keep_default_na=False` (so text like `NA` is not read as NaN). Tests
  cover both.
- `test_real_dataset_shape` is skipped when `data/` has not been downloaded.
  The other tests use tiny made-up data.
- If model choice or settings change, re-run `evaluate.py` and update the
  numbers in `README.md`.
