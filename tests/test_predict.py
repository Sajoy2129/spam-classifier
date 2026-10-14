"""Tests for classifying single messages."""

import pytest

from model import build_pipeline
from predict import classify, format_result, load_model

TEXTS = [
    "Free prize, claim now",
    "Win cash, call now",
    "Free entry, text WIN",
    "See you at lunch",
    "Are you coming home",
    "Ok, talk later",
]
LABELS = ["spam", "spam", "spam", "ham", "ham", "ham"]


def test_classify_returns_label_and_confidence():
    pipeline = build_pipeline()
    pipeline.fit(TEXTS, LABELS)

    label, confidence = classify(pipeline, "Claim your free prize now")

    assert label == "spam"
    # The winning label always has at least half the probability.
    assert 0.5 <= confidence <= 1


def test_format_result():
    assert format_result("spam", 0.932) == "SPAM (93.2% confident)"
    assert format_result("ham", 1.0) == "HAM (100.0% confident)"


def test_load_model_missing_file_says_run_train(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_model(tmp_path / "no_model.joblib")
