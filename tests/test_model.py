"""Tests for the model pipeline."""

from sklearn.linear_model import LogisticRegression

from model import build_pipeline

TEXTS = [
    "Free prize, claim now",
    "Win cash, call now",
    "Free entry, text WIN",
    "See you at lunch",
    "Are you coming home",
    "Ok, talk later",
]
LABELS = ["spam", "spam", "spam", "ham", "ham", "ham"]


def test_pipeline_predicts_valid_labels():
    pipeline = build_pipeline()
    pipeline.fit(TEXTS, LABELS)

    predictions = pipeline.predict(["Claim your free prize", "See you later"])

    assert set(predictions) <= {"ham", "spam"}
    assert list(predictions) == ["spam", "ham"]


def test_pipeline_accepts_another_classifier():
    pipeline = build_pipeline(LogisticRegression())
    pipeline.fit(TEXTS, LABELS)

    predictions = pipeline.predict(["Claim your free prize"])

    assert predictions[0] in {"ham", "spam"}
