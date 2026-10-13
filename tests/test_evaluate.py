"""Tests for the evaluation helpers."""

import joblib
import pandas as pd
import pytest
from sklearn.naive_bayes import MultinomialNB

import train
from evaluate import cross_validate_f1, find_mistakes, score_predictions

# 3 real spam and 5 real ham. The "model" catches 2 of the 3 spam
# and wrongly flags 1 ham message.
ACTUAL = ["spam", "spam", "spam", "ham", "ham", "ham", "ham", "ham"]
PREDICTED = ["spam", "spam", "ham", "spam", "ham", "ham", "ham", "ham"]


def test_score_predictions_known_values():
    scores = score_predictions(ACTUAL, PREDICTED)

    assert scores["accuracy"] == pytest.approx(6 / 8)
    assert scores["precision"] == pytest.approx(2 / 3)  # 2 of 3 flagged
    assert scores["recall"] == pytest.approx(2 / 3)  # 2 of 3 real spam
    assert scores["f1"] == pytest.approx(2 / 3)

    matrix = scores["confusion_matrix"]
    assert matrix.loc["actual ham", "predicted ham"] == 4
    assert matrix.loc["actual ham", "predicted spam"] == 1
    assert matrix.loc["actual spam", "predicted ham"] == 1
    assert matrix.loc["actual spam", "predicted spam"] == 2


def test_score_predictions_always_ham_scores_zero_on_spam():
    scores = score_predictions(ACTUAL, ["ham"] * len(ACTUAL))

    assert scores["accuracy"] == pytest.approx(5 / 8)
    assert scores["precision"] == 0
    assert scores["recall"] == 0
    assert scores["f1"] == 0


def test_find_mistakes_returns_only_wrong_guesses():
    texts = [f"message {i}" for i in range(len(ACTUAL))]

    mistakes = find_mistakes(texts, ACTUAL, PREDICTED)

    assert list(mistakes["message"]) == ["message 2", "message 3"]
    assert list(mistakes["actual"]) == ["spam", "ham"]
    assert list(mistakes["predicted"]) == ["ham", "spam"]


def test_cross_validate_f1_gives_one_score_per_fold():
    texts = ["free prize claim now", "see you at lunch"] * 6
    labels = ["spam", "ham"] * 6

    fold_scores = cross_validate_f1(MultinomialNB(), texts, labels, folds=3)

    assert len(fold_scores) == 3
    assert all(0 <= score <= 1 for score in fold_scores)


def test_train_final_model_saves_a_working_pipeline(tmp_path, monkeypatch):
    fake_messages = pd.DataFrame(
        {
            "label": ["spam", "ham"] * 10,
            "message": [
                text
                for i in range(10)
                for text in (f"free prize claim now {i}", f"see you at {i}")
            ],
        }
    )
    monkeypatch.setattr(train, "load_data", lambda: fake_messages)
    model_file = tmp_path / "models" / "spam_pipeline.joblib"

    train.train_final_model(model_file)

    saved_pipeline = joblib.load(model_file)
    assert saved_pipeline.predict(["free prize"])[0] == "spam"
