"""Evaluate the models properly: confusion matrix, precision, recall, F1."""

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score

from data import load_data, remove_duplicates, split_data
from model import build_pipeline
from train import FINAL_MODEL, MODELS

LABELS = ["ham", "spam"]


def score_predictions(actual, predicted):
    """Return accuracy, precision, recall, F1 and the confusion matrix.

    Precision, recall and F1 are all measured for the 'spam' class.
    """
    # zero_division=0: a model that never says spam has no precision to
    # measure, so we report 0 instead of getting a warning.
    spam_scores = {"pos_label": "spam", "zero_division": 0}
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)

    return {
        "accuracy": (actual == predicted).mean(),
        "precision": precision_score(actual, predicted, **spam_scores),
        "recall": recall_score(actual, predicted, **spam_scores),
        "f1": f1_score(actual, predicted, **spam_scores),
        # Rows are the true labels, columns are the model's guesses.
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(actual, predicted, labels=LABELS),
            index=[f"actual {label}" for label in LABELS],
            columns=[f"predicted {label}" for label in LABELS],
        ),
    }


def find_mistakes(texts, actual, predicted):
    """Return the messages the model got wrong, with both labels."""
    results = pd.DataFrame(
        {
            "message": list(texts),
            "actual": list(actual),
            "predicted": list(predicted),
        }
    )
    wrong = results[results["actual"] != results["predicted"]]
    return wrong.reset_index(drop=True)


def cross_validate_f1(classifier, texts, labels, folds=5):
    """Return one spam F1 score per fold of cross-validation.

    The whole pipeline is refitted inside every fold, so the vectorizer
    never sees the slice it is about to be scored on (no data leakage).
    """
    spam_f1 = make_scorer(f1_score, pos_label="spam", zero_division=0)
    return cross_val_score(
        build_pipeline(classifier), texts, labels, cv=folds, scoring=spam_f1
    )


def print_report(name, scores):
    """Print the scores for one model."""
    print(f"--- {name} ---")
    print(scores["confusion_matrix"])
    print(f"Accuracy:  {scores['accuracy']:.1%}")
    print(f"Precision: {scores['precision']:.1%}")
    print(f"Recall:    {scores['recall']:.1%}")
    print(f"F1:        {scores['f1']:.1%}")
    print()


def print_mistakes(mistakes):
    """Print each misclassified message under a heading for its kind."""
    kinds = {
        "Spam that got through (false negatives)": "spam",
        "Real messages flagged as spam (false positives)": "ham",
    }
    for heading, actual_label in kinds.items():
        messages = mistakes[mistakes["actual"] == actual_label]["message"]
        print(f"{heading}: {len(messages)}")
        for message in messages:
            print(f"  - {message}")
        print()


def evaluate():
    """Score every model on the test set, then cross-validate on training."""
    messages = remove_duplicates(load_data())
    train_texts, test_texts, train_labels, test_labels = split_data(messages)

    print("=== Test set scores ===")
    print()
    for name, classifier in MODELS.items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(train_texts, train_labels)
        predictions = pipeline.predict(test_texts)

        print_report(name, score_predictions(test_labels, predictions))
        if name == FINAL_MODEL:
            mistakes = find_mistakes(test_texts, test_labels, predictions)

    print("=== 5-fold cross-validation on the training set (spam F1) ===")
    print()
    for name, classifier in MODELS.items():
        fold_scores = cross_validate_f1(classifier, train_texts, train_labels)
        per_fold = "  ".join(f"{score:.1%}" for score in fold_scores)
        print(f"{name:<24}mean {fold_scores.mean():.1%}   folds: {per_fold}")
    print()

    print(f"=== Mistakes made by {FINAL_MODEL} ===")
    print()
    print_mistakes(mistakes)


if __name__ == "__main__":
    evaluate()
