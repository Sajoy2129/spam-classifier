"""Train several models and compare them on the test set."""

from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from data import load_data, remove_duplicates, split_data
from download_data import PROJECT_ROOT
from model import build_pipeline

# The baseline always answers with the most common label (ham).
# A real model has to beat it to be worth anything.
MODELS = {
    "Always ham (baseline)": DummyClassifier(strategy="most_frequent"),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}

# Chosen in Phase 5 from the numbers `python src/evaluate.py` prints.
FINAL_MODEL = "Logistic Regression"
MODEL_FILE = PROJECT_ROOT / "models" / "spam_pipeline.joblib"


def train_and_compare():
    """Fit every model on the training set and score it on the test set."""
    messages = remove_duplicates(load_data())
    train_texts, test_texts, train_labels, test_labels = split_data(messages)

    print(f"Training messages: {len(train_texts)}")
    print(f"Test messages:     {len(test_texts)}")
    print()

    spam_in_test = (test_labels == "spam").sum()
    results = {}

    print(f"{'Model':<24}{'Accuracy':>10}{'Spam caught':>16}")
    for name, classifier in MODELS.items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(train_texts, train_labels)

        predictions = pipeline.predict(test_texts)
        accuracy = (predictions == test_labels).mean()
        spam_caught = ((predictions == "spam") & (test_labels == "spam")).sum()

        results[name] = accuracy
        caught = f"{spam_caught} of {spam_in_test}"
        print(f"{name:<24}{accuracy:>10.1%}{caught:>16}")

    return results


def train_final_model(path=MODEL_FILE):
    """Fit the final model on the training set and save it to disk."""
    messages = remove_duplicates(load_data())
    train_texts, _, train_labels, _ = split_data(messages)

    pipeline = build_pipeline(MODELS[FINAL_MODEL])
    pipeline.fit(train_texts, train_labels)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)  # write the fitted pipeline to one file
    print(f"Saved {FINAL_MODEL} to: {path}")
    return pipeline


if __name__ == "__main__":
    train_and_compare()
    print()
    train_final_model()
