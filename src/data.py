"""Load the SMS Spam Collection dataset into a pandas DataFrame."""

import csv
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from download_data import DATA_FILE


def load_data(path=DATA_FILE):
    """Return the dataset as a DataFrame with 'label' and 'message' columns.

    Each line of the file looks like:  ham<TAB>See you at 5?
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run `python src/download_data.py` first."
        )

    return pd.read_csv(
        path,
        sep="\t",  # columns are separated by a tab character
        header=None,  # the file has no header row...
        names=["label", "message"],  # ...so we name the columns ourselves
        quoting=csv.QUOTE_NONE,  # treat " as normal text, not a field wrapper
        # By default pandas turns text like "NA" or an empty field into a
        # missing value (NaN), which the vectorizer cannot read. Keep it text.
        keep_default_na=False,
    )


def remove_duplicates(messages):
    """Return the messages with repeated message texts dropped.

    A message that sits in both the training and the test data would let
    the model be tested on something it has already seen.
    """
    return messages.drop_duplicates(subset="message").reset_index(drop=True)


def split_data(messages, test_size=0.2, random_state=42):
    """Split into training and test sets, keeping the spam ratio in both.

    Returns: train_texts, test_texts, train_labels, test_labels
    """
    return train_test_split(
        messages["message"],
        messages["label"],
        test_size=test_size,
        random_state=random_state,  # same split every run
        stratify=messages["label"],  # same share of spam in both parts
    )


if __name__ == "__main__":
    messages = load_data()
    print(f"Shape (rows, columns): {messages.shape}")
    print()
    print(messages["label"].value_counts())
    print()
    print(messages.head())
