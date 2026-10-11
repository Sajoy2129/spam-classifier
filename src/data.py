"""Load the SMS Spam Collection dataset into a pandas DataFrame."""

import csv
from pathlib import Path

import pandas as pd

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
    )


if __name__ == "__main__":
    messages = load_data()
    print(f"Shape (rows, columns): {messages.shape}")
    print()
    print(messages["label"].value_counts())
    print()
    print(messages.head())
