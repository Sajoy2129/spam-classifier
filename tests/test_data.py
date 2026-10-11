"""Tests for loading the dataset."""

import pytest

from data import load_data
from download_data import DATA_FILE


def write_fake_dataset(folder, lines):
    """Write a small tab-separated file shaped like the real dataset."""
    path = folder / "fake_sms.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_data_columns_and_labels(tmp_path):
    path = write_fake_dataset(
        tmp_path,
        [
            "ham\tSee you at 5?",
            "spam\tWIN a free prize now!",
            "ham\tOk thanks",
        ],
    )

    messages = load_data(path)

    assert list(messages.columns) == ["label", "message"]
    assert len(messages) == 3
    assert set(messages["label"]) == {"ham", "spam"}


def test_load_data_keeps_quote_characters(tmp_path):
    # A lone " must not swallow the lines that follow it.
    path = write_fake_dataset(
        tmp_path,
        [
            'ham\tShe said "hello and left',
            "spam\tFree entry, text WIN",
            "ham\tOn my way",
        ],
    )

    messages = load_data(path)

    assert len(messages) == 3
    assert messages.loc[0, "message"] == 'She said "hello and left'


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_data(tmp_path / "does_not_exist.tsv")


@pytest.mark.skipif(
    not DATA_FILE.exists(), reason="real dataset has not been downloaded"
)
def test_real_dataset_shape():
    messages = load_data()

    assert len(messages) == 5574
    assert set(messages["label"]) == {"ham", "spam"}
    assert messages["message"].notna().all()
    assert (messages["message"].str.strip() != "").all()
