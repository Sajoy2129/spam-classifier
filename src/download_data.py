"""Download the SMS Spam Collection dataset from UCI into the data/ folder."""

import urllib.request
import zipfile
from pathlib import Path

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)

# Path(__file__) is this script's location, so these paths work no matter
# which folder you run the script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "SMSSpamCollection"

# The only files we want out of the zip archive.
WANTED_FILES = ["SMSSpamCollection", "readme"]


def download_dataset():
    """Download and unzip the dataset, unless it is already on disk."""
    if DATA_FILE.exists():
        print(f"Already downloaded: {DATA_FILE}")
        return DATA_FILE

    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "sms_spam_collection.zip"

    print(f"Downloading {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        for name in WANTED_FILES:
            archive.extract(name, DATA_DIR)

    zip_path.unlink()  # the zip is no longer needed once unpacked
    print(f"Saved dataset to: {DATA_FILE}")
    return DATA_FILE


if __name__ == "__main__":
    download_dataset()
