"""Step 1 - Download the raw dataset (Sentiment140).

Source: Stanford (Go, Bhayani & Huang, 2009) - 1.6 million tweets, Apr-Jun 2009.
Output: data/raw/training.1600000.processed.noemoticon.csv   (1.6M tweets, labels from emoticons)
        data/raw/testdata.manual.2009.06.14.csv              (498 tweets, labelled BY HAND, has neutral)
Raw data stays untouched after this step.
"""
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
ZIP_PATH = RAW_DIR / "trainingandtestdata.zip"

# The files have no header row, so we name the columns ourselves.
COLUMNS = ["target", "tweet_id", "date", "query", "user", "text"]

if not ZIP_PATH.exists():
    print("Downloading (about 80 MB) ...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
with zipfile.ZipFile(ZIP_PATH) as z:
    z.extractall(RAW_DIR)
print("Files in data/raw:", sorted(p.name for p in RAW_DIR.iterdir()))

train = pd.read_csv(RAW_DIR / "training.1600000.processed.noemoticon.csv",
                    names=COLUMNS, encoding="utf-8", encoding_errors="replace")
test = pd.read_csv(RAW_DIR / "testdata.manual.2009.06.14.csv",
                   names=COLUMNS, encoding="utf-8", encoding_errors="replace")
print(f"Train rows: {len(train):,}   Manual test rows: {len(test):,}")
print(train.head())
