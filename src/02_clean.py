"""Step 2 - Clean the raw tweets and remove true duplicates  (Req 1).

Input : data/raw/training.1600000.processed.noemoticon.csv  (1.6M, emoticon labels)
        data/raw/testdata.manual.2009.06.14.csv             (498, hand labels, has neutral)
Output: data/interim/cleaned.csv       (clean, de-duplicated, no language info yet)
        reports/cleaning_log.txt       (how many rows each step removed)

Cleaning keeps the ORIGINAL text in `text_raw` so nothing is lost, and makes a
tidy `text_clean` for the models. Case and emojis are kept on purpose
(transformer models use them; TF-IDF can lowercase by itself).
"""
import html
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim" / "cleaned.csv"
LOG = ROOT / "reports" / "cleaning_log.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

# Language detection on 1.6M tweets would take hours, so we keep a random sample
# of the (cleaned, de-duplicated) big file. Set to None to keep everything.
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42

COLUMNS = ["target", "tweet_id", "date", "query", "user", "text"]
LABELS = {0: "negative", 2: "neutral", 4: "positive"}

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


# ---------- 1. Load both files and mark where each label came from ----------
train = pd.read_csv(RAW_DIR / "training.1600000.processed.noemoticon.csv",
                    names=COLUMNS, encoding="utf-8", encoding_errors="replace")
train["label_source"] = "emoticon"      # noisy: guessed from :) and :( in the tweet
train["topic_group"] = "unknown"        # the big file has no topic information
test = pd.read_csv(RAW_DIR / "testdata.manual.2009.06.14.csv",
                   names=COLUMNS, encoding="utf-8", encoding_errors="replace")
test["label_source"] = "manual"         # trustworthy: labelled by humans
test["topic_group"] = test["query"]     # the search term, e.g. "nike", "obama"

log(f"Raw rows loaded (train):                   {len(train):>9,}")
log(f"Raw rows loaded (manual test):            {len(test):>9,}")

df = pd.concat([train, test], ignore_index=True)
df = df.rename(columns={"text": "text_raw", "date": "created_at"})
df["label"] = df["target"].map(LABELS)

# ---------- 2. Basic sanity: missing values, valid labels ----------
before = len(df)
df = df.dropna(subset=["tweet_id", "text_raw", "label", "created_at"])
log(f"Removed missing / invalid-label rows:     {before - len(df):>9,}")

# ---------- 3. Dates: the big file says PDT, the test file says UTC -> convert all to UTC ----------
is_pdt = df["created_at"].str.contains(" PDT ", regex=False)
naive = pd.to_datetime(df["created_at"].str.replace(r" (PDT|UTC) ", " ", regex=True),
                       format="%a %b %d %H:%M:%S %Y")
df["created_at"] = (naive + pd.to_timedelta(is_pdt * 7, unit="h")).dt.tz_localize("UTC")

# ---------- 4. Text cleaning ----------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = html.unescape(text)          # "&amp;" -> "&", "&quot;" -> '"'
    text = text.replace("�", " ")  # bytes damaged in the source file (unrecoverable)
    text = URL_RE.sub(" ", text)        # links carry no sentiment
    text = MENTION_RE.sub(" ", text)    # "@user" is just who is being addressed
    text = text.replace("#", "")        # keep the hashtag word, drop the symbol
    text = SPACE_RE.sub(" ", text)      # collapse spaces / newlines
    return text.strip()


df["text_clean"] = df["text_raw"].map(clean_text)

before = len(df)
df = df[df["text_clean"].str.len() > 0]
log(f"Removed tweets empty after cleaning:      {before - len(df):>9,}")


# ---------- 5. De-duplication (true duplicates only) ----------
def make_key(text: str) -> str:
    """Lowercase letters/digits only - so 'Thanks!!' and 'thanks' match."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


df["text_key"] = df["text_clean"].map(make_key)

# 5a. Same tweet stored twice (same tweet_id)
before = len(df)
df = df.drop_duplicates(subset="tweet_id", keep="first")
log(f"Removed repeated tweet_id:                {before - len(df):>9,}")

# 5b. Same user posting the same text again (spam / re-post)
before = len(df)
df = df.drop_duplicates(subset=["user", "text_key"], keep="first")
log(f"Removed same-user identical re-posts:     {before - len(df):>9,}")

# 5c. Do NOT delete copy-paste from DIFFERENT users: that is repeated / coordinated
#     messaging, which Req 4 (similarity index) needs to see. We only flag it.
#     Counted on the FULL cleaned corpus, before sampling.
df["dup_group_size"] = df.groupby("text_key")["tweet_id"].transform("size")
log(f"Kept but flagged (same text, other users):{(df['dup_group_size'] > 1).sum():>9,}  -> column dup_group_size")

# ---------- 6. Sample the big file (keep ALL hand-labelled tweets) ----------
if SAMPLE_SIZE is not None:
    manual = df[df["label_source"] == "manual"]
    big = df[df["label_source"] == "emoticon"]
    if len(big) > SAMPLE_SIZE:
        big = big.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    log(f"Sampled emoticon-labelled tweets:         {len(big):>9,}  (random, seed {RANDOM_SEED})")
    df = pd.concat([big, manual])

# ---------- 7. Tidy up and save ----------
df["n_words"] = df["text_clean"].str.split().str.len()
df = df.drop(columns=["user", "target", "query"])   # usernames not needed downstream (privacy)
df = df.sort_values("created_at").reset_index(drop=True)
df.to_csv(OUT, index=False)

log(f"\nFinal rows after cleaning + dedup + sample:{len(df):>8,}")
log(f"Saved to {OUT}")
log("Label counts: " + str(df["label"].value_counts().to_dict()))
log("Label source: " + str(df["label_source"].value_counts().to_dict()))
LOG.write_text("\n".join(log_lines), encoding="utf-8")
