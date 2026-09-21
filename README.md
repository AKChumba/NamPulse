# NamPulse - Data & Preprocessing (Req 1 and Req 6)

Owner: Alvan Chumba. This part produces the **final cleaned dataset** the rest of the team uses.

## Use the data (teammates start here)

```python
import pandas as pd
df = pd.read_parquet("data/processed/nampulse_clean.parquet")   # or nampulse_clean.csv
```

Dataset: **Sentiment140** (Stanford, 2009), Apr-Jun 2009. After cleaning: 100,493 tweets =
100,000 random tweets from the 1.6M emoticon-labelled file + 493 hand-labelled tweets.

| Column | Meaning | Mostly used by |
|---|---|---|
| `tweet_id` | unique tweet id | everyone |
| `created_at` | timestamp, UTC (source was PDT/UTC, converted) | Req 3 / 7 (trends over time) |
| `label` | sentiment: positive / negative / neutral | Req 2 (train/evaluate) |
| `label_source` | `emoticon` = noisy guess from :) / :( ; `manual` = labelled by humans | Req 2 (**evaluate on `manual`**) |
| `topic_group` | search term for `manual` tweets (e.g. nike, obama); `unknown` for the rest | Req 3 / 7 |
| `text_raw` | original tweet, untouched (except damaged bytes) | Req 7 (show real examples) |
| `text_clean` | HTML unescaped, URLs and @mentions removed, `#` dropped; case and emojis kept | Req 2, 3, 4 (model input) |
| `language` | main language code (`en`, `af`, ...) or `und` (undetermined) | Req 6 / 8 |
| `language_conf` | 0-1. For English: how clearly English beats its closest rival language. For other languages: detector confidence | Req 6 |
| `is_english` | `language == "en"` | Req 2 / 3 |
| `is_code_switched` | candidate flag: a second language covers 3+ words (see limitations) | Req 6 / 8 |
| `languages_found` | languages spotted in the tweet, e.g. `en\|af` | Req 6 |
| `other_lang_words` | words outside the main language | Req 6 |
| `dup_group_size` | how many tweets in the full 1.6M corpus share this exact normalised text (1 = unique) | Req 4 |
| `n_words` | word count of `text_clean` | any |
| `text_key` | lowercase letters/digits-only version of the text, for exact-match grouping | Req 4 |

**Important for Req 2:** the big file has only positive and negative. **Neutral exists only in the 139
hand-labelled neutral tweets**, so a three-class model cannot be trained on this data; use binary
sentiment for training and the `manual` rows as a trustworthy test set.

## Rebuild it from scratch

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python src\01_download.py          # data/raw/*   (80 MB download)
.venv\Scripts\python src\02_clean.py             # data/interim/cleaned.csv  + reports/cleaning_log.txt
.venv\Scripts\python src\03_detect_language.py   # data/processed/*  + reports/language_report.txt  (10-15 min)
```

Notebooks in `notebooks/` (01 download+explore, 02 cleaning+dedup, 03 language detection) run the same scripts and
show the results with tables and charts - open them in VS Code or `jupyter notebook`. Notebook 03 loads the saved
result by default (set `RERUN = True` in its first cell to redo the 10-15 minute detection).

`data/raw` and `data/interim` are git-ignored (re-created by the scripts); `data/processed` is committed.
To use more (or all) of the 1.6M tweets, change `SAMPLE_SIZE` in `src/02_clean.py` (language detection then takes
correspondingly longer).

## What the pipeline does

1. **Download** - raw files saved untouched in `data/raw/`.
2. **Clean** - read as UTF-8 (the file has about 12,000 damaged bytes in the source; those are dropped),
   parse dates to UTC, tidy text (see table). Original kept in `text_raw`.
3. **De-duplicate** - removed 1,685 repeated `tweet_id`s and 10,884 identical re-posts by the same user.
   Identical text from *different* users is **kept and flagged** (`dup_group_size`, 80,082 rows in the full
   corpus) because Req 4 (similarity index) needs to see repeated / coordinated messaging.
   Near-duplicates (not exact) are left for Req 4.
4. **Sample** - 100,000 random tweets from the big file (seed 42) + all hand-labelled tweets.
5. **Language detection** - Lingua library, all 75 languages (a short list mislabelled Indonesian/Tagalog as
   African languages). Result on the 100,493 tweets: **92.7% `en`, 7.2% `und`, about 150 tweets (0.15%) in
   ~35 other languages** (Russian, Tagalog, Vietnamese, German, Indonesian, ...), and 317 code-switch candidates.
   Full table: `reports/language_report.txt`.

## Limitations (please read - useful for the Req 8 bias note)

- **Mostly English.** Only a small fraction of tweets are other languages (Vietnamese, Indonesian, Spanish,
  Portuguese, ...). There is essentially **no Afrikaans/Oshiwambo** here, so Req 6 can only be shown properly
  on the Namibia stretch-goal sample. The detector was checked on hand-written English/Afrikaans mixes:
  it caught 3 of 4 (and correctly left a plain-English control alone).
- **Emoticon labels are noisy.** Sentiment140's big file was labelled automatically from :) and :(, not by
  people. Trust the `manual` rows for evaluation.
- **Code-switch flag is a candidate flag, not ground truth.** Tuned for precision, but the ones I checked
  by eye on this data were mostly false alarms (slang like "yayyy", names). Review manually before reporting numbers.
- **Oshiwambo is not supported** by Lingua or any common detector; it will appear as `und` or a wrong
  language. Needs a small keyword list or manual labelling if local data is collected.
- **`und` (7.2%) is mostly very short or noisy tweets**, not foreign language (median 3 words vs 12 for
  English; 45% have under 3 words). Short real non-English tweets also end up as `und` (we prefer "unknown"
  to a wrong guess). `und` tweets are 59% positive vs 50% for English, so keep them separate when reporting.
- **Per-language sentiment for non-English is not meaningful yet**: most languages have fewer than 20 tweets.
- **Bias:** US-centric English Twitter from 2009 (a very different user base from Namibia today),
  one platform, tweets are only from ~50 distinct days, and the class balance (50/50) was set by the
  collection method, not by real-world sentiment.
- Usernames were dropped after de-duplication (privacy); `text_raw` may still contain @mentions.
