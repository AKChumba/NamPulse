# NamPulse

Group project for an NLP course: a tool that tracks public sentiment in social media posts. This repo has the
first step, getting the data ready (cleaning it, removing duplicates and detecting the language of each post).

The dataset is [Sentiment140](http://help.sentiment140.com/for-students), about 1.6 million tweets from
April to June 2009. I use 100,000 random tweets from it, plus the 493 tweets that were labelled by hand.

## Using the data

```python
import pandas as pd
df = pd.read_parquet("data/processed/nampulse_clean.parquet")
```

There is also a CSV version in the same folder.

Columns:

- `tweet_id`, `created_at` (UTC)
- `label`: positive, negative or neutral
- `label_source`: `emoticon` or `manual` (see the notes below)
- `topic_group`: search term for the hand-labelled tweets (like nike or obama), `unknown` for the rest
- `text_raw`: the tweet as it was
- `text_clean`: URLs and @mentions removed, HTML characters fixed, `#` dropped. Case and emoji are kept.
- `language`: language code like `en`, or `und` when it can't tell
- `language_conf`: confidence from 0 to 1
- `is_english`: true when `language` is `en`
- `is_code_switched`, `languages_found`, `other_lang_words`: a rough guess at tweets that mix languages
- `dup_group_size`: how many tweets share exactly the same text (1 means it is unique)
- `n_words`, `text_key`: word count, and a lowercase letters-and-numbers version of the text for matching duplicates

## Things to know before using it

The big file only has positive and negative tweets. Neutral only shows up in the 139 hand-labelled tweets, so you
can't train a three-way classifier on this.

The 100,000 big-file labels were made automatically from smileys (:) and :( ), so they are noisy. The 493
`manual` tweets were labelled by people, so use those for testing.

Nearly everything is English (92.7%). About 7% are `und`, mostly very short tweets. Only around 150 tweets are
detected as another language, so there isn't enough here to say anything about sentiment per language.

The mixed-language flag is only a rough guess, and the flagged tweets I looked at were mostly slang or names, not
real mixing. It did catch 3 of 4 Afrikaans/English sentences I wrote to test it. Nothing I found can detect
Oshiwambo.

The tweets are from 2009 and mostly from the US, so they say little about Namibia today.

## Running it yourself

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python src\01_download.py
.venv\Scripts\python src\02_clean.py
.venv\Scripts\python src\03_detect_language.py
```

The first script downloads about 80 MB. The last one takes 10 to 15 minutes. The raw and in-between files are
not in the repo, but the scripts recreate them.

The notebooks in `notebooks/` go through the same three steps with tables and charts. Notebook 3 loads the saved
result unless you set `RERUN = True` in its first cell.

## What the scripts do

1. `01_download.py` downloads the raw files into `data/raw`.
2. `02_clean.py` cleans the text and removes duplicates: tweets stored twice (same id) and the same user posting
   the same text again. Together with tweets that were empty once links and mentions were removed, that drops
   about 15,000 tweets. Identical text from different users is kept and only counted in `dup_group_size`, since
   repeated messages might matter later. The script then takes the 100,000 sample. Change `SAMPLE_SIZE` at the
   top to use more.
3. `03_detect_language.py` detects the language with the [Lingua](https://github.com/pemistahl/lingua-py)
   library. Lingua struggles with very short informal tweets, so when another language wins only weakly, English
   is tested against that language one-on-one. Anything still unclear is marked `und`.

The Sentiment140 file has about 12,000 damaged characters in it that can't be recovered, so those are dropped.
