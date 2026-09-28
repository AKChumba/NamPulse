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

The first script downloads about 80 MB. The third one takes 10 to 15 minutes. The raw and in-between files are
not in the repo, but the scripts recreate them.

The notebooks in `notebooks/` go through the same steps with tables and charts. Notebook 3 loads the saved
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

## Transformer classifier and explanations (Req 2b, part of Req 7)

Two transformer models, both evaluated on the hand-labelled tweets:

- **Model A, `cardiffnlp/twitter-roberta-base-sentiment-latest`** (main model). RoBERTa that was already trained on
  millions of tweets and fine-tuned for negative / neutral / positive. We use it as it is. It is the only model we have
  that can say neutral, because our big file has no neutral tweets to train on.
- **Model B, `distilbert-base-uncased` fine-tuned by us** on 20,000 emoticon-labelled tweets (positive / negative
  only). This shows whether training on our own noisy labels helps.

For the comparison with the classical model (Req 2a), both are also scored on the 354 hand-labelled positive/negative
tweets.

Every label can be explained word by word with Integrated Gradients: each word gets a score for how much it pushed the
tweet towards the predicted label. `src/nampulse_transformer.py` has the helpers the dashboard can import:

```python
import sys; sys.path.insert(0, "src")
from nampulse_transformer import load_model, predict, explain, to_html
tok, model = load_model("cardiffnlp/twitter-roberta-base-sentiment-latest")
predict(["the water is off again"], tok, model)      # list of (label, confidence)
explain("the water is off again", tok, model)        # label, probabilities and a score per word
```

Run it (a GPU makes a big difference; notebook 4 runs on Google Colab with *Runtime -> T4 GPU*):

```
.venv\Scripts\python src\04_transformer_pretrained.py      (100k tweets: ~3 min GPU, ~1 hour CPU; add --limit 5000 to test)
.venv\Scripts\python src\05_transformer_finetune.py        (~5 min GPU; on CPU use --n-train 4000 --epochs 1)
.venv\Scripts\python src\06_transformer_explain.py         (~2 min GPU, ~20 min CPU)
```

The models are downloaded from Hugging Face the first time (about 500 MB and 250 MB).

Outputs:

- `data/processed/transformer_predictions.parquet`: `tweet_id`, `tf_label`, `tf_confidence`, `p_negative`,
  `p_neutral`, `p_positive` (model A). Join on `tweet_id`.
- `data/processed/transformer_explanations.jsonl`: word scores for each hand-labelled tweet, ready for highlighting.
- `reports/transformer_*`: results, confusion matrix, training curve, top words, and highlighted examples
  (`transformer_explanations.html`).
- `models/distilbert-nampulse/`: the fine-tuned model B (not in git, too big; `src/05` recreates it).

Things to know:

- The `emoticon` labels are only used to train model B and as a rough agreement check. Accuracy numbers come from the
  `manual` tweets only.
- For model B, each distinct text is used once and texts that are also in the hand-labelled set are removed, so the
  test tweets are never seen in training.
- The explanations start from a tweet where every word is replaced by `<mask>`. We use `<mask>` instead of `<pad>`
  because RoBERTa gives `<pad>` tokens special positions. With `<pad>` the word scores did not add up to the change
  in the model's output.
- `src/06` also checks the Integrated Gradients scores against a simpler test: remove one word and see how much the
  prediction changes.

### Results (hand-labelled tweets)

| Model | Pos/neg accuracy (354) | Pos/neg macro-F1 | 3-class accuracy (493) | 3-class macro-F1 |
|---|---|---|---|---|
| A: Twitter-RoBERTa (pretrained) | 0.927 | 0.926 | 0.868 | 0.866 |
| B: DistilBERT (fine-tuned on 20k emoticon tweets) | 0.859 | 0.859 | - | - |

Model A is better on the same test tweets (McNemar exact test, p ≈ 0.0007). Tweets where model A's confidence is
below 0.6 (9% of them) are only about 50% accurate, so the dashboard should mark them as uncertain.

Known limitation: the Integrated Gradients word scores add up correctly for 415 of the 493 explained tweets
(`ig_delta` ≤ 0.05). For the other 78 they are approximate.