"""Step 4 - Pretrained transformer sentiment classifier  (Req 2b).

Model : cardiffnlp/twitter-roberta-base-sentiment-latest
        RoBERTa-base, further pre-trained on ~124M tweets and then fine-tuned for
        negative / neutral / positive on the TweetEval sentiment benchmark.
        We use it AS IS (no extra training) - the "prompting/using a pretrained
        model" option in the brief. It is our main transformer model because it
        can say "neutral", which a model trained on our emoticon labels cannot.

Input : data/processed/nampulse_clean.parquet
Output: data/processed/transformer_predictions.parquet (+ .csv)
            tweet_id, tf_label, tf_confidence, p_negative, p_neutral, p_positive
        reports/transformer_pretrained_report.txt
        reports/transformer_pretrained_metrics.json   (numbers for the report / dashboard)
        reports/transformer_pretrained_confusion.png

Evaluation:
  1. The 493 HAND-labelled tweets (the only trustworthy labels, and the only
     ones with neutral) -> accuracy, macro-F1, per-class scores, confusion matrix.
  2. Same tweets without neutral (354) -> positive vs negative, so we can compare
     fairly with the classical TF-IDF + logistic regression model (Req 2a).
  3. The 100k EMOTICON-labelled tweets -> agreement only. Those labels are noisy
     (a smiley is not always sentiment), so this is a sanity check, not accuracy.

Speed: on a laptop CPU the 100k tweets take roughly an hour. Use a GPU (Google
Colab is free) or `--limit 5000` for a quick run.
    python src/04_transformer_pretrained.py
    python src/04_transformer_pretrained.py --limit 5000
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)

from nampulse_transformer import DEVICE, label_names, load_model, predict_proba

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "nampulse_clean.parquet"
OUT = ROOT / "data" / "processed" / "transformer_predictions"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
RANDOM_SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--limit", type=int, default=None,
                    help="only score this many emoticon tweets (all hand-labelled tweets are always scored)")
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


# ---------- Load data ----------
df = pd.read_parquet(IN)
manual = df[df["label_source"] == "manual"]
emoticon = df[df["label_source"] == "emoticon"]
if args.limit:
    emoticon = emoticon.sample(n=min(args.limit, len(emoticon)), random_state=RANDOM_SEED)
work = pd.concat([manual, emoticon]).reset_index(drop=True)

# ---------- Predict ----------
log(f"Model: {args.model}   device: {DEVICE}")
tok, model = load_model(args.model)
names = label_names(model)
log(f"Model classes: {names}")
log(f"Scoring {len(work):,} tweets ({len(manual)} hand-labelled + {len(emoticon):,} emoticon) ...")

start = time.time()
probs = predict_proba(work["text_clean"].tolist(), tok, model, batch_size=args.batch_size)
secs = time.time() - start
log(f"Done in {secs / 60:.1f} min ({len(work) / secs:.0f} tweets/sec)")

pred = pd.DataFrame({"tweet_id": work["tweet_id"].values,
                     "tf_label": [names[k] for k in probs.argmax(1)],
                     "tf_confidence": probs.max(1).round(4)})
for k, n in enumerate(names):
    pred[f"p_{n}"] = probs[:, k].round(4)
pred["tf_model"] = args.model
pred.to_parquet(OUT.with_suffix(".parquet"), index=False)
pred.to_csv(OUT.with_suffix(".csv"), index=False)
work = pd.concat([work, pred.drop(columns="tweet_id")], axis=1)
metrics = {"model": args.model, "device": DEVICE, "n_scored": len(work),
           "tweets_per_sec": round(len(work) / secs, 1)}

# ---------- 1. Hand-labelled tweets, 3 classes ----------
m = work[work["label_source"] == "manual"]
log("\n=== 1. Hand-labelled tweets (3 classes) ===")
log(f"n = {len(m)}   {m['label'].value_counts().to_dict()}")
acc3 = accuracy_score(m["label"], m["tf_label"])
f13 = f1_score(m["label"], m["tf_label"], average="macro")
log(f"Accuracy: {acc3:.3f}   Macro-F1: {f13:.3f}")
log(classification_report(m["label"], m["tf_label"], labels=names, digits=3, zero_division=0))
cm = confusion_matrix(m["label"], m["tf_label"], labels=names)
log("Confusion matrix (rows = true, columns = predicted):")
log(pd.DataFrame(cm, index=names, columns=names).to_string())
metrics["manual_3class"] = {"n": len(m), "accuracy": round(acc3, 4), "macro_f1": round(f13, 4),
                            "report": classification_report(m["label"], m["tf_label"], labels=names,
                                                            output_dict=True, zero_division=0)}

# ---------- 2. Hand-labelled tweets, positive vs negative only ----------
# Neutral tweets are dropped, and the model picks the larger of p_positive / p_negative
# (ignoring its neutral score) - the same task the classical model is trained for.
b = m[m["label"] != "neutral"]
b_pred = np.where(b["p_positive"] >= b["p_negative"], "positive", "negative")
acc2 = accuracy_score(b["label"], b_pred)
f12 = f1_score(b["label"], b_pred, average="macro")
log("\n=== 2. Hand-labelled tweets, positive vs negative (comparable with Req 2a) ===")
log(f"n = {len(b)}   Accuracy: {acc2:.3f}   Macro-F1: {f12:.3f}")
log(classification_report(b["label"], b_pred, digits=3, zero_division=0))
metrics["manual_binary"] = {"n": len(b), "accuracy": round(acc2, 4), "macro_f1": round(f12, 4)}

# ---------- 3. Emoticon-labelled tweets: agreement ----------
e = work[work["label_source"] == "emoticon"].copy()
if len(e):
    e["pn_pred"] = np.where(e["p_positive"] >= e["p_negative"], "positive", "negative")
    agree = (e["pn_pred"] == e["label"]).mean()
    log("\n=== 3. Emoticon-labelled tweets (noisy labels - agreement, not accuracy) ===")
    log(f"n = {len(e):,}   agreement (pos vs neg): {agree:.3f}")
    log("Model's 3-class labels on these tweets (%):")
    log((e["tf_label"].value_counts(normalize=True) * 100).round(1).to_string())
    log("\nAgreement by detected language (Req 6 - how non-English text is handled):")
    by_lang = e.assign(lang=e["language"].where(e["language"].isin(["en", "und"]), "other"))
    by_lang = by_lang.groupby("lang").apply(
        lambda g: pd.Series({"tweets": len(g), "agreement": (g["pn_pred"] == g["label"]).mean(),
                             "avg_confidence": g["tf_confidence"].mean()}), include_groups=False)
    log(by_lang.round(3).to_string())
    log("\nAgreement for tweets flagged as code-switched vs not:")
    by_cs = e.groupby("is_code_switched").apply(
        lambda g: pd.Series({"tweets": len(g), "agreement": (g["pn_pred"] == g["label"]).mean()}),
        include_groups=False)
    log(by_cs.round(3).to_string())
    metrics["emoticon_agreement"] = {"n": len(e), "agreement": round(agree, 4),
                                     "by_language": by_lang.round(4).to_dict(orient="index")}

# ---------- Save ----------
(REPORT_DIR / "transformer_pretrained_report.txt").write_text("\n".join(lines), encoding="utf-8")
(REPORT_DIR / "transformer_pretrained_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(names)), names)
ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Twitter-RoBERTa, hand-labelled tweets\naccuracy {acc3:.2f}, macro-F1 {f13:.2f}")
fig.tight_layout()
fig.savefig(REPORT_DIR / "transformer_pretrained_confusion.png", dpi=150)

print(f"\nSaved predictions -> {OUT.with_suffix('.parquet')}")
print(f"Saved report      -> {REPORT_DIR / 'transformer_pretrained_report.txt'}")
