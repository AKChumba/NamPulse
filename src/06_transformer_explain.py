"""Step 6 - Explain the transformer's labels word by word  (Req 7, transformer part).

For each hand-labelled tweet we compute Integrated Gradients word scores
(see src/nampulse_transformer.py) and save them so the dashboard can highlight
the words behind every label without re-running the model.

Input : data/processed/nampulse_clean.parquet
Output: data/processed/transformer_explanations.jsonl   one JSON line per tweet:
            {tweet_id, text, true_label, label, confidence, probs, words: [[word, score], ...]}
        reports/transformer_top_words.csv    words that most often push towards each label
        reports/transformer_explanations.html  highlighted examples (right and wrong) for the slides
        reports/transformer_explain_report.txt includes the IG vs occlusion sanity check

    python src/06_transformer_explain.py              (493 tweets: ~2 min GPU, ~20 min CPU)
    python src/06_transformer_explain.py --limit 50
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nampulse_transformer import explain, explain_occlusion, load_model, to_html

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "nampulse_clean.parquet"
OUT = ROOT / "data" / "processed" / "transformer_explanations.jsonl"
REPORT_DIR = ROOT / "reports"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--n-check", type=int, default=30, help="tweets used for the IG vs occlusion check")
args = parser.parse_args()

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


df = pd.read_parquet(IN)
manual = df[df["label_source"] == "manual"].reset_index(drop=True)
if args.limit:
    manual = manual.sample(n=min(args.limit, len(manual)), random_state=42).reset_index(drop=True)

tok, model = load_model(args.model)
log(f"Explaining {len(manual)} hand-labelled tweets with Integrated Gradients ({args.model})")

records = []
for i, row in manual.iterrows():
    e = explain(row["text_clean"], tok, model)
    records.append({"tweet_id": int(row["tweet_id"]), "text": row["text_clean"],
                    "true_label": row["label"], **e})
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(manual)}")
with open(OUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
log(f"Saved {len(records)} explanations -> {OUT}")
log(f"IG completeness gap (ig_delta): median {np.median([r['ig_delta'] for r in records]):.4f}, "
    f"max {max(r['ig_delta'] for r in records):.4f}  (small = attributions add up correctly)")

# ---------- Global view: which words push towards each label? ----------
rows = []
for r in records:
    for word, score in r["words"]:
        w = re.sub(r"[^\w']", "", word.lower())
        if w:
            rows.append({"predicted": r["label"], "word": w, "score": score})
words = pd.DataFrame(rows)
top = (words.groupby(["predicted", "word"])["score"].agg(["mean", "count"]).reset_index()
       .query("count >= 3").sort_values(["predicted", "mean"], ascending=[True, False]))
top.groupby("predicted").head(20).to_csv(REPORT_DIR / "transformer_top_words.csv", index=False)
log("\nWords that most strongly support each predicted label (mean IG score, seen >= 3 times):")
for label, g in top.groupby("predicted"):
    log(f"  {label:>8}: " + ", ".join(f"{w} ({m:+.2f})" for w, m in zip(g["word"].head(12), g["mean"].head(12))))

# ---------- Sanity check: does IG agree with the simple leave-one-word-out method? ----------
# If IG were just noise, its word ranking would not match what happens when we
# actually delete the words. Spearman correlation per tweet, then the average.
check = [r for r in records if len(r["words"]) >= 4][:args.n_check]
rhos, top_match = [], []
for r in check:
    occ = explain_occlusion(r["text"], tok, model, target=r["label"])
    ig_s = np.array([s for _, s in r["words"]])
    oc_s = np.array([s for _, s in occ["words"]])
    if len(ig_s) == len(oc_s) and ig_s.std() > 0 and oc_s.std() > 0:
        rhos.append(spearmanr(ig_s, oc_s).statistic)
        top_match.append(int(ig_s.argmax() == oc_s.argmax()))
if rhos:
    log(f"\nIG vs occlusion on {len(rhos)} tweets: mean Spearman rho {np.mean(rhos):.2f}, "
        f"same most-important word in {np.mean(top_match):.0%} of tweets")

# ---------- Highlighted examples for the slides / report ----------
res = pd.DataFrame(records)
right = res[res["label"] == res["true_label"]].sort_values("confidence", ascending=False)
wrong = res[res["label"] != res["true_label"]].sort_values("confidence", ascending=False)
examples = pd.concat([right.groupby("label").head(3), wrong.head(8)])
html = ["<html><head><meta charset='utf-8'><title>NamPulse - transformer explanations</title></head>",
        "<body style='font-family:sans-serif;max-width:900px;margin:auto'>",
        "<h2>Why did the transformer choose this label?</h2>",
        "<p>Green words pushed the tweet towards the predicted label, red words pushed against it "
        "(Integrated Gradients). Hover a word to see its score.</p>"]
for title, part in [("Correct predictions", examples[examples["label"] == examples["true_label"]]),
                    ("Mistakes", examples[examples["label"] != examples["true_label"]])]:
    html.append(f"<h3>{title}</h3>")
    for _, r in part.iterrows():
        html.append(f"<p>{to_html(r)} &nbsp; <i>(true: {r['true_label']})</i></p>")
html.append("</body></html>")
(REPORT_DIR / "transformer_explanations.html").write_text("\n".join(html), encoding="utf-8")
(REPORT_DIR / "transformer_explain_report.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved examples -> {REPORT_DIR / 'transformer_explanations.html'}")
