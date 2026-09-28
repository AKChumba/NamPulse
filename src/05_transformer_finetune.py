"""Step 5 - Fine-tune DistilBERT on our own emoticon-labelled tweets  (Req 2b, second model).

Why a second transformer? Step 4 uses a model someone else fine-tuned on
different tweets. Here we fine-tune a general English model (distilbert-base-uncased)
on OUR data, so we can see whether training on our own (noisy) labels helps.
It can only learn positive vs negative, because that is all the big file has.

Input : data/processed/nampulse_clean.parquet
Output: models/distilbert-nampulse/            (the trained model - NOT in git, ~260 MB)
        data/processed/finetuned_test_predictions.csv   (predictions on the hand-labelled tweets)
        reports/transformer_finetune_report.txt
        reports/transformer_finetune_metrics.json
        reports/transformer_finetune_curve.png

Data split:
  train / validation : emoticon-labelled tweets. Each distinct text (`text_key`) is used
                       once, so the same tweet text can't be in train AND validation, and
                       any text that also appears in the hand-labelled set is removed.
  test               : the 354 hand-labelled positive/negative tweets (never trained on).

Speed: with a GPU (Colab T4) the default 20,000 tweets x 2 epochs takes ~5 minutes.
On a laptop CPU use something like `--n-train 4000 --epochs 1` (~30-60 minutes).
    python src/05_transformer_finetune.py
    python src/05_transformer_finetune.py --n-train 4000 --epochs 1
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from nampulse_transformer import DEVICE, predict_proba

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "nampulse_clean.parquet"
MODEL_DIR = ROOT / "models" / "distilbert-nampulse"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["negative", "positive"]          # index 0, 1

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", default="distilbert-base-uncased")
parser.add_argument("--n-train", type=int, default=20_000)
parser.add_argument("--n-val", type=int, default=2_000)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-5)      # usual range for BERT-style fine-tuning
parser.add_argument("--max-len", type=int, default=64)     # 99% of our tweets are shorter than this
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


# ---------- Data ----------
df = pd.read_parquet(IN)
test = df[(df["label_source"] == "manual") & (df["label"] != "neutral")].reset_index(drop=True)
pool = df[df["label_source"] == "emoticon"]
pool = pool[~pool["text_key"].isin(set(df.loc[df["label_source"] == "manual", "text_key"]))]
pool = pool.drop_duplicates("text_key")                      # one copy of each text
# balanced sample: half positive, half negative
per_class = (args.n_train + args.n_val) // 2
pool = pd.concat([g.sample(n=min(per_class, len(g)), random_state=args.seed)
                  for _, g in pool.groupby("label")]).sample(frac=1, random_state=args.seed)
val, train = pool.iloc[:args.n_val], pool.iloc[args.n_val:]
log(f"Base model: {args.base_model}   device: {DEVICE}")
log(f"Train {len(train):,}  {train['label'].value_counts().to_dict()}")
log(f"Val   {len(val):,}  {val['label'].value_counts().to_dict()}")
log(f"Test  {len(test):,}  {test['label'].value_counts().to_dict()}  (hand-labelled)")

tok = AutoTokenizer.from_pretrained(args.base_model)
model = AutoModelForSequenceClassification.from_pretrained(
    args.base_model, num_labels=2,
    id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
model.to(DEVICE)


def batches(frame, shuffle):
    """Yield tokenised batches. Padding is per batch, so short batches stay short."""
    idx = np.random.permutation(len(frame)) if shuffle else np.arange(len(frame))
    texts = frame["text_clean"].tolist()
    y = frame["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    for i in range(0, len(idx), args.batch_size):
        b = idx[i:i + args.batch_size]
        enc = tok([texts[j] for j in b], padding=True, truncation=True,
                  max_length=args.max_len, return_tensors="pt")
        enc["labels"] = torch.tensor(y[b])
        yield {k: v.to(DEVICE) for k, v in enc.items()}


def evaluate(frame):
    model.eval()
    probs = predict_proba(frame["text_clean"].tolist(), tok, model, batch_size=128)
    pred = [LABELS[k] for k in probs.argmax(1)]
    return accuracy_score(frame["label"], pred), pred, probs


# ---------- Train ----------
steps_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
total_steps = steps_per_epoch * args.epochs
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

history, best_val = [], -1
start = time.time()
for epoch in range(1, args.epochs + 1):
    model.train()
    losses = []
    for step, batch in enumerate(batches(train, shuffle=True), 1):
        loss = model(**batch).loss          # cross-entropy between logits and labels
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        losses.append(loss.item())
        if step % 100 == 0 or step == steps_per_epoch:
            print(f"  epoch {epoch} step {step}/{steps_per_epoch}  loss {np.mean(losses[-100:]):.4f}")
    val_acc, _, _ = evaluate(val)
    history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": val_acc})
    log(f"Epoch {epoch}: train loss {np.mean(losses):.4f}   val accuracy {val_acc:.3f}")
    if val_acc > best_val:                  # keep the best epoch (by validation, never by test)
        best_val = val_acc
        model.save_pretrained(MODEL_DIR); tok.save_pretrained(MODEL_DIR)
train_min = (time.time() - start) / 60
log(f"Training time: {train_min:.1f} min. Best val accuracy {best_val:.3f}, saved to {MODEL_DIR}")

# ---------- Test on the hand-labelled tweets ----------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
acc, pred, probs = evaluate(test)
f1 = f1_score(test["label"], pred, average="macro")
log("\n=== Hand-labelled tweets, positive vs negative ===")
log(f"n = {len(test)}   Accuracy: {acc:.3f}   Macro-F1: {f1:.3f}")
log(classification_report(test["label"], pred, digits=3, zero_division=0))
log("Confusion matrix (rows = true, columns = predicted):")
log(pd.DataFrame(confusion_matrix(test["label"], pred, labels=LABELS),
                 index=LABELS, columns=LABELS).to_string())

pd.DataFrame({"tweet_id": test["tweet_id"], "label": test["label"], "ft_label": pred,
              "ft_confidence": probs.max(1).round(4), "text_clean": test["text_clean"]}) \
    .to_csv(ROOT / "data" / "processed" / "finetuned_test_predictions.csv", index=False)

metrics = {"base_model": args.base_model, "device": DEVICE, "n_train": len(train), "n_val": len(val),
           "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "max_len": args.max_len,
           "train_minutes": round(train_min, 1), "history": history,
           "manual_binary": {"n": len(test), "accuracy": round(acc, 4), "macro_f1": round(f1, 4)}}
(REPORT_DIR / "transformer_finetune_metrics.json").write_text(json.dumps(metrics, indent=2))
(REPORT_DIR / "transformer_finetune_report.txt").write_text("\n".join(lines), encoding="utf-8")

h = pd.DataFrame(history)
fig, ax1 = plt.subplots(figsize=(5, 3.5))
ax1.plot(h["epoch"], h["train_loss"], "o-", label="train loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss")
ax2 = ax1.twinx()
ax2.plot(h["epoch"], h["val_accuracy"], "s--", color="tab:orange", label="val accuracy")
ax2.set_ylabel("val accuracy")
ax1.set_xticks(h["epoch"])
fig.legend(loc="upper center", ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(REPORT_DIR / "transformer_finetune_curve.png", dpi=150)
print(f"\nSaved report -> {REPORT_DIR / 'transformer_finetune_report.txt'}")
