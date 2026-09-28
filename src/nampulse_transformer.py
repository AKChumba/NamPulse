"""Shared helpers for the transformer sentiment classifier  (Req 2b + Req 7).

Used by src/04, src/05 and src/06, and meant to be imported by the dashboard:

    from nampulse_transformer import load_model, predict, explain, to_html
    tok, model = load_model("cardiffnlp/twitter-roberta-base-sentiment-latest")
    predict(["the water is off again"], tok, model)
    explain("the water is off again", tok, model)     # word -> score, for highlighting

Explainability (Req 7) uses Integrated Gradients (Sundararajan et al., 2017):
start from a "blank" input (every word replaced by <mask>), move step by step to
the real input, and add up how much the predicted class score changes because of
each token along the way. A positive score means the word pushed the tweet
TOWARDS the predicted label; negative means it pushed against it. Word scores
are the sum of their sub-word pieces ("unacceptable" -> "un", "accept", "able").

There is also `explain_occlusion`: remove one word at a time and see how much the
predicted probability drops. It is slower and cruder, but trivial to explain, and
we use it as a sanity check that IG is not showing noise.
"""
import html as html_lib
import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 128          # tweets are short; 128 sub-word tokens is far more than needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- loading
def load_model(name_or_path):
    """Load a tokenizer + classifier from the Hugging Face Hub or a local folder."""
    tok = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(name_or_path)
    model.to(DEVICE).eval()
    return tok, model


def label_names(model):
    """Class names in index order, lowercase, e.g. ['negative', 'neutral', 'positive']."""
    return [model.config.id2label[i].lower() for i in range(model.config.num_labels)]


# ---------------------------------------------------------------- prediction
@torch.no_grad()
def predict_proba(texts, tok, model, batch_size=64):
    """Return an (n_texts, n_classes) array of softmax probabilities."""
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i + batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, model.config.num_labels))


def predict(texts, tok, model, batch_size=64):
    """Return a list of (label, confidence) pairs."""
    names = label_names(model)
    probs = predict_proba(texts, tok, model, batch_size)
    return [(names[k], float(p[k])) for p, k in zip(probs, probs.argmax(1))]


# ---------------------------------------------------------------- explanation
def _words_with_spans(text):
    """Whitespace-separated words and their (start, end) character positions."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _tokens_to_words(text, offsets):
    """For each sub-word token, the index of the whitespace word it belongs to (or -1).

    We use the tokenizer's character offsets instead of looking at token strings,
    so this works for any tokenizer (RoBERTa's 'Ġ' pieces, BERT's '##' pieces, ...).
    """
    words = _words_with_spans(text)
    owner = []
    for s, e in offsets:
        if e <= s:                       # special tokens (<s>, </s>, [CLS]) have empty spans
            owner.append(-1)
            continue
        while s < e and text[s].isspace():   # RoBERTa offsets can include the leading space
            s += 1
        owner.append(next((w for w, (_, ws, we) in enumerate(words) if ws <= s < we), -1))
    return words, owner


def explain(text, tok, model, target=None, n_steps=50):
    """Integrated Gradients word attributions for one text.

    Returns a dict: label, confidence, probs {class: p}, words [(word, score), ...].
    Scores are normalised so the largest absolute score is 1.0.
    """
    from captum.attr import LayerIntegratedGradients

    names = label_names(model)
    enc = tok(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True,
              return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(input_ids=input_ids, attention_mask=mask).logits, -1)[0]
    if target is None:
        target = int(probs.argmax())
    elif isinstance(target, str):
        target = names.index(target)

    # Baseline = "a tweet with no words": same length, special tokens kept, every real
    # token replaced by <mask>. We use <mask> rather than <pad> because RoBERTa treats
    # <pad> tokens specially when it numbers word positions, which would change more than
    # just the words.
    special = {tok.cls_token_id, tok.sep_token_id, tok.bos_token_id, tok.eos_token_id,
               tok.pad_token_id} - {None}
    blank_id = tok.mask_token_id if tok.mask_token_id is not None else tok.pad_token_id
    baseline = input_ids.clone()
    for j, t in enumerate(input_ids[0].tolist()):
        if t not in special:
            baseline[0, j] = blank_id

    def forward(ids, attn):
        return model(input_ids=ids, attention_mask=attn).logits

    lig = LayerIntegratedGradients(forward, model.get_input_embeddings())
    attr, delta = lig.attribute(inputs=input_ids, baselines=baseline, target=target,
                                additional_forward_args=(mask,), n_steps=n_steps,
                                internal_batch_size=n_steps, return_convergence_delta=True)
    token_scores = attr.sum(dim=-1)[0].detach().cpu().numpy()   # one number per token

    words, owner = _tokens_to_words(text, offsets)
    word_scores = np.zeros(len(words))
    for score, w in zip(token_scores, owner):
        if w >= 0:
            word_scores[w] += score
    top = np.abs(word_scores).max() if len(words) else 0
    if top > 0:
        word_scores = word_scores / top

    return {"label": names[target], "confidence": float(probs[target]),
            "probs": {n: round(float(p), 4) for n, p in zip(names, probs)},
            # completeness check: raw scores should add up to (score of input - score of
            # baseline); ig_delta is the gap. Near 0 = enough steps were used.
            "ig_delta": round(float(delta.abs().item()), 4),
            "words": [(w, round(float(s), 4)) for (w, _, _), s in zip(words, word_scores)]}


def explain_occlusion(text, tok, model, target=None):
    """Leave-one-word-out: score = drop in the target probability when the word is removed."""
    names = label_names(model)
    words = text.split()
    variants = [text] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = predict_proba(variants, tok, model, batch_size=64)
    if target is None:
        target = int(probs[0].argmax())
    elif isinstance(target, str):
        target = names.index(target)
    drops = probs[0, target] - probs[1:, target]
    top = np.abs(drops).max() if len(drops) else 0
    if top > 0:
        drops = drops / top
    return {"label": names[target], "confidence": float(probs[0, target]),
            "words": [(w, round(float(s), 4)) for w, s in zip(words, drops)]}


def to_html(explanation):
    """Render word scores as coloured text: green = supports the label, red = against it."""
    parts = []
    for word, s in explanation["words"]:
        colour = "0,150,70" if s >= 0 else "210,40,40"
        alpha = min(abs(s), 1.0) * 0.6
        parts.append(f'<span title="{s:+.2f}" style="background:rgba({colour},{alpha:.2f});'
                     f'padding:1px 2px;border-radius:3px">{html_lib.escape(word)}</span>')
    head = f'<b>{explanation["label"]}</b> ({explanation["confidence"]:.0%}) &nbsp; '
    return head + " ".join(parts)
