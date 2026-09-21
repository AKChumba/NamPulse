"""Step 3 - Language detection and code-switch flagging  (Req 6).

Input : data/interim/cleaned.csv
Output: data/processed/nampulse_clean.csv      <- FINAL dataset for the team
        data/processed/nampulse_clean.parquet  (same data, faster to load)
        reports/language_report.txt

New columns:
  language           main language code (en, af, de, ...) or "und" (undetermined)
  language_conf      0-1. English: how clearly English beats its closest rival language.
                     Other languages: detector confidence. "und": confidence of its best guess.
  is_english         True/False shortcut for teammates
  is_code_switched   CANDIDATE flag: a 2nd language covers >=3 words (needs manual review)
  languages_found    all languages spotted, e.g. "en|af"
  other_lang_words   how many words were NOT in the main language

Note: no off-the-shelf detector knows Oshiwambo. Such text will show up as
"und" or a wrong language - see README "Limitations".
"""
from pathlib import Path

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "interim" / "cleaned.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = ROOT / "reports" / "language_report.txt"

MIN_WORDS = 3          # shorter than this -> too little text to judge
MIN_CONF_OTHER = 0.80  # a non-English language needs strong evidence ...
MIN_EN_VS_RIVAL = 0.50 # ... otherwise it is English if English beats that language in a 2-way test
MIN_SEG_WORDS = 3      # a language segment needs at least this many words ...
MIN_SEG_CONF = 0.95    # ... and must win an "English vs that language" test by this much

# We use ALL languages Lingua knows (75, including Afrikaans, German, Portuguese, Shona,
# Zulu, Xhosa, Tswana, Swahili). A short list forces wrong answers: with only Southern
# African languages, Indonesian or Tagalog tweets were mislabelled as Swahili/Sotho.
LANGS = list(Language.all())
detector = LanguageDetectorBuilder.from_all_languages().build()
# One extra 2-language detector per foreign language (English vs X). Used to double-check
# segments: a 2-way choice is far more reliable on short text than a 75-way choice.
pair_detectors = {lang: LanguageDetectorBuilder.from_languages(Language.ENGLISH, lang).build()
                  for lang in LANGS if lang != Language.ENGLISH}


def code(lang: Language) -> str:
    return lang.iso_code_639_1.name.lower()


def analyse(text: str) -> dict:
    result = {"language": "und", "language_conf": 0.0,
              "is_code_switched": False, "languages_found": "", "other_lang_words": 0}
    if len(text.split()) < MIN_WORDS:
        return result

    # --- main language + confidence (whole tweet) ---
    # With 75 languages, informal English scores low (probability is spread thin), so we
    # do not use a plain confidence cutoff. Instead:
    #   1. English is the top language      -> "en"
    #   2. another language wins strongly   -> that language
    #   3. another language wins weakly     -> "en" if English beats it 1-vs-1, else "und"
    ranked_langs = detector.compute_language_confidence_values(text)
    top, second = ranked_langs[0], ranked_langs[1]
    if top.language == Language.ENGLISH:
        rival = pair_detectors[second.language]
        result["language"] = "en"
        result["language_conf"] = round(rival.compute_language_confidence(text, Language.ENGLISH), 3)
    elif top.value >= MIN_CONF_OTHER:
        result["language"] = code(top.language)
        result["language_conf"] = round(top.value, 3)
    else:
        vs_en = pair_detectors[top.language].compute_language_confidence(text, Language.ENGLISH)
        if vs_en >= MIN_EN_VS_RIVAL:
            result["language"], result["language_conf"] = "en", round(vs_en, 3)
        else:
            result["language_conf"] = round(top.value, 3)

    # --- code-switching: split into single-language segments, then keep only
    #     segments that pass the checks (the raw splitter over-splits plain English) ---
    words_by_lang = {}
    for seg in detector.detect_multiple_languages_of(text):
        if seg.word_count < MIN_SEG_WORDS:
            continue
        if seg.language != Language.ENGLISH:
            seg_text = text[seg.start_index:seg.end_index]
            score = pair_detectors[seg.language].compute_language_confidence(seg_text, seg.language)
            if score < MIN_SEG_CONF:
                continue
        lang = code(seg.language)
        words_by_lang[lang] = words_by_lang.get(lang, 0) + seg.word_count

    ranked = sorted(words_by_lang, key=words_by_lang.get, reverse=True)
    result["languages_found"] = "|".join(ranked)
    if len(ranked) >= 2:
        result["is_code_switched"] = True
        result["other_lang_words"] = sum(words_by_lang.values()) - words_by_lang[ranked[0]]
        if result["language"] == "und":          # mixed tweet: use the dominant language
            result["language"] = ranked[0]
    return result


# ---------- Sanity check on hand-written mixed-language examples ----------
print("Sanity check (hand-written mixed tweets):")
for t in ["The flight was delayed again, ek is regtig moeg vir hierdie diens.",
          "Water is weer af in Windhoek, this is unacceptable, ons betaal vir niks",
          "Die diens was baie sleg and the staff were rude to everyone",
          "Ich bin so disappointed with this service, wirklich schlecht",
          "The flight was delayed and nobody told us anything about it"]:
    r = analyse(t)
    print(f"  {r['language']:>3} switched={r['is_code_switched']!s:<5} found={r['languages_found']:<8} | {t[:60]}")
print()

df = pd.read_csv(IN, parse_dates=["created_at"])
print(f"Detecting language for {len(df):,} tweets (a few minutes for 100k) ...")
info = pd.DataFrame([analyse(t) for t in df["text_clean"]])
df = pd.concat([df, info], axis=1)
df["is_english"] = df["language"] == "en"

# Final tidy column order
cols = ["tweet_id", "created_at", "label", "label_source", "topic_group",
        "text_raw", "text_clean",
        "language", "language_conf", "is_english", "is_code_switched",
        "languages_found", "other_lang_words",
        "dup_group_size", "n_words", "text_key"]
df = df[cols]
df.to_csv(OUT_DIR / "nampulse_clean.csv", index=False)
df.to_parquet(OUT_DIR / "nampulse_clean.parquet", index=False)

# ---------- Report: how does language relate to sentiment? ----------
lines = [f"Total tweets: {len(df):,}", "", "Language counts:",
         df["language"].value_counts().to_string(), "",
         f"Code-switched tweets: {int(df['is_code_switched'].sum()):,}", "",
         "Label mix by language (row %):",
         (pd.crosstab(df["language"], df["label"], normalize="index") * 100).round(1).to_string(), "",
         "Language counts, hand-labelled tweets only:",
         df[df["label_source"] == "manual"]["language"].value_counts().to_string()]
REPORT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved final dataset -> {OUT_DIR / 'nampulse_clean.csv'}")
