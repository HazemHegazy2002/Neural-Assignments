"""
Task 10 — Step 4: Evaluation  (Google Colab ready)
===================================================
Evaluates all 5 models on Keyphrase Extraction using:
  • Precision@3  and  Precision@5
  • Overall + per Arabic-variety breakdown
  • Comparison table, 4 charts, error analysis
"""

# ─────────────────────────────────────────────────────────────────
# CELL 1 ── Install & import
# ─────────────────────────────────────────────────────────────────
# Run this first then restart runtime if prompted

!pip install pandas matplotlib seaborn arabic-reshaper python-bidi -q

import json, re, unicodedata, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display
from collections import defaultdict

print("✅ Libraries ready")


# ─────────────────────────────────────────────────────────────────
# CELL 2 ── Load files
# ─────────────────────────────────────────────────────────────────
# Upload gold_final.json and final_5_models.json via:
#   from google.colab import files; files.upload()
# OR drag & drop in the left sidebar ─ then run this cell.

GOLD_PATH    = "gold_final.json"
OUTPUTS_PATH = "final_5_models.json"

with open(GOLD_PATH,    encoding="utf-8") as f:
    gold_data = json.load(f)          # list of 80 dicts

with open(OUTPUTS_PATH, encoding="utf-8") as f:
    outputs_data = json.load(f)       # {task, k, models:{...}}

# ── Build gold lookup: id → {variety, topic, gold_keyphrases}
gold_lookup = {}
for item in gold_data:
    gold_lookup[item["id"]] = {
        "variety"        : item["variety"],
        "topic"          : item["topic"],
        "paragraph"      : item["paragraph"],
        "gold_keyphrases": item["gold_keyphrases"],
    }

MODEL_KEYS   = list(outputs_data["models"].keys())
MODEL_LABELS = {
    "chatgpt": "ChatGPT",
    "gemini" : "Gemini",
    "allam"  : "ALLaM",
    "jais"   : "Jais",
    "fanar"  : "Fanar",
}
K_DEFAULT = outputs_data.get("k", 5)

print(f"✅ Gold items    : {len(gold_lookup)}")
print(f"✅ Models        : {MODEL_KEYS}")
print(f"✅ Default k     : {K_DEFAULT}")


# ─────────────────────────────────────────────────────────────────
# CELL 3 ── Arabic text normalisation
# ─────────────────────────────────────────────────────────────────
# Why normalise?
#  1. Arabic Alef has 4 forms (أ إ آ ا) — same letter, different code-points
#  2. Models sometimes add tashkeel (diacritics) that gold omits
#  3. Models add punctuation ( , . ، ) that should not affect matching
#  4. Leading/trailing whitespace causes false mismatches
#  5. Definition of Arabic diacritic: unicodedata category "Mn"

ARABIC_PUNCT = re.compile(r'[،,\.؟\?!\"\'()\[\]{}\-–—/\\:;]+')

def normalize(text: str) -> str:
    """
    Steps applied to BOTH predicted and gold keyphrases before comparison:
      1. Strip surrounding whitespace
      2. Normalise Alef variants  →  plain Alef  ا
      3. Remove Arabic diacritics (tashkeel) — category Mn
      4. Remove punctuation marks
      5. Collapse multiple spaces → single space
      6. Lowercase (helps with any Latin characters in mixed outputs)
    """
    text = text.strip()
    # Step 2 – Alef normalisation
    text = re.sub(r'[أإآٱ]', 'ا', text)
    # Step 3 – remove diacritics
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Step 4 – remove punctuation
    text = ARABIC_PUNCT.sub(' ', text)
    # Step 5 – collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Step 6 – lowercase
    return text.lower()


# Quick sanity check
tests = [("أسئلة", "اسئله"), ("المُخرِج", "المخرج"), ("كسر،", "كسر")]
for a, b in tests:
    match = (normalize(a) == normalize(b))
    print(f"  '{a}' == '{b}' → {match}")

print("\n✅ Normalisation function ready")


# ─────────────────────────────────────────────────────────────────
# CELL 4 ── Output parser
# ─────────────────────────────────────────────────────────────────
# Model outputs can be:
#   • A Python list already       ["كلمة1", ...]
#   • A JSON string               "[\"كلمة1\", ...]"
#   • A numbered list string      "1. كلمة1\n2. كلمة2"
#   • A plain comma-separated str "كلمة1، كلمة2"
#   • Malformed / too long text   (Fanar edge cases)

def parse_output(raw) -> list:
    """
    Converts any model output format into a clean Python list of strings.
    Returns [] on total failure.
    """
    if raw is None:
        return []

    # Already a list
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]

    raw = str(raw).strip()

    # Strip markdown fences
    raw = re.sub(r'```json|```', '', raw).strip()

    # Try JSON array anywhere in the string
    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return [str(x).strip() for x in result if str(x).strip()]
        except Exception:
            pass

    # Numbered list:  "1. كلمة\n2. كلمة"
    numbered = re.findall(r'^\s*\d+[\.\)]\s*(.+)', raw, re.MULTILINE)
    if numbered:
        return [x.strip() for x in numbered if x.strip()]

    # Comma / Arabic comma separated
    parts = re.split(r'[,،\n]+', raw)
    cleaned = [p.strip().strip('"\'- ') for p in parts if p.strip()]
    return cleaned if cleaned else []


# Test parser on the formats seen in our data
samples = [
    ["كلمة1", "كلمة2"],                       # already list
    '["كلمة1", "كلمة2", "كلمة3"]',             # JSON string
    "1. البرامج الجديدة\n2. التليفزيون",       # numbered
    "كسر الإيهام، نجيب محفوظ",                 # Arabic comma
]
for s in samples:
    print(f"  Input:  {str(s)[:50]}")
    print(f"  Output: {parse_output(s)}\n")

print("✅ Parser ready")


# ─────────────────────────────────────────────────────────────────
# CELL 5 ── Precision@k metric
# ─────────────────────────────────────────────────────────────────
# Definition:
#   Precision@k = (# of top-k predictions found in gold set) / k
#
# Notes:
#   • We normalise both sides before comparison
#   • "found" means the normalised predicted phrase matches
#     ANY normalised gold phrase (partial word-in-phrase matching
#     is NOT used — exact normalised match only)
#   • If model produced fewer than k phrases, we still divide by k
#     (penalises incomplete outputs)

def precision_at_k(predicted: list, gold: list, k: int) -> float:
    if not gold or not predicted:
        return 0.0
    pred_norm = [normalize(p) for p in predicted[:k]]
    gold_norm = {normalize(g) for g in gold}
    hits = sum(1 for p in pred_norm if p in gold_norm)
    return hits / k


# ─────────────────────────────────────────────────────────────────
# CELL 6 ── Run evaluation for all models at k=3 and k=5
# ─────────────────────────────────────────────────────────────────

rows = []   # one row per (model, item)

for model_key in MODEL_KEYS:
    model_outputs = outputs_data["models"][model_key]["outputs"]
    # Build lookup: id → raw output
    raw_lookup = {entry["id"]: entry.get("output", []) for entry in model_outputs}

    for item_id, meta in gold_lookup.items():
        raw      = raw_lookup.get(item_id, [])
        predicted = parse_output(raw)
        gold      = meta["gold_keyphrases"]

        p3 = precision_at_k(predicted, gold, k=3)
        p5 = precision_at_k(predicted, gold, k=5)

        rows.append({
            "model"     : model_key,
            "model_name": MODEL_LABELS.get(model_key, model_key),
            "id"        : item_id,
            "variety"   : meta["variety"],
            "topic"     : meta["topic"],
            "p3"        : p3,
            "p5"        : p5,
            "n_pred"    : len(predicted),
            "predicted" : predicted,
            "gold"      : gold,
            "paragraph" : meta["paragraph"],
        })

df = pd.DataFrame(rows)
print(f"✅ Evaluation complete — {len(df)} rows ({len(MODEL_KEYS)} models × 80 items)")
print(f"\nColumns: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────
# CELL 7 ── Overall Precision@3 and Precision@5 per model
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  OVERALL RESULTS — Task 10: Keyphrase Extraction")
print(f"  80 paragraphs × {len(MODEL_KEYS)} models")
print("="*70)
print(f"  {'Model':<12} | {'P@3 Mean':>9} | {'P@3 Std':>8} | {'P@5 Mean':>9} | {'P@5 Std':>8}")
print(f"  {'-'*60}")

overall_stats = []
for m in MODEL_KEYS:
    sub  = df[df["model"] == m]
    row  = {
        "model"    : MODEL_LABELS[m],
        "p3_mean"  : sub["p3"].mean(),
        "p3_std"   : sub["p3"].std(),
        "p5_mean"  : sub["p5"].mean(),
        "p5_std"   : sub["p5"].std(),
    }
    overall_stats.append(row)
    print(f"  {row['model']:<12} | {row['p3_mean']:>9.4f} | {row['p3_std']:>8.4f} "
          f"| {row['p5_mean']:>9.4f} | {row['p5_std']:>8.4f}")

print("="*70)
overall_df = pd.DataFrame(overall_stats).set_index("model")


# ─────────────────────────────────────────────────────────────────
# CELL 8 ── Breakdown by Arabic variety (MSA / Classical / Dialect)
# ─────────────────────────────────────────────────────────────────

VARIETIES = ["MSA", "Classical", "Dialect"]

print("\n" + "="*80)
print("  P@5 BY ARABIC VARIETY")
print("="*80)
print(f"  {'Model':<12} | {'MSA P@5':>9} | {'Classical P@5':>14} | {'Dialect P@5':>12}")
print(f"  {'-'*55}")

variety_rows = []
for m in MODEL_KEYS:
    sub = df[df["model"] == m]
    row = {"model": MODEL_LABELS[m]}
    for v in VARIETIES:
        vsub      = sub[sub["variety"] == v]
        row[v]    = vsub["p5"].mean() if len(vsub) > 0 else float("nan")
    variety_rows.append(row)
    print(f"  {row['model']:<12} | {row['MSA']:>9.4f} | {row['Classical']:>14.4f} | {row['Dialect']:>12.4f}")

variety_df = pd.DataFrame(variety_rows).set_index("model")
print("="*80)


# ─────────────────────────────────────────────────────────────────
# CELL 9 ── VISUALISATIONS (4 charts)
# ─────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "chatgpt": "#2196F3",   # blue
    "gemini" : "#4CAF50",   # green
    "allam"  : "#FF9800",   # orange
    "jais"   : "#9C27B0",   # purple
    "fanar"  : "#F44336",   # red
}
label_names  = [MODEL_LABELS[m] for m in MODEL_KEYS]
colors_list  = [MODEL_COLORS[m] for m in MODEL_KEYS]

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Task 10 — Keyphrase Extraction: Evaluation Results\n5 Models × 80 Arabic Paragraphs",
             fontsize=14, fontweight="bold")

# ── Chart 1: Overall P@3 and P@5 grouped bar ─────────────────────
x     = np.arange(len(MODEL_KEYS))
width = 0.35
p3s   = [df[df["model"]==m]["p3"].mean() for m in MODEL_KEYS]
p5s   = [df[df["model"]==m]["p5"].mean() for m in MODEL_KEYS]
p3_std= [df[df["model"]==m]["p3"].std()  for m in MODEL_KEYS]
p5_std= [df[df["model"]==m]["p5"].std()  for m in MODEL_KEYS]

b3 = axes[0,0].bar(x - width/2, p3s, width, label="P@3",
                   color=colors_list, alpha=0.75,
                   yerr=p3_std, capsize=4, edgecolor="white")
b5 = axes[0,0].bar(x + width/2, p5s, width, label="P@5",
                   color=colors_list, alpha=1.0,
                   yerr=p5_std, capsize=4, edgecolor="white", hatch="//")
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(label_names, fontsize=10)
axes[0,0].set_ylabel("Precision@k")
axes[0,0].set_ylim(0, 1)
axes[0,0].set_title("Overall P@3 vs P@5 per Model", fontweight="bold")
axes[0,0].legend(fontsize=9)
axes[0,0].grid(alpha=0.3, axis="y")
for bar, val in zip(list(b3) + list(b5), p3s + p5s):
    axes[0,0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.01,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=7)

# ── Chart 2: P@5 by Arabic variety ───────────────────────────────
xv    = np.arange(len(VARIETIES))
wv    = 0.15
for i, (m, color) in enumerate(zip(MODEL_KEYS, colors_list)):
    vals = [df[(df["model"]==m) & (df["variety"]==v)]["p5"].mean()
            for v in VARIETIES]
    axes[0,1].bar(xv + i*wv, vals, wv,
                  label=MODEL_LABELS[m], color=color, edgecolor="white")
axes[0,1].set_xticks(xv + wv*2)
axes[0,1].set_xticklabels(VARIETIES, fontsize=11)
axes[0,1].set_ylabel("P@5")
axes[0,1].set_ylim(0, 1)
axes[0,1].set_title("P@5 by Arabic Variety", fontweight="bold")
axes[0,1].legend(fontsize=8, ncol=2)
axes[0,1].grid(alpha=0.3, axis="y")

# ── Chart 3: Box plot distribution of P@5 ────────────────────────
box_data = [df[df["model"]==m]["p5"].tolist() for m in MODEL_KEYS]
bp = axes[1,0].boxplot(box_data, labels=label_names,
                        patch_artist=True, notch=False)
for patch, color in zip(bp["boxes"], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
axes[1,0].set_ylabel("P@5")
axes[1,0].set_ylim(-0.05, 1.05)
axes[1,0].set_title("P@5 Distribution per Model (Box Plot)", fontweight="bold")
axes[1,0].grid(alpha=0.3, axis="y")

# ── Chart 4: Heat-map — P@5 per item for all models ──────────────
heat = np.array([df[df["model"]==m].sort_values("id")["p5"].tolist()
                 for m in MODEL_KEYS])
im   = axes[1,1].imshow(heat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
axes[1,1].set_yticks(range(len(MODEL_KEYS)))
axes[1,1].set_yticklabels(label_names, fontsize=9)
axes[1,1].set_xlabel("Paragraph ID (1–80)", fontsize=9)
axes[1,1].set_title("P@5 Heatmap — All Models × All Paragraphs\n"
                     "Green = high | Red = low", fontweight="bold")
plt.colorbar(im, ax=axes[1,1], fraction=0.02, pad=0.02, label="P@5")

# Draw variety separators on heatmap
ids_sorted = sorted(gold_lookup.keys())
variety_seq = [gold_lookup[i]["variety"] for i in ids_sorted]
for idx in range(1, len(variety_seq)):
    if variety_seq[idx] != variety_seq[idx-1]:
        axes[1,1].axvline(x=idx - 0.5, color="black", lw=2, linestyle="--")

plt.tight_layout()
plt.savefig("task10_evaluation_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved → task10_evaluation_charts.png")


# ─────────────────────────────────────────────────────────────────
# CELL 10 ── ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────
# For each model: 3 worst cases per variety
# Then: 5 documented Arabic-specific challenges

print("="*75)
print("  ERROR ANALYSIS — 3 Worst Cases per Model × Variety")
print("="*75)

for m in MODEL_KEYS:
    print(f"\n{'━'*75}")
    print(f"  MODEL: {MODEL_LABELS[m].upper()}")
    print(f"{'━'*75}")
    sub = df[df["model"] == m]
    for v in VARIETIES:
        vsub  = sub[sub["variety"] == v].nsmallest(3, "p5")
        count = len(sub[sub["variety"] == v])
        print(f"\n  ── {v} (n={count}) ──────────────────────────────────────")
        for _, row in vsub.iterrows():
            para_snip = row["paragraph"][:110].replace("\n", " ")
            print(f"  ID={int(row['id'])} | Topic={row['topic']} | P@5={row['p5']:.3f} | P@3={row['p3']:.3f}")
            print(f"  Paragraph (snippet) : {para_snip}...")
            print(f"  Gold keyphrases     : {row['gold']}")
            print(f"  Model predicted     : {row['predicted'][:5]}")
            print()

print("="*75)
print("  DOCUMENTED ARABIC-SPECIFIC CHALLENGES")
print("="*75)

challenges = [
    ("1. Dialect Vocabulary Gap",
     "Dialect paragraphs (Egyptian, Gulf, Levantine, North African) use non-MSA words "
     "that models trained primarily on MSA fail to identify as keyphrases. "
     "For example, 'بنوخذوه' and 'ازاى' are core dialect words invisible to MSA-centric models."),

    ("2. Arabic Morphology — Root/Pattern System",
     "Arabic derives dozens of forms from one root (ك-ت-ب → كتب، كاتب، مكتوب، كتابة). "
     "A model may extract 'الكتابة' while gold uses 'كتب' — semantically equivalent, "
     "but a mismatch in exact normalised form. Normalisation helps with diacritics but "
     "cannot bridge morphological variation without stemming."),

    ("3. Attached Prepositions & Pronouns (Cliticisation)",
     "Arabic glues prepositions (بـ، لـ، كـ) and pronouns (هـ، ها) to host words. "
     "Gold may say 'حرية' while a model says 'بالحرية' — same concept, different surface. "
     "Our normalisation removes diacritics and punctuation but does not strip clitics."),

    ("4. JSON / Format Non-Compliance (Fanar & Jais)",
     "Fanar and Jais occasionally produced long explanatory text instead of a clean JSON array. "
     "The parser falls back to splitting by newlines/commas which recovers some keyphrases, "
     "but the quality is lower than models that strictly follow the output format."),

    ("5. Very Short or Ambiguous Paragraphs",
     "Several dialect paragraphs (IDs 57, 58, 60, 75) are only one sentence long "
     "with 5–10 words total. The gold keyphrases cover nearly every content word, "
     "making any 5-out-of-5 selection trivially correct — yet models that produce "
     "structured Arabic output still miss them due to dialect form mismatch."),
]

for title, desc in challenges:
    print(f"\n  {title}")
    print(f"  → {desc}")


# ─────────────────────────────────────────────────────────────────
# CELL 11 ── FINAL COMPARISON TABLE (assignment format)
# ─────────────────────────────────────────────────────────────────

print("\n" + "="*85)
print("  FINAL COMPARISON TABLE — Task 10: Keyphrase Extraction")
print("  Metric: Precision@3 and Precision@5  |  80 Arabic paragraphs")
print("="*85)
hdr = (f"  {'Model':<12} | {'P@3 Overall':>11} | {'P@5 Overall':>11} | "
       f"{'P@5 MSA':>9} | {'P@5 Classical':>14} | {'P@5 Dialect':>12} | {'Rank':>5}")
print(hdr)
print(f"  {'-'*83}")

rank_data = []
for m in MODEL_KEYS:
    sub = df[df["model"] == m]
    rank_data.append({
        "model"    : m,
        "label"    : MODEL_LABELS[m],
        "p3"       : sub["p3"].mean(),
        "p5"       : sub["p5"].mean(),
        "msa"      : sub[sub["variety"]=="MSA"]["p5"].mean(),
        "classical": sub[sub["variety"]=="Classical"]["p5"].mean(),
        "dialect"  : sub[sub["variety"]=="Dialect"]["p5"].mean(),
    })

rank_data.sort(key=lambda x: x["p5"], reverse=True)
for rank, r in enumerate(rank_data, 1):
    print(f"  {r['label']:<12} | {r['p3']:>11.4f} | {r['p5']:>11.4f} | "
          f"{r['msa']:>9.4f} | {r['classical']:>14.4f} | {r['dialect']:>12.4f} | {rank:>5}")

print(f"  {'-'*83}")
print(f"\n  🏆 Best overall (P@5) : {rank_data[0]['label']} ({rank_data[0]['p5']:.4f})")
print(f"  📉 Worst overall (P@5): {rank_data[-1]['label']} ({rank_data[-1]['p5']:.4f})")

best_msa      = max(rank_data, key=lambda x: x["msa"])
best_dialect  = max(rank_data, key=lambda x: x["dialect"])
print(f"\n  Best on MSA      : {best_msa['label']} ({best_msa['msa']:.4f})")
print(f"  Best on Dialect  : {best_dialect['label']} ({best_dialect['dialect']:.4f})")


# ─────────────────────────────────────────────────────────────────
# CELL 12 ── EXPORT
# ─────────────────────────────────────────────────────────────────

# Full per-item results
export_df = df[["model","model_name","id","variety","topic","p3","p5",
                "n_pred","predicted","gold"]].copy()
export_df["predicted"] = export_df["predicted"].apply(
    lambda x: " | ".join(x) if isinstance(x, list) else str(x))
export_df["gold"] = export_df["gold"].apply(
    lambda x: " | ".join(x) if isinstance(x, list) else str(x))
export_df.to_csv("task10_full_results.csv", index=False, encoding="utf-8-sig")
print("✅ Saved → task10_full_results.csv")

# Summary table
summary_rows = []
for r in rank_data:
    sub = df[df["model"] == r["model"]]
    summary_rows.append({
        "Model"          : r["label"],
        "P@3 Overall"    : round(r["p3"], 4),
        "P@5 Overall"    : round(r["p5"], 4),
        "P@5 MSA"        : round(r["msa"], 4),
        "P@5 Classical"  : round(r["classical"], 4),
        "P@5 Dialect"    : round(r["dialect"], 4),
        "P@5 Std"        : round(sub["p5"].std(), 4),
    })

pd.DataFrame(summary_rows).to_csv(
    "task10_summary_table.csv", index=False, encoding="utf-8-sig")
print("✅ Saved → task10_summary_table.csv")

# Download
from google.colab import files
for fname in ["task10_evaluation_charts.png",
              "task10_full_results.csv",
              "task10_summary_table.csv"]:
    if os.path.exists(fname):
        files.download(fname)
        print(f"  ⬇️  {fname}")

print("\n✅ Step 4 complete — ready for report writing.")