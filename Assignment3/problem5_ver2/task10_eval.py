#!/usr/bin/env python3
"""
Task 10 Evaluation: Arabic Keyphrase Extraction
===============================================

Evaluates all 5 models on Keyphrase Extraction using:
  • Precision@3 and Precision@5
  • Overall + per Arabic-variety breakdown
  • Comparison tables, rankings, charts, and CSV exports

Usage: python task10_eval.py
Requires: gold_final.json (or golden_final.json) and final_5_models.json in the same directory.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130

BASE_DIR = Path.cwd()
MODEL_ORDER = ["chatgpt", "gemini", "allam", "jais", "fanar"]
MODEL_LABELS = {
    "chatgpt": "ChatGPT",
    "gemini": "Gemini",
    "allam": "ALLaM",
    "jais": "Jais",
    "fanar": "Fanar",
}

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────

def resolve_existing_file(candidates: list[str]) -> Path:
    for candidate in candidates:
        path = BASE_DIR / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of these files: {candidates}")

print("=" * 70)
print("Task 10 Evaluation: Arabic Keyphrase Extraction")
print("=" * 70)

GOLD_PATH = resolve_existing_file(["gold_final.json", "golden_final.json"])
OUTPUTS_PATH = resolve_existing_file(["final_5_models.json"])

with GOLD_PATH.open(encoding="utf-8") as f:
    gold_data = json.load(f)

with OUTPUTS_PATH.open(encoding="utf-8") as f:
    outputs_data = json.load(f)

MODEL_KEYS = [key for key in MODEL_ORDER if key in outputs_data.get("models", {})]
if len(MODEL_KEYS) != 5:
    raise ValueError(f"Expected 5 models, found {MODEL_KEYS}")

print(f"\nLoaded gold file   : {GOLD_PATH.name}")
print(f"Loaded outputs file: {OUTPUTS_PATH.name}")
print(f"Gold items         : {len(gold_data)}")
print(f"Models             : {MODEL_KEYS}")
print(f"Task field         : {outputs_data.get('task', 'not provided')}")

# ─────────────────────────────────────────────────────────────────
# NORMALIZATION & PARSING
# ─────────────────────────────────────────────────────────────────

ARABIC_PUNCT_RE = re.compile(r'[،,\.؟\?!\"\'()\[\]{}\-–—/\\:;]+')
TATWEEL_RE = re.compile(r'ـ+')

def normalize_arabic(text: object) -> str:
    """Normalize Arabic text for comparison."""
    if text is None:
        return ""
    text = str(text).strip()
    text = TATWEEL_RE.sub("", text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = ARABIC_PUNCT_RE.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def _string_to_list(value: str) -> list[str]:
    """Convert various string formats to list."""
    value = re.sub(r'```(?:json)?|```', '', value).strip()
    if not value:
        return []

    candidate_strings = [value]
    match = re.search(r'\[[\s\S]*\]', value)
    if match and match.group(0) not in candidate_strings:
        candidate_strings.insert(0, match.group(0))

    for candidate in candidate_strings:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(candidate)
            except Exception:
                continue
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            if isinstance(parsed, str) and parsed.strip() and parsed.strip() != candidate:
                nested = _string_to_list(parsed)
                if nested:
                    return nested

    numbered = [match.group(1).strip() for match in re.finditer(r'^\s*\d+[\.\)]\s*(.+)$', value, flags=re.MULTILINE)]
    if numbered:
        return numbered

    parts = [part.strip().strip('\'"- ') for part in re.split(r'[,،\n]+', value) if part.strip()]
    return parts

def parse_output(raw: object) -> list[str]:
    """Parse model output in various formats."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, dict):
        for key in ('output', 'text', 'response', 'content', 'result'):
            if key in raw:
                return parse_output(raw[key])
        return []
    return _string_to_list(str(raw))

def precision_at_k(predicted: list[str], gold: list[str], k: int) -> float:
    """Compute Precision@k metric."""
    if k <= 0 or not gold:
        return 0.0
    pred_norm = [normalize_arabic(item) for item in predicted[:k] if normalize_arabic(item)]
    gold_norm = {normalize_arabic(item) for item in gold if normalize_arabic(item)}
    if not pred_norm or not gold_norm:
        return 0.0
    hits = sum(1 for item in pred_norm if item in gold_norm)
    return hits / k

print("\n✓ Normalization and parsing functions ready")

# ─────────────────────────────────────────────────────────────────
# BUILD EVALUATION DATAFRAME
# ─────────────────────────────────────────────────────────────────

gold_lookup = {}
for item in gold_data:
    gold_lookup[item['id']] = {
        'variety': item['variety'],
        'topic': item['topic'],
        'paragraph': item['paragraph'],
        'gold_keyphrases': item['gold_keyphrases'],
    }

rows = []
for model_key in MODEL_KEYS:
    raw_lookup = {entry['id']: entry.get('output', []) for entry in outputs_data['models'][model_key]['outputs']}
    for item_id, meta in gold_lookup.items():
        predicted = parse_output(raw_lookup.get(item_id, []))
        gold = meta['gold_keyphrases']
        rows.append({
            'model': model_key,
            'model_name': MODEL_LABELS.get(model_key, model_key),
            'id': item_id,
            'variety': meta['variety'],
            'topic': meta['topic'],
            'p3': precision_at_k(predicted, gold, 3),
            'p5': precision_at_k(predicted, gold, 5),
            'n_pred': len(predicted),
            'predicted': predicted,
            'gold': gold,
            'paragraph': meta['paragraph'],
        })

df = pd.DataFrame(rows).sort_values(['model', 'id']).reset_index(drop=True)
print(f"\n✓ Evaluation complete: {len(df)} rows ({len(MODEL_KEYS)} models × {len(gold_lookup)} items)")

# ─────────────────────────────────────────────────────────────────
# OVERALL COMPARISON
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("OVERALL RESULTS — Task 10: Keyphrase Extraction")
print("=" * 70)
print(f"{'Model':<12} | {'P@3 Mean':>9} | {'P@3 Std':>8} | {'P@5 Mean':>9} | {'P@5 Std':>8}")
print("-" * 70)

overall_rows = []
for m in MODEL_KEYS:
    sub = df[df['model'] == m]
    row = {
        'Model': MODEL_LABELS[m],
        'P@3 Mean': sub['p3'].mean(),
        'P@3 Std': sub['p3'].std(),
        'P@5 Mean': sub['p5'].mean(),
        'P@5 Std': sub['p5'].std(),
    }
    overall_rows.append(row)
    print(f"{row['Model']:<12} | {row['P@3 Mean']:>9.4f} | {row['P@3 Std']:>8.4f} | {row['P@5 Mean']:>9.4f} | {row['P@5 Std']:>8.4f}")

print("=" * 70)
overall_df = pd.DataFrame(overall_rows)

# ─────────────────────────────────────────────────────────────────
# VARIETY BREAKDOWN
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("P@5 BY ARABIC VARIETY")
print("=" * 80)
print(f"{'Model':<12} | {'MSA P@5':>9} | {'Classical P@5':>14} | {'Dialect P@5':>12}")
print("-" * 80)

varieties = ['MSA', 'Classical', 'Dialect']
variety_rows = []
for m in MODEL_KEYS:
    sub = df[df['model'] == m]
    row = {'Model': MODEL_LABELS[m]}
    for v in varieties:
        vsub = sub[sub['variety'] == v]
        row[v] = vsub['p5'].mean() if len(vsub) > 0 else float('nan')
    variety_rows.append(row)
    print(f"{row['Model']:<12} | {row['MSA']:>9.4f} | {row['Classical']:>14.4f} | {row['Dialect']:>12.4f}")

print("=" * 80)
variety_df = pd.DataFrame(variety_rows).set_index('Model')

# ─────────────────────────────────────────────────────────────────
# RANKINGS
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 85)
print("FINAL RANKING — Task 10: Keyphrase Extraction")
print("=" * 85)
print(f"{'Rank':<5} | {'Model':<12} | {'P@3 Overall':>11} | {'P@5 Overall':>11} | {'P@5 MSA':>9} | {'P@5 Classical':>14} | {'P@5 Dialect':>12}")
print("-" * 85)

rank_data = []
for m in MODEL_KEYS:
    sub = df[df['model'] == m]
    rank_data.append({
        'model': m,
        'label': MODEL_LABELS[m],
        'p3': sub['p3'].mean(),
        'p5': sub['p5'].mean(),
        'msa': sub[sub['variety'] == 'MSA']['p5'].mean(),
        'classical': sub[sub['variety'] == 'Classical']['p5'].mean(),
        'dialect': sub[sub['variety'] == 'Dialect']['p5'].mean(),
    })

rank_data.sort(key=lambda x: x['p5'], reverse=True)
for rank, r in enumerate(rank_data, 1):
    print(f"{rank:<5} | {r['label']:<12} | {r['p3']:>11.4f} | {r['p5']:>11.4f} | {r['msa']:>9.4f} | {r['classical']:>14.4f} | {r['dialect']:>12.4f}")

print("=" * 85)

ranking_df = pd.DataFrame(rank_data)

# ─────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────

print("\nGenerating visualizations...")

colors = {
    'chatgpt': '#1f77b4',
    'gemini': '#2ca02c',
    'allam': '#ff7f0e',
    'jais': '#9467bd',
    'fanar': '#d62728',
}

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Chart 1: Overall P@3 vs P@5
bar_positions = np.arange(len(MODEL_KEYS))
width = 0.35
p3_means = [df[df['model'] == model_key]['p3'].mean() for model_key in MODEL_KEYS]
p5_means = [df[df['model'] == model_key]['p5'].mean() for model_key in MODEL_KEYS]
p3_stds = [df[df['model'] == model_key]['p3'].std() for model_key in MODEL_KEYS]
p5_stds = [df[df['model'] == model_key]['p5'].std() for model_key in MODEL_KEYS]

ax1.bar(bar_positions - width / 2, p3_means, width, yerr=p3_stds, capsize=4, 
        color=[colors[m] for m in MODEL_KEYS], label='P@3', alpha=0.75)
ax1.bar(bar_positions + width / 2, p5_means, width, yerr=p5_stds, capsize=4, 
        color=[colors[m] for m in MODEL_KEYS], alpha=1.0, label='P@5', hatch='//')
ax1.set_xticks(bar_positions)
ax1.set_xticklabels([MODEL_LABELS[m] for m in MODEL_KEYS])
ax1.set_ylim(0, 1)
ax1.set_title('Overall P@3 vs P@5 per Model', fontweight='bold')
ax1.set_ylabel('Score')
ax1.legend()
ax1.grid(alpha=0.3, axis='y')

# Chart 2: P@5 by variety
var_x = np.arange(len(varieties))
var_width = 0.15
for idx, model_key in enumerate(MODEL_KEYS):
    values = [df[(df['model'] == model_key) & (df['variety'] == variety)]['p5'].mean() for variety in varieties]
    ax2.bar(var_x + idx * var_width, values, var_width, color=colors[model_key], label=MODEL_LABELS[model_key])
ax2.set_xticks(var_x + var_width * 2)
ax2.set_xticklabels(varieties)
ax2.set_ylim(0, 1)
ax2.set_title('P@5 by Arabic Variety', fontweight='bold')
ax2.set_ylabel('P@5')
ax2.legend(fontsize=8, ncol=2)
ax2.grid(alpha=0.3, axis='y')

# Chart 3: Box plot
heatmap_matrix = np.array([df[df['model'] == model_key].sort_values('id')['p5'].to_list() for model_key in MODEL_KEYS])
sns.heatmap(heatmap_matrix, ax=ax3, cmap='RdYlGn', vmin=0, vmax=1, cbar=True, 
            xticklabels=8, yticklabels=[MODEL_LABELS[m] for m in MODEL_KEYS])
ax3.set_title('P@5 Heatmap: All Models × All Items', fontweight='bold')
ax3.set_xlabel('Item Index')
ax3.set_ylabel('Model')

# Chart 4: Box plot distribution
sns.boxplot(data=df, x='model', y='p5', order=MODEL_KEYS, 
            palette=[colors[m] for m in MODEL_KEYS], ax=ax4)
ax4.set_xticklabels([MODEL_LABELS[m] for m in MODEL_KEYS])
ax4.set_ylim(0, 1)
ax4.set_title('P@5 Distribution per Model (Box Plot)', fontweight='bold')
ax4.set_xlabel('Model')
ax4.set_ylabel('P@5')
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
charts_path = BASE_DIR / 'task10_evaluation_charts.png'
plt.savefig(charts_path, bbox_inches='tight', dpi=150)
print(f"✓ Saved chart file: {charts_path.name}")

# ─────────────────────────────────────────────────────────────────
# EXPORT RESULTS
# ─────────────────────────────────────────────────────────────────

print("\nExporting results...")

full_results = df[['model_name', 'id', 'variety', 'topic', 'p3', 'p5', 'n_pred', 'predicted', 'gold']].copy()
full_results['predicted'] = full_results['predicted'].apply(lambda value: ' | '.join(value) if isinstance(value, list) else str(value))
full_results['gold'] = full_results['gold'].apply(lambda value: ' | '.join(value) if isinstance(value, list) else str(value))

overall_csv = BASE_DIR / 'task10_overall_comparison.csv'
ranking_csv = BASE_DIR / 'task10_ranking_table.csv'
variety_csv = BASE_DIR / 'task10_variety_comparison.csv'
full_csv = BASE_DIR / 'task10_full_results.csv'

overall_df.to_csv(overall_csv, index=False, encoding='utf-8-sig')
ranking_df.to_csv(ranking_csv, index=False, encoding='utf-8-sig')
variety_df.to_csv(variety_csv, encoding='utf-8-sig')
full_results.to_csv(full_csv, index=False, encoding='utf-8-sig')

print(f"✓ Saved CSV files:")
for path in [overall_csv, ranking_csv, variety_csv, full_csv]:
    print(f"  - {path.name}")

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────

best_model_label = rank_data[0]['label']
best_model_p5 = rank_data[0]['p5']
best_dialect_model = variety_df['Dialect'].idxmax()
best_dialect_p5 = variety_df.loc[best_dialect_model, 'Dialect']
hardest_variety = variety_df.mean(axis=0).sort_values().index[0]
hardest_p5 = variety_df.mean(axis=0).sort_values().iloc[0]
avg_p3 = df['p3'].mean()
avg_p5 = df['p5'].mean()

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"🏆 Best model overall   : {best_model_label} (P@5={best_model_p5:.4f})")
print(f"📊 Best dialect model   : {best_dialect_model} (P@5={best_dialect_p5:.4f})")
print(f"📉 Hardest variety      : {hardest_variety} (mean P@5={hardest_p5:.4f})")
print(f"📈 Average P@3          : {avg_p3:.4f}")
print(f"📈 Average P@5          : {avg_p5:.4f}")
print("=" * 70)

print("\n✓ Evaluation complete!")
