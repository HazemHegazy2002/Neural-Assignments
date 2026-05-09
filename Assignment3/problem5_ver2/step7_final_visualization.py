#!/usr/bin/env python3
"""
Step 7: Final Visualization & Report Figures for Task 10
=========================================================

Generates publication-quality visualization figures for the final report on
Arabic Keyphrase Extraction model evaluation.

Produces 5 publication-ready figures:
  A) Overall F1-score comparison (bar chart)
  B) Precision vs Recall scatter plot
  C) Recall by Arabic variety (grouped bar chart)
  D) Hallucination comparison (false positives)
  E) Error distribution heatmap (models × varieties)

All figures saved to: visualizations/ directory

Usage: python step7_final_visualization.py
Requires: final_5_models.json, error_cases.json in the same directory.

Dependencies: matplotlib, seaborn, pandas, numpy, json, pathlib
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ─────────────────────────────────────────────────────────────────
# SETUP & CONFIG
# ─────────────────────────────────────────────────────────────────

BASE_DIR = Path.cwd()
VIZ_DIR = BASE_DIR / 'visualizations'
VIZ_DIR.mkdir(exist_ok=True)

MODEL_ORDER = ["chatgpt", "gemini", "allam", "jais", "fanar"]
MODEL_LABELS = {
    "chatgpt": "ChatGPT",
    "gemini": "Gemini",
    "allam": "ALLaM",
    "jais": "Jais",
    "fanar": "Fanar",
}

# Professional color palette
COLOR_PALETTE = {
    "chatgpt": "#1f77b4",    # Blue
    "gemini": "#2ca02c",     # Green
    "allam": "#ff7f0e",      # Orange
    "jais": "#9467bd",       # Purple
    "fanar": "#d62728",      # Red
}

# Setup matplotlib style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

print("=" * 80)
print("Step 7: Final Visualization & Report Figures")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────

def resolve_existing_file(candidates: list[str]) -> Path:
    for candidate in candidates:
        path = BASE_DIR / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of these files: {candidates}")

print(f"\nLoading data files...")

OUTPUTS_PATH = resolve_existing_file(["final_5_models.json"])
ERROR_CASES_PATH = resolve_existing_file(["error_cases.json"])
DATASET_PATH = resolve_existing_file(["dataset.json"])

with OUTPUTS_PATH.open(encoding="utf-8") as f:
    outputs_data = json.load(f)

with ERROR_CASES_PATH.open(encoding="utf-8") as f:
    error_cases_data = json.load(f)

with DATASET_PATH.open(encoding="utf-8") as f:
    dataset = json.load(f)

print(f"✓ Loaded final_5_models.json")
print(f"✓ Loaded error_cases.json")
print(f"✓ Loaded dataset.json ({len(dataset)} items)")

# ─────────────────────────────────────────────────────────────────
# NORMALIZE ARABIC TEXT (from step 6)
# ─────────────────────────────────────────────────────────────────

ARABIC_TASHKEEL_RE = re.compile(r'[\u064B-\u065F\u0670]')
ARABIC_PUNCT_RE = re.compile(r'[،,\.؟\?!\"\'()\[\]{}\-–—/\\:;]+')
TATWEEL_RE = re.compile(r'ـ+')

def normalize_arabic_extended(text: object) -> str:
    """Comprehensive Arabic text normalization."""
    if text is None:
        return ""
    text = str(text).strip()
    text = TATWEEL_RE.sub("", text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = ARABIC_TASHKEEL_RE.sub("", text)
    text = ARABIC_PUNCT_RE.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ─────────────────────────────────────────────────────────────────
# OUTPUT PARSING (from step 5)
# ─────────────────────────────────────────────────────────────────

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
        for loader in (json.loads, __import__('ast').literal_eval):
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

# ─────────────────────────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────

def compute_errors(predicted: list[str], gold: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Compute TP, FP, FN using normalized text."""
    pred_norm = {normalize_arabic_extended(p): p for p in predicted if normalize_arabic_extended(p)}
    gold_norm = {normalize_arabic_extended(g): g for g in gold if normalize_arabic_extended(g)}
    
    tp_keys = pred_norm.keys() & gold_norm.keys()
    fp_keys = pred_norm.keys() - gold_norm.keys()
    fn_keys = gold_norm.keys() - pred_norm.keys()
    
    tp = [pred_norm[k] for k in tp_keys]
    fp = [pred_norm[k] for k in fp_keys]
    fn = [gold_norm[k] for k in fn_keys]
    
    return tp, fp, fn

def compute_metrics(tp: int, fp: int, fn: int) -> dict:
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

print(f"\nComputing metrics for all models...")

# Build evaluation data
gold_lookup = {}
for item in dataset:
    gold_lookup[item['id']] = {
        'variety': item.get('type', 'unknown'),
        'topic': item.get('topic', 'unknown'),
        'paragraph': item.get('paragraph', ''),
        'gold_keyphrases': item.get('gold_keyphrases', []),
    }

metrics_by_model = {}
metrics_by_model_variety = defaultdict(dict)

for model_key in MODEL_ORDER:
    if model_key not in outputs_data.get("models", {}):
        continue
    
    raw_lookup = {
        str(entry['id']): entry.get('output', [])
        for entry in outputs_data['models'][model_key]['outputs']
    }
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    variety_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for item_id, meta in gold_lookup.items():
        predicted = parse_output(raw_lookup.get(str(item_id), []))
        gold = meta['gold_keyphrases']
        variety = meta['variety']
        
        tp, fp, fn = compute_errors(predicted, gold)
        
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)
        
        variety_stats[variety]['tp'] += len(tp)
        variety_stats[variety]['fp'] += len(fp)
        variety_stats[variety]['fn'] += len(fn)
    
    metrics_by_model[model_key] = compute_metrics(total_tp, total_fp, total_fn)
    
    for variety in ['MSA', 'Classical', 'Dialect']:
        stats = variety_stats.get(variety, {'tp': 0, 'fp': 0, 'fn': 0})
        metrics_by_model_variety[model_key][variety] = compute_metrics(
            stats['tp'], stats['fp'], stats['fn']
        )

print(f"✓ Metrics computed for {len(metrics_by_model)} models")

# ─────────────────────────────────────────────────────────────────
# FIGURE A: OVERALL F1-SCORE COMPARISON
# ─────────────────────────────────────────────────────────────────

print(f"\nGenerating Figure A: Overall F1-score comparison...")

fig_a, ax_a = plt.subplots(figsize=(10, 6))

models = [MODEL_LABELS[m] for m in MODEL_ORDER]
f1_scores = [metrics_by_model[m]['f1'] for m in MODEL_ORDER]
colors = [COLOR_PALETTE[m] for m in MODEL_ORDER]

bars = ax_a.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels on bars
for bar, f1 in zip(bars, f1_scores):
    height = bar.get_height()
    ax_a.text(bar.get_x() + bar.get_width() / 2., height,
              f'{f1:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax_a.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
ax_a.set_title('Overall F1-Score Comparison\nTask 10: Arabic Keyphrase Extraction', 
               fontweight='bold', fontsize=14, pad=20)
ax_a.set_ylim(0, max(f1_scores) * 1.15)
ax_a.grid(axis='y', alpha=0.3, linestyle='--')
ax_a.set_axisbelow(True)

plt.tight_layout()
fig_a_path = VIZ_DIR / 'overall_f1_comparison.png'
fig_a.savefig(fig_a_path, dpi=150, bbox_inches='tight')
plt.close(fig_a)
print(f"  ✓ Saved: {fig_a_path.name}")

# ─────────────────────────────────────────────────────────────────
# FIGURE B: PRECISION VS RECALL SCATTER PLOT
# ─────────────────────────────────────────────────────────────────

print(f"Generating Figure B: Precision vs Recall scatter plot...")

fig_b, ax_b = plt.subplots(figsize=(10, 8))

precisions = [metrics_by_model[m]['precision'] for m in MODEL_ORDER]
recalls = [metrics_by_model[m]['recall'] for m in MODEL_ORDER]

for i, model_key in enumerate(MODEL_ORDER):
    ax_b.scatter(precisions[i], recalls[i], s=400, 
                color=COLOR_PALETTE[model_key], edgecolor='black', linewidth=2,
                alpha=0.8, zorder=3, label=MODEL_LABELS[model_key])
    
    # Annotate model names
    ax_b.annotate(MODEL_LABELS[model_key], 
                 (precisions[i], recalls[i]),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1))

ax_b.set_xlabel('Precision', fontweight='bold', fontsize=12)
ax_b.set_ylabel('Recall', fontweight='bold', fontsize=12)
ax_b.set_title('Precision vs Recall Analysis\nTask 10: Arabic Keyphrase Extraction',
              fontweight='bold', fontsize=14, pad=20)
ax_b.set_xlim(-0.05, 1.05)
ax_b.set_ylim(-0.05, 1.05)
ax_b.grid(True, alpha=0.3, linestyle='--')
ax_b.set_axisbelow(True)

# Add diagonal F1 contours
for f1_val in [0.1, 0.3, 0.5, 0.7]:
    x_vals = np.linspace(0, 1, 100)
    y_vals = f1_val * x_vals / (2 * x_vals - f1_val)
    y_vals = np.clip(y_vals, 0, 1)
    ax_b.plot(x_vals, y_vals, 'k--', alpha=0.1, linewidth=0.5)
    
    # Label the contours
    idx = np.argmax(y_vals <= 1)
    if idx > 0:
        ax_b.text(x_vals[idx], y_vals[idx], f'F1={f1_val}', fontsize=8, alpha=0.4)

plt.tight_layout()
fig_b_path = VIZ_DIR / 'precision_recall_scatter.png'
fig_b.savefig(fig_b_path, dpi=150, bbox_inches='tight')
plt.close(fig_b)
print(f"  ✓ Saved: {fig_b_path.name}")

# ─────────────────────────────────────────────────────────────────
# FIGURE C: RECALL BY ARABIC VARIETY (GROUPED BAR CHART)
# ─────────────────────────────────────────────────────────────────

print(f"Generating Figure C: Recall by Arabic variety...")

fig_c, ax_c = plt.subplots(figsize=(12, 7))

varieties = ['MSA', 'Classical', 'Dialect']
x = np.arange(len(varieties))
width = 0.15

for idx, model_key in enumerate(MODEL_ORDER):
    recalls_by_variety = [
        metrics_by_model_variety[model_key][variety]['recall']
        for variety in varieties
    ]
    ax_c.bar(x + idx * width, recalls_by_variety, width, 
            label=MODEL_LABELS[model_key], color=COLOR_PALETTE[model_key],
            edgecolor='black', linewidth=1, alpha=0.85)

ax_c.set_xlabel('Arabic Variety', fontweight='bold', fontsize=12)
ax_c.set_ylabel('Recall', fontweight='bold', fontsize=12)
ax_c.set_title('Recall by Arabic Variety\nTask 10: Arabic Keyphrase Extraction',
              fontweight='bold', fontsize=14, pad=20)
ax_c.set_xticks(x + width * 2)
ax_c.set_xticklabels(varieties)
ax_c.set_ylim(0, 1.0)
ax_c.legend(loc='upper right', framealpha=0.95, edgecolor='black')
ax_c.grid(axis='y', alpha=0.3, linestyle='--')
ax_c.set_axisbelow(True)

plt.tight_layout()
fig_c_path = VIZ_DIR / 'recall_by_variety.png'
fig_c.savefig(fig_c_path, dpi=150, bbox_inches='tight')
plt.close(fig_c)
print(f"  ✓ Saved: {fig_c_path.name}")

# ─────────────────────────────────────────────────────────────────
# FIGURE D: HALLUCINATION COMPARISON (FALSE POSITIVES)
# ─────────────────────────────────────────────────────────────────

print(f"Generating Figure D: Hallucination comparison...")

fig_d, ax_d = plt.subplots(figsize=(10, 6))

fp_counts = [metrics_by_model[m]['fp'] for m in MODEL_ORDER]
models_d = [MODEL_LABELS[m] for m in MODEL_ORDER]
colors_d = [COLOR_PALETTE[m] for m in MODEL_ORDER]

bars_d = ax_d.bar(models_d, fp_counts, color=colors_d, edgecolor='black', 
                  linewidth=1.5, alpha=0.85)

# Add value labels on bars
for bar, fp_count in zip(bars_d, fp_counts):
    height = bar.get_height()
    ax_d.text(bar.get_x() + bar.get_width() / 2., height,
             f'{int(fp_count)}', ha='center', va='bottom', 
             fontweight='bold', fontsize=11)

ax_d.set_ylabel('Number of False Positives (Hallucinations)', fontweight='bold', fontsize=12)
ax_d.set_title('Hallucination Comparison: False Positives\nTask 10: Arabic Keyphrase Extraction',
              fontweight='bold', fontsize=14, pad=20)
ax_d.set_ylim(0, max(fp_counts) * 1.15)
ax_d.grid(axis='y', alpha=0.3, linestyle='--')
ax_d.set_axisbelow(True)

plt.tight_layout()
fig_d_path = VIZ_DIR / 'hallucination_comparison.png'
fig_d.savefig(fig_d_path, dpi=150, bbox_inches='tight')
plt.close(fig_d)
print(f"  ✓ Saved: {fig_d_path.name}")

# ─────────────────────────────────────────────────────────────────
# FIGURE E: ERROR DISTRIBUTION HEATMAP
# ─────────────────────────────────────────────────────────────────

print(f"Generating Figure E: Error distribution heatmap...")

fig_e, ax_e = plt.subplots(figsize=(10, 6))

# Prepare heatmap data (FN / Total by model and variety)
heatmap_data = []
for model_key in MODEL_ORDER:
    row = []
    for variety in ['MSA', 'Classical', 'Dialect']:
        metrics = metrics_by_model_variety[model_key][variety]
        total = metrics['tp'] + metrics['fp'] + metrics['fn']
        error_rate = 1 - metrics['recall'] if total > 0 else 0  # Error rate as 1 - recall
        row.append(error_rate)
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(
    heatmap_data,
    index=[MODEL_LABELS[m] for m in MODEL_ORDER],
    columns=['MSA', 'Classical', 'Dialect']
)

sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', 
           cbar_kws={'label': 'Error Rate (1 - Recall)'}, ax=ax_e,
           linewidths=1.5, linecolor='black', square=True, 
           vmin=0, vmax=1, annot_kws={'fontsize': 11, 'weight': 'bold'})

ax_e.set_title('Error Distribution: Models × Arabic Varieties\n(Higher = More Errors)',
              fontweight='bold', fontsize=14, pad=20)
ax_e.set_xlabel('Arabic Variety', fontweight='bold', fontsize=12)
ax_e.set_ylabel('Model', fontweight='bold', fontsize=12)

plt.tight_layout()
fig_e_path = VIZ_DIR / 'error_heatmap.png'
fig_e.savefig(fig_e_path, dpi=150, bbox_inches='tight')
plt.close(fig_e)
print(f"  ✓ Saved: {fig_e_path.name}")

# ─────────────────────────────────────────────────────────────────
# SUMMARY TABLE & FINAL REPORT
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("METRICS SUMMARY TABLE")
print("=" * 80)

print(f"\n{'Model':<12} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'TP':>5} | {'FP':>5} | {'FN':>5}")
print(f"{'-' * 90}")

for model_key in MODEL_ORDER:
    m = metrics_by_model[model_key]
    print(f"{MODEL_LABELS[model_key]:<12} | {m['precision']:>9.4f} | {m['recall']:>9.4f} | "
          f"{m['f1']:>9.4f} | {m['tp']:>5d} | {m['fp']:>5d} | {m['fn']:>5d}")

# Variety breakdown
print(f"\n" + "=" * 80)
print("RECALL BY ARABIC VARIETY")
print("=" * 80)

print(f"\n{'Model':<12} | {'MSA':>9} | {'Classical':>12} | {'Dialect':>9}")
print(f"{'-' * 60}")

for model_key in MODEL_ORDER:
    msa = metrics_by_model_variety[model_key]['MSA']['recall']
    classical = metrics_by_model_variety[model_key]['Classical']['recall']
    dialect = metrics_by_model_variety[model_key]['Dialect']['recall']
    print(f"{MODEL_LABELS[model_key]:<12} | {msa:>9.4f} | {classical:>12.4f} | {dialect:>9.4f}")

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────

print(f"\n" + "=" * 80)
print("VISUALIZATION FILES GENERATED")
print("=" * 80)

viz_files = [
    ("A", "overall_f1_comparison.png", "Overall F1-score comparison (bar chart)"),
    ("B", "precision_recall_scatter.png", "Precision vs Recall scatter plot"),
    ("C", "recall_by_variety.png", "Recall by Arabic variety (grouped bars)"),
    ("D", "hallucination_comparison.png", "Hallucination comparison (false positives)"),
    ("E", "error_heatmap.png", "Error distribution heatmap (models × varieties)"),
]

print(f"\nLocation: {VIZ_DIR}/\n")

for fig_id, filename, description in viz_files:
    filepath = VIZ_DIR / filename
    if filepath.exists():
        size_kb = filepath.stat().st_size / 1024
        print(f"  [{fig_id}] ✓ {filename:<35} ({size_kb:>6.1f} KB) — {description}")
    else:
        print(f"  [{fig_id}] ✗ {filename:<35} — NOT FOUND")

# Model ranking
print(f"\n" + "=" * 80)
print("MODEL RANKING (by F1-Score)")
print("=" * 80)

ranked = sorted(
    [(MODEL_LABELS[m], metrics_by_model[m]['f1']) for m in MODEL_ORDER],
    key=lambda x: x[1],
    reverse=True
)

for rank, (model_name, f1_score) in enumerate(ranked, 1):
    print(f"  {rank}. {model_name:<12} — F1 = {f1_score:.4f}")

print("\n" + "=" * 80)
print("✓ Visualization complete!")
print("=" * 80)
