#!/usr/bin/env python3
"""
Step 6: Error Analysis for Task 10 - Arabic Keyphrase Extraction
================================================================

Comprehensive error analysis of all 5 models on Arabic keyphrase extraction.

Performs:
  • Arabic text normalization (tashkeel, Alef, taa marbuta, yaa, punctuation)
  • TP/FP/FN computation per model
  • Error pattern analysis (most common missed/hallucinated keyphrases)
  • Variety-specific error breakdowns (MSA, Classical, Dialect)
  • Linguistic observations (dialect difficulty, named entities, semantic mismatches)
  • Difficult failure case extraction

Usage: python step6_error_analysis.py
Requires: dataset.json, final_5_models.json in the same directory.

Output:
  • Printed summaries for each model
  • Printed error pattern analysis
  • error_cases.json (top 20 difficult cases per model)
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

# ─────────────────────────────────────────────────────────────────
# TYPES & CONFIG
# ─────────────────────────────────────────────────────────────────

BASE_DIR = Path.cwd()
MODEL_ORDER = ["chatgpt", "gemini", "allam", "jais", "fanar"]
MODEL_LABELS = {
    "chatgpt": "ChatGPT",
    "gemini": "Gemini",
    "allam": "ALLaM",
    "jais": "Jais",
    "fanar": "Fanar",
}

class ErrorMetrics(NamedTuple):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

class ErrorCase(NamedTuple):
    item_id: str
    variety: str
    topic: str
    gold: list[str]
    predicted: list[str]
    tp: list[str]
    fp: list[str]
    fn: list[str]
    error_severity: float

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────

def resolve_existing_file(candidates: list[str]) -> Path:
    for candidate in candidates:
        path = BASE_DIR / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of these files: {candidates}")

print("=" * 80)
print("Step 6: Error Analysis - Task 10: Arabic Keyphrase Extraction")
print("=" * 80)

DATASET_PATH = resolve_existing_file(["dataset.json"])
OUTPUTS_PATH = resolve_existing_file(["final_5_models.json"])

with DATASET_PATH.open(encoding="utf-8") as f:
    dataset = json.load(f)

with OUTPUTS_PATH.open(encoding="utf-8") as f:
    outputs_data = json.load(f)

MODEL_KEYS = [key for key in MODEL_ORDER if key in outputs_data.get("models", {})]

print(f"\nLoaded dataset file : {DATASET_PATH.name} ({len(dataset)} items)")
print(f"Loaded outputs file : {OUTPUTS_PATH.name}")
print(f"Models to analyze   : {MODEL_KEYS}")

# ─────────────────────────────────────────────────────────────────
# ARABIC NORMALIZATION (Extended)
# ─────────────────────────────────────────────────────────────────

ARABIC_TASHKEEL_RE = re.compile(r'[\u064B-\u065F\u0670]')
ARABIC_PUNCT_RE = re.compile(r'[،,\.؟\?!\"\'()\[\]{}\-–—/\\:;]+')
TATWEEL_RE = re.compile(r'ـ+')

def normalize_arabic_extended(text: object) -> str:
    """
    Comprehensive Arabic text normalization for error analysis.
    
    Handles:
      • Tashkeel (diacritics): removes all diacritical marks
      • Alef forms: normalizes أ, إ, آ, ٱ → ا
      • Taa Marbuta: normalizes ة → ه for consistency
      • Yaa/Alef Maqsura: normalizes ى → ي
      • Tatweel: removes ـ
      • Punctuation: removes Arabic and Latin punctuation
      • Whitespace: collapses multiple spaces
    """
    if text is None:
        return ""
    
    text = str(text).strip()
    
    # Remove Tatweel (elongation mark)
    text = TATWEEL_RE.sub("", text)
    
    # Normalize Alef forms: أ إ آ ٱ → ا
    text = re.sub(r'[أإآٱ]', 'ا', text)
    
    # Normalize Taa Marbuta: ة → ه
    text = re.sub(r'ة', 'ه', text)
    
    # Normalize Yaa and Alef Maqsura: ى → ي
    text = re.sub(r'ى', 'ي', text)
    
    # Remove all tashkeel (diacritical marks)
    text = ARABIC_TASHKEEL_RE.sub("", text)
    
    # Remove punctuation
    text = ARABIC_PUNCT_RE.sub(' ', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

# ─────────────────────────────────────────────────────────────────
# OUTPUT PARSING (from Step 5)
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

print("\n✓ Normalization and parsing functions ready")

# ─────────────────────────────────────────────────────────────────
# ERROR COMPUTATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def compute_errors(predicted: list[str], gold: list[str]) -> tuple[list[str], list[str], list[str]]:
    """
    Compute TP, FP, FN using normalized text.
    
    Returns:
        (true_positives, false_positives, false_negatives)
    """
    pred_norm = {normalize_arabic_extended(p): p for p in predicted if normalize_arabic_extended(p)}
    gold_norm = {normalize_arabic_extended(g): g for g in gold if normalize_arabic_extended(g)}
    
    tp_keys = pred_norm.keys() & gold_norm.keys()
    fp_keys = pred_norm.keys() - gold_norm.keys()
    fn_keys = gold_norm.keys() - pred_norm.keys()
    
    tp = [pred_norm[k] for k in tp_keys]
    fp = [pred_norm[k] for k in fp_keys]
    fn = [gold_norm[k] for k in fn_keys]
    
    return tp, fp, fn

def compute_metrics(tp: int, fp: int, fn: int) -> ErrorMetrics:
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return ErrorMetrics(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)

def categorize_error_type(gold_kw: str, predicted_kw: str) -> str:
    """Categorize the type of error."""
    gold_norm = normalize_arabic_extended(gold_kw)
    pred_norm = normalize_arabic_extended(predicted_kw)
    
    if len(pred_norm) > len(gold_norm) * 1.3:
        return "over_generation"
    elif len(pred_norm) < len(gold_norm) * 0.7:
        return "truncation"
    elif len(pred_norm.split()) > 1 and len(gold_norm.split()) == 1:
        return "phrase_hallucination"
    else:
        return "lexical_mismatch"

def is_named_entity(text: str) -> bool:
    """
    Heuristic detection for named entities.
    Arabic NEs often have specific patterns (proper nouns, etc.).
    """
    # Starts with uppercase-like characters (in Arabic context)
    # Very simple heuristic - proper nouns tend to be specific patterns
    return len(text) > 1 and text[0].isupper()

def is_generic_keyword(text: str) -> bool:
    """Detect generic/common keywords."""
    generic_patterns = [
        r'(الرسالة|الدراسة|البحث|المقال)',  # thesis, study, research
        r'(يناقش|يتناول|يركز)',  # discusses, addresses, focuses
        r'(إلى|من|في|عن|على)',  # prepositions
    ]
    text_norm = normalize_arabic_extended(text)
    return any(re.search(pattern, text_norm) for pattern in generic_patterns)

# ─────────────────────────────────────────────────────────────────
# BUILD ANALYSIS DATASET
# ─────────────────────────────────────────────────────────────────

# Create gold keyphrase lookup
gold_lookup = {}
for item in dataset:
    gold_lookup[item['id']] = {
        'variety': item.get('type', 'unknown'),  # 'type' contains MSA, Classical, Dialect
        'topic': item.get('topic', 'unknown'),
        'paragraph': item.get('paragraph', ''),
        'gold_keyphrases': item.get('gold_keyphrases', []),
    }

# Analyze each model
model_errors = {}

for model_key in MODEL_KEYS:
    print(f"\n--- Analyzing {MODEL_LABELS[model_key]} ---")
    
    raw_lookup = {
        str(entry['id']): entry.get('output', [])  # Convert id to string for matching
        for entry in outputs_data['models'][model_key]['outputs']
    }
    
    # Aggregate statistics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    variety_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    missed_keywords = Counter()  # FN
    hallucinated_keywords = Counter()  # FP
    error_cases = []
    error_type_dist = Counter()
    
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
        
        # Track error patterns
        for missed in fn:
            missed_keywords[normalize_arabic_extended(missed)] += 1
        
        for hallucinated in fp:
            hallucinated_keywords[normalize_arabic_extended(hallucinated)] += 1
            error_type_dist[categorize_error_type(hallucinated, hallucinated)] += 1
        
        # Compute error severity (inverse of recall for this case)
        error_severity = len(fn) / (len(gold) + 1e-6) if len(gold) > 0 else 0
        
        # Store difficult cases (where recall is low and FN is high)
        if error_severity > 0.3 or len(fp) > len(tp):
            error_cases.append(ErrorCase(
                item_id=item_id,
                variety=variety,
                topic=meta['topic'],
                gold=gold,
                predicted=predicted,
                tp=tp,
                fp=fp,
                fn=fn,
                error_severity=error_severity,
            ))
    
    # Sort error cases by severity
    error_cases.sort(key=lambda x: x.error_severity, reverse=True)
    
    model_errors[model_key] = {
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'variety_stats': dict(variety_stats),
        'missed_keywords': missed_keywords,
        'hallucinated_keywords': hallucinated_keywords,
        'error_type_dist': error_type_dist,
        'error_cases': error_cases[:20],  # Top 20 difficult cases
    }

print("\n✓ Error analysis complete for all models")

# ─────────────────────────────────────────────────────────────────
# PRINT DETAILED REPORTS
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("DETAILED ERROR ANALYSIS BY MODEL")
print("=" * 80)

for model_key in MODEL_KEYS:
    errors = model_errors[model_key]
    metrics = compute_metrics(errors['total_tp'], errors['total_fp'], errors['total_fn'])
    
    print(f"\n{'─' * 80}")
    print(f"MODEL: {MODEL_LABELS[model_key]}")
    print(f"{'─' * 80}")
    
    # Overall metrics
    print(f"\n📊 OVERALL METRICS:")
    print(f"  True Positives (TP)    : {metrics.tp:>5d}")
    print(f"  False Positives (FP)   : {metrics.fp:>5d}")
    print(f"  False Negatives (FN)   : {metrics.fn:>5d}")
    print(f"  Precision              : {metrics.precision:>7.4f}")
    print(f"  Recall                 : {metrics.recall:>7.4f}")
    print(f"  F1-Score               : {metrics.f1:>7.4f}")
    
    # Variety breakdown
    print(f"\n🌍 ERROR BREAKDOWN BY ARABIC VARIETY:")
    print(f"  {'Variety':<15} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'Precision':>9} | {'Recall':>9}")
    print(f"  {'-' * 70}")
    
    for variety in ['MSA', 'Classical', 'Dialect']:
        stats = errors['variety_stats'].get(variety, {'tp': 0, 'fp': 0, 'fn': 0})
        vm = compute_metrics(stats['tp'], stats['fp'], stats['fn'])
        print(f"  {variety:<15} | {vm.tp:>5} | {vm.fp:>5} | {vm.fn:>5} | {vm.precision:>9.4f} | {vm.recall:>9.4f}")
    
    # Top missed keywords
    print(f"\n❌ TOP 10 MISSED KEYWORDS (False Negatives):")
    for idx, (keyword, count) in enumerate(errors['missed_keywords'].most_common(10), 1):
        print(f"  {idx:>2d}. {keyword:<40} (missed {count:>3d} times)")
    
    # Top hallucinated keywords
    print(f"\n🎭 TOP 10 HALLUCINATED KEYWORDS (False Positives):")
    for idx, (keyword, count) in enumerate(errors['hallucinated_keywords'].most_common(10), 1):
        print(f"  {idx:>2d}. {keyword:<40} (generated {count:>3d} times)")
    
    # Error type distribution
    if errors['error_type_dist']:
        print(f"\n🔍 ERROR TYPE DISTRIBUTION:")
        for error_type, count in errors['error_type_dist'].most_common():
            print(f"  {error_type:<20}: {count:>4d}")

# ─────────────────────────────────────────────────────────────────
# COMPARATIVE ANALYSIS
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("COMPARATIVE MODEL ANALYSIS")
print("=" * 80)

print(f"\n{'MODEL':<12} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'Prec':>7} | {'Recall':>7} | {'F1':>7}")
print(f"{'-' * 75}")

for model_key in MODEL_KEYS:
    errors = model_errors[model_key]
    metrics = compute_metrics(errors['total_tp'], errors['total_fp'], errors['total_fn'])
    print(f"{MODEL_LABELS[model_key]:<12} | {metrics.tp:>5} | {metrics.fp:>5} | {metrics.fn:>5} | "
          f"{metrics.precision:>7.4f} | {metrics.recall:>7.4f} | {metrics.f1:>7.4f}")

# ─────────────────────────────────────────────────────────────────
# LINGUISTIC OBSERVATIONS
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("LINGUISTIC OBSERVATIONS & INSIGHTS")
print("=" * 80)

for model_key in MODEL_KEYS:
    errors = model_errors[model_key]
    metrics = compute_metrics(errors['total_tp'], errors['total_fp'], errors['total_fn'])
    
    print(f"\n📝 {MODEL_LABELS[model_key]}:")
    
    # Variety difficulty analysis
    variety_recalls = {}
    for variety in ['MSA', 'Classical', 'Dialect']:
        stats = errors['variety_stats'].get(variety, {'tp': 0, 'fp': 0, 'fn': 0})
        vm = compute_metrics(stats['tp'], stats['fp'], stats['fn'])
        variety_recalls[variety] = vm.recall
    
    hardest = min(variety_recalls, key=variety_recalls.get)
    easiest = max(variety_recalls, key=variety_recalls.get)
    
    print(f"  • Easiest variety: {easiest} (recall={variety_recalls[easiest]:.4f})")
    print(f"  • Hardest variety: {hardest} (recall={variety_recalls[hardest]:.4f})")
    
    # Named entity vs generic analysis
    missed_ne_count = sum(1 for kw in errors['missed_keywords'].keys() if is_named_entity(kw))
    missed_generic_count = sum(1 for kw in errors['missed_keywords'].keys() if is_generic_keyword(kw))
    
    print(f"  • Missed keywords: {len(errors['missed_keywords'])} unique")
    print(f"    - Named entities (approx): {missed_ne_count}")
    print(f"    - Generic/common keywords (approx): {missed_generic_count}")
    
    # Hallucination patterns
    if errors['hallucinated_keywords']:
        halluc_avg_length = sum(len(k.split()) for k in errors['hallucinated_keywords'].keys()) / len(errors['hallucinated_keywords'])
        print(f"  • Hallucinated keywords: {len(errors['hallucinated_keywords'])} unique")
        print(f"    - Average phrase length: {halluc_avg_length:.2f} words")
    
    # Precision vs Recall balance
    if metrics.precision > metrics.recall:
        print(f"  • Model is conservative (high precision {metrics.precision:.4f}, low recall {metrics.recall:.4f})")
    elif metrics.recall > metrics.precision:
        print(f"  • Model is generative (low precision {metrics.precision:.4f}, high recall {metrics.recall:.4f})")
    else:
        print(f"  • Balanced behavior (precision={metrics.precision:.4f}, recall={metrics.recall:.4f})")

# ─────────────────────────────────────────────────────────────────
# SAVE DIFFICULT ERROR CASES
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SAVING DIFFICULT ERROR CASES")
print("=" * 80)

error_cases_export = {}

for model_key in MODEL_KEYS:
    error_cases_export[model_key] = {
        'model_name': MODEL_LABELS[model_key],
        'top_20_cases': []
    }
    
    for case in model_errors[model_key]['error_cases']:
        error_cases_export[model_key]['top_20_cases'].append({
            'id': case.item_id,
            'variety': case.variety,
            'topic': case.topic,
            'gold_keyphrases': case.gold,
            'predicted_keyphrases': case.predicted,
            'true_positives': case.tp,
            'false_positives': case.fp,
            'false_negatives': case.fn,
            'error_severity': round(case.error_severity, 4),
        })

error_cases_path = BASE_DIR / 'error_cases.json'
with error_cases_path.open('w', encoding='utf-8') as f:
    json.dump(error_cases_export, f, ensure_ascii=False, indent=2)

print(f"\n✓ Saved error cases file: {error_cases_path.name}")
print(f"  Includes top 20 difficult cases per model")

# ─────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

# Best and worst models
model_f1s = {}
for model_key in MODEL_KEYS:
    errors = model_errors[model_key]
    metrics = compute_metrics(errors['total_tp'], errors['total_fp'], errors['total_fn'])
    model_f1s[model_key] = metrics.f1

best_model = max(model_f1s, key=model_f1s.get)
worst_model = min(model_f1s, key=model_f1s.get)

print(f"\n🏆 Best model (by F1):  {MODEL_LABELS[best_model]} (F1={model_f1s[best_model]:.4f})")
print(f"📉 Worst model (by F1): {MODEL_LABELS[worst_model]} (F1={model_f1s[worst_model]:.4f})")

# Variety-wide summary
print(f"\n🌍 DIFFICULTY BY VARIETY (average recall across all models):")
variety_recalls_all = defaultdict(list)
for model_key in MODEL_KEYS:
    errors = model_errors[model_key]
    for variety in ['MSA', 'Classical', 'Dialect']:
        stats = errors['variety_stats'].get(variety, {'tp': 0, 'fp': 0, 'fn': 0})
        vm = compute_metrics(stats['tp'], stats['fp'], stats['fn'])
        variety_recalls_all[variety].append(vm.recall)

for variety in ['MSA', 'Classical', 'Dialect']:
    avg_recall = sum(variety_recalls_all[variety]) / len(variety_recalls_all[variety])
    print(f"  {variety:<15}: avg recall = {avg_recall:.4f}")

print("\n" + "=" * 80)
print("✓ Error analysis complete!")
print("=" * 80)
