"""
=============================================================================
STEP 3 — KEYPHRASE BASELINE EVALUATION
Task 10: Keyphrase Extraction
Problem 5 — Arabic NLP Benchmarking
=============================================================================

This script evaluates a simple unsupervised baseline on the real Problem 5
dataset using Precision@5, Recall@5, and F1@5.

Outputs:
  • step3_report.json   — per-item predictions and metrics
  • step3_summary.txt   — human-readable evaluation summary
=============================================================================
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


DATASET_PATH = Path(__file__).with_name("dataset.json")
REPORT_JSON = Path(__file__).with_name("step3_report.json")
REPORT_TXT = Path(__file__).with_name("step3_summary.txt")
K = 5

STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "هناك", "هنا", "كما", "كان", "كانت", "يكون", "تكون", "وقد", "التي",
    "الذي", "الذين", "ما", "ماذا", "متى", "أو", "و", "ثم", "بل", "لكن",
    "لأن", "إن", "أن", "إذا", "حتى", "قد", "لا", "لم", "لن", "ليس",
    "كل", "بعض", "أي", "هو", "هي", "هم", "هن", "أنا", "نحن", "أنت",
    "أنتم", "له", "لها", "لهم", "فيه", "فيها", "عند", "عندما", "بين",
    "بعد", "قبل", "خلال", "حول", "فوق", "تحت", "على", "الى", "يا", "دي",
    "ده", "دا", "بس", "كمان", "لما", "فيه", "عشان", "اللي", "التي", "اللى",
}

WORD_RE = re.compile(r"[\u0600-\u06FF]+")


def normalize(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    tokens = WORD_RE.findall(normalize(text))
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def load_dataset() -> list[dict]:
    with DATASET_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_idf(dataset: list[dict]) -> tuple[Counter, int]:
    document_frequency = Counter()
    for item in dataset:
        document_frequency.update(set(tokenize(item["paragraph"])))
    return document_frequency, len(dataset)


def predict_keyphrases(paragraph: str, document_frequency: Counter, total_docs: int, k: int = K) -> list[str]:
    tokens = tokenize(paragraph)
    if not tokens:
        return []

    term_frequency = Counter(tokens)
    first_seen = {}
    for index, token in enumerate(tokens):
        first_seen.setdefault(token, index)

    scored = []
    for token, freq in term_frequency.items():
        idf = math.log((1 + total_docs) / (1 + document_frequency[token])) + 1.0
        score = freq * idf
        scored.append((token, score, first_seen[token]))

    scored.sort(key=lambda item: (-item[1], item[2], item[0]))

    predictions = []
    for token, _, _ in scored:
        if token not in predictions:
            predictions.append(token)
        if len(predictions) >= k:
            break
    return predictions


def metric_scores(predicted: list[str], gold: list[str]) -> dict:
    predicted_set = {normalize(item) for item in predicted}
    gold_set = {normalize(item) for item in gold}
    if not predicted_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hits": 0}

    hits = len(predicted_set & gold_set)
    precision = hits / len(predicted_set)
    recall = hits / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hits": hits,
    }


def summarize_by_group(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["type"]].append(record)

    summary = {}
    for group, group_records in grouped.items():
        summary[group] = {
            "count": len(group_records),
            "avg_precision": round(sum(r["precision"] for r in group_records) / len(group_records), 4),
            "avg_recall": round(sum(r["recall"] for r in group_records) / len(group_records), 4),
            "avg_f1": round(sum(r["f1"] for r in group_records) / len(group_records), 4),
        }
    return summary


def main() -> None:
    dataset = load_dataset()
    document_frequency, total_docs = build_idf(dataset)

    per_item = []
    for item in dataset:
        predicted = predict_keyphrases(item["paragraph"], document_frequency, total_docs, K)
        metrics = metric_scores(predicted, item.get("gold_keyphrases", []))
        per_item.append({
            "id": item["id"],
            "type": item["type"],
            "topic": item.get("topic", ""),
            "source": item.get("source", ""),
            "source_detail": item.get("source_detail", ""),
            "predicted_keyphrases": predicted,
            "gold_keyphrases": item.get("gold_keyphrases", []),
            **metrics,
        })

    macro_precision = round(sum(item["precision"] for item in per_item) / len(per_item), 4)
    macro_recall = round(sum(item["recall"] for item in per_item) / len(per_item), 4)
    macro_f1 = round(sum(item["f1"] for item in per_item) / len(per_item), 4)
    hit_rate = round(sum(1 for item in per_item if item["hits"] > 0) / len(per_item), 4)

    report = {
        "task": "Task 10 — Keyphrase Extraction",
        "step": "Step 3 — Precision@5 Baseline Evaluation",
        "dataset_size": len(dataset),
        "k": K,
        "metrics": {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "hit_rate": hit_rate,
        },
        "by_type": summarize_by_group(per_item),
        "per_item": per_item,
    }

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("=" * 72)
    lines.append("STEP 3 — KEYPHRASE BASELINE EVALUATION")
    lines.append("Task 10: Keyphrase Extraction")
    lines.append("Problem 5 — Arabic NLP Benchmarking")
    lines.append("=" * 72)
    lines.append(f"Dataset size        : {len(dataset)}")
    lines.append(f"Evaluation cutoff   : Precision@{K}")
    lines.append(f"Macro Precision@{K}  : {macro_precision}")
    lines.append(f"Macro Recall@{K}     : {macro_recall}")
    lines.append(f"Macro F1@{K}         : {macro_f1}")
    lines.append(f"Hit rate            : {hit_rate}")
    lines.append("")
    lines.append("By variety:")
    for variety in ["MSA", "Classical", "Dialect"]:
        stats = report["by_type"].get(variety, {})
        lines.append(
            f"  {variety:<10} count={stats.get('count', 0)} "
            f"P@{K}={stats.get('avg_precision', 0)} "
            f"R@{K}={stats.get('avg_recall', 0)} F1@{K}={stats.get('avg_f1', 0)}"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("  • This is a lightweight unsupervised baseline, not the final model.")
    lines.append("  • Use the report as the starting point for model comparison in later steps.")
    lines.append("  • Real-data provenance is preserved in the dataset.json file.")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {REPORT_JSON.name} and {REPORT_TXT.name}")


if __name__ == "__main__":
    main()