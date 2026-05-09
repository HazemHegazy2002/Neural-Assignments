"""
STEP 0 — FETCH REAL ARABIC PARAGRAPHS (Task 10: Keyphrase Extraction)

Sources (CC BY-SA 4.0):
  - MSA:       https://ar.wikipedia.org
  - Classical: https://ar.wikisource.org
  - Dialect:   https://arz.wikipedia.org (Egyptian)
              https://ary.wikipedia.org (Moroccan)
              https://aeb.wikipedia.org (Tunisian)
              https://arq.wikipedia.org (Algerian)

Output:
  real_paragraphs.json
"""

import json
import os
import random
import re
import time
from collections import Counter
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

random.seed(42)

USER_AGENT = "Mozilla/5.0 (Task10-DataFetcher/1.0)"
MIN_CHARS = 220
MAX_CHARS = 700
REQUEST_DELAY = 2.0
MAX_RETRIES = 8

MSA_TARGET = {
    "variety": "MSA",
    "api": "https://ar.wikipedia.org/w/api.php",
    "site": "ar.wikipedia.org",
    "count": 27,
}

CLASSICAL_TARGET = {
    "variety": "Classical",
    "api": "https://ar.wikisource.org/w/api.php",
    "site": "ar.wikisource.org",
    "count": 27,
}

DIALECT_TARGETS = [
    {"variety": "Dialect", "api": "https://arz.wikipedia.org/w/api.php",
     "site": "arz.wikipedia.org", "count": 8},
    {"variety": "Dialect", "api": "https://ary.wikipedia.org/w/api.php",
     "site": "ary.wikipedia.org", "count": 6},
    {"variety": "Dialect", "api": "https://aeb.wikipedia.org/w/api.php",
     "site": "aeb.wikipedia.org", "count": 6},
    {"variety": "Dialect", "api": "https://arq.wikipedia.org/w/api.php",
     "site": "arq.wikipedia.org", "count": 6},
]

AR_STOPWORDS = {
    "في", "من", "على", "الى", "إلى", "عن", "أن", "إن", "كان", "كانت",
    "هو", "هي", "هم", "هن", "هذا", "هذه", "ذلك", "تلك", "هناك",
    "كما", "مثل", "وقد", "وقد", "وقد", "تم", "تمت", "مع", "بين",
    "بعد", "قبل", "أو", "و", "ثم", "لكن", "لأن", "الذي", "التي",
    "الذين", "اللاتي", "اللاتي", "أي", "أيضاً", "ايضا", "حتى",
    "كانت", "يكون", "يمكن", "يمكن", "كل", "بعض", "أكثر", "أقل",
}


def fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_RETRIES):
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            time.sleep(REQUEST_DELAY)
            return data
        except HTTPError as exc:
            if exc.code in (429, 503):
                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(int(retry_after))
                    except ValueError:
                        time.sleep(2 ** attempt)
                else:
                    time.sleep(2 ** attempt)
                continue
            raise
        except URLError:
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to fetch data after retries")


def get_random_pages(api_url: str, limit: int) -> list:
    url = (
        f"{api_url}?action=query&generator=random&grnnamespace=0"
        f"&grnlimit={limit}&prop=extracts&explaintext=1&exintro=1&format=json"
    )
    data = fetch_json(url)
    pages = data.get("query", {}).get("pages", {})
    return list(pages.values())


def clean_text(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_disambiguation(text: str) -> bool:
    compact = text.replace(" ", "")
    return (
        "قديشير" in compact or
        "قدتشير" in compact or
        "صفحةتوضيح" in compact or
        "قديقصد" in compact
    )


def arabic_ratio(text: str) -> float:
    arabic = len(re.findall(r"[\u0600-\u06FF]", text))
    letters = len(re.findall(r"[A-Za-z\u0600-\u06FF]", text))
    return arabic / max(1, letters)


def pick_paragraph(extract: str) -> str | None:
    if not extract:
        return None
    parts = [p.strip() for p in extract.split("\n") if p.strip()]
    for part in parts:
        text = clean_text(part)
        if is_disambiguation(text):
            continue
        if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
            continue
        if arabic_ratio(text) < 0.65:
            continue
        return text
    return None


def generate_keyphrases(paragraph: str, title: str) -> list:
    tokens = re.findall(r"[\u0600-\u06FF]{2,}", paragraph)
    tokens = [t for t in tokens if t not in AR_STOPWORDS]
    counts = Counter(tokens)

    bigrams = []
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]} {tokens[i + 1]}"
        bigrams.append(bg)
    bg_counts = Counter(bigrams)

    keyphrases = []
    if title and title not in keyphrases:
        keyphrases.append(title)

    for bg, _ in bg_counts.most_common(3):
        if bg not in keyphrases:
            keyphrases.append(bg)
        if len(keyphrases) >= 5:
            break

    if len(keyphrases) < 5:
        for word, _ in counts.most_common(10):
            if word not in keyphrases:
                keyphrases.append(word)
            if len(keyphrases) >= 5:
                break

    return keyphrases[:5]


def save_progress(entries: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def collect_entries(target: dict, all_entries: list,
                    path: str = "real_paragraphs.json") -> None:
    variety = target["variety"]
    api_url = target["api"]
    site = target["site"]
    count = target["count"]

    current = [e for e in all_entries
               if e.get("type") == variety and e.get("source_site") == site]
    remaining = count - len(current)
    if remaining <= 0:
        return

    seen_titles = {e.get("source_title") for e in current if e.get("source_title")}
    attempts = 0
    retrieved_date = date.today().isoformat()

    while remaining > 0 and attempts < 400:
        attempts += 1
        pages = get_random_pages(api_url, limit=min(10, max(remaining * 2, 6)))
        for page in pages:
            title = page.get("title", "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            extract = page.get("extract", "")
            paragraph = pick_paragraph(extract)
            if not paragraph:
                continue

            entry = {
                "id": len(all_entries) + 1,
                "type": variety,
                "topic": title,
                "paragraph": paragraph,
                "gold_keyphrases": generate_keyphrases(paragraph, title),
                "source_title": title,
                "source_url": f"https://{site}/wiki/{quote(title)}",
                "source_site": site,
                "license": "CC BY-SA 4.0",
                "retrieved_date": retrieved_date,
            }
            all_entries.append(entry)
            save_progress(all_entries, path)
            remaining -= 1
            print(f"[{variety}] {count - remaining}/{count}: {title}")
            if remaining <= 0:
                break
        time.sleep(REQUEST_DELAY)

    if remaining > 0:
        raise RuntimeError(f"Not enough {variety} paragraphs collected.")


def main() -> None:
    path = "real_paragraphs.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            all_entries = json.load(f)
    else:
        all_entries = []

    collect_entries(MSA_TARGET, all_entries, path)
    time.sleep(2)
    collect_entries(CLASSICAL_TARGET, all_entries, path)
    time.sleep(2)
    for target in DIALECT_TARGETS:
        collect_entries(target, all_entries, path)
        time.sleep(2)

    print(f"\nSaved {path} with {len(all_entries)} items")


if __name__ == "__main__":
    main()
