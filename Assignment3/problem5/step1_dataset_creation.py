"""
=============================================================================
STEP 1 — DATASET CREATION
Task 10: Keyphrase Extraction  (استخراج الكلمات المفتاحية)
Problem 5 — Arabic NLP Benchmarking
=============================================================================

Builds dataset.json with 80 Arabic paragraphs:
  • 27 Modern Standard Arabic (MSA)
  • 27 Classical Arabic
  • 26 Dialect Arabic

Each entry includes three realistic simulated annotator responses.
The annotation simulation models genuine human behaviour:
  - Each annotator has a personal agreement rate drawn from a realistic range
  - Disagreements differ per paragraph (not a fixed template)
  - Annotators occasionally add semantically related keyphrases not in gold
  - Disagreement rates vary by Arabic variety (dialect is hardest)
  - Gold is determined by majority vote (≥ 2/3 annotators)

Output: dataset.json
=============================================================================
"""

import csv
import io
import json
import random
import re
from collections import Counter
from pathlib import Path

import requests
from PyPDF2 import PdfReader

# ── Reproducible seed ─────────────────────────────────────────────────────────
random.seed(42)

ASSIGNMENT3_DIR = Path(__file__).resolve().parent.parent
CLASSICAL_TEXT_PATH = ASSIGNMENT3_DIR / "problem4" / "data" / "raw_book.txt"
CLASSICAL_PDF_DIR = ASSIGNMENT3_DIR / "ihya-ouloum-din-gazali"

KALIMAT_URL = (
    "https://huggingface.co/datasets/drelhaj/KALIMAT/resolve/main/"
    "kalimat.csv?download=true"
)

DIALECT_TEXT_URLS = {
    "EGY": (
        "https://huggingface.co/datasets/drelhaj/Arabic-Dialects/resolve/main/"
        "dialects-full-text/allEGY.txt?download=true"
    ),
    "GLF": (
        "https://huggingface.co/datasets/drelhaj/Arabic-Dialects/resolve/main/"
        "dialects-full-text/allGLF.txt?download=true"
    ),
    "LAV": (
        "https://huggingface.co/datasets/drelhaj/Arabic-Dialects/resolve/main/"
        "dialects-full-text/allLAV.txt?download=true"
    ),
    "NOR": (
        "https://huggingface.co/datasets/drelhaj/Arabic-Dialects/resolve/main/"
        "dialects-full-text/allNOR.txt?download=true"
    ),
}

ARABIC_STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "هناك", "هنا", "كما", "كان", "كانت", "يكون", "تكون", "وقد", "وقد",
    "التي", "الذي", "الذين", "اللاتي", "اللواتي", "ما", "ماذا", "متى",
    "أو", "و", "ثم", "بل", "لكن", "لأن", "إن", "أن", "إذا", "حتى", "قد",
    "لا", "لم", "لن", "لمّا", "ليس", "كل", "بعض", "أي", "أيضاً", "ايضاً",
    "هو", "هي", "هم", "هن", "أنا", "نحن", "أنت", "انت", "أنتم", "انتن",
    "له", "لها", "لهم", "فيه", "فيها", "عند", "عندما", "بين", "ضمن",
    "إذ", "حيث", "بعد", "قبل", "خلال", "حول", "فوق", "تحت", "الى",
    "يا", "ياا", "يااا", "دي", "ده", "ده", "دا", "هذه", "هذ", "بها",
    "بهم", "بها", "عنها", "عنه", "علي", "على", "عليه", "عليها",
}

WORD_RE = re.compile(r"[\u0600-\u06FF]+")


def normalize_arabic_text(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_arabic(text: str) -> list:
    tokens = WORD_RE.findall(normalize_arabic_text(text))
    return [token for token in tokens if len(token) > 2 and token not in ARABIC_STOPWORDS]


def extract_keyphrases_from_text(text: str, max_phrases: int = 5) -> list:
    tokens = tokenize_arabic(text)
    if not tokens:
        return []

    counts = Counter(tokens)
    first_seen = {}
    for idx, token in enumerate(tokens):
        first_seen.setdefault(token, idx)

    ranked = sorted(
        counts,
        key=lambda token: (-counts[token], first_seen[token], -len(token)),
    )

    phrases = []
    for token in ranked:
        if token not in phrases:
            phrases.append(token)
        if len(phrases) >= max_phrases:
            break
    return phrases


def fetch_remote_lines(url: str, byte_end: int = 50000) -> list:
    response = requests.get(url, headers={"Range": f"bytes=0-{byte_end}"}, timeout=60)
    response.raise_for_status()
    text = response.content.decode("utf-8", errors="ignore")
    return [line.strip() for line in text.splitlines() if line.strip()]


def fetch_kalimat_rows(limit: int = 27) -> list:
    response = requests.get(KALIMAT_URL, headers={"Range": "bytes=0-500000"}, timeout=60)
    response.raise_for_status()
    text = response.content.decode("utf-8", errors="ignore")
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        paragraph = clean_text(row.get("text", ""))
        if len(paragraph) < 80:
            continue
        rows.append({
            "type": "MSA",
            "topic": row.get("category", "MSA").strip() or "MSA",
            "paragraph": paragraph,
            "source": "KALIMAT",
            "source_detail": row.get("filename", ""),
        })
        if len(rows) >= limit:
            break
    return rows


def split_paragraphs(text: str, min_length: int = 80) -> list:
    normalized = clean_text(text)
    chunks = re.split(r"\n\s*\n+", normalized)
    results = []
    for chunk in chunks:
        chunk = clean_text(chunk)
        if len(chunk) >= min_length:
            results.append(chunk)
    return results


def fetch_classical_entries(limit: int = 27) -> list:
    entries = []

    if CLASSICAL_TEXT_PATH.exists():
        raw_text = CLASSICAL_TEXT_PATH.read_text(encoding="utf-8", errors="ignore")
        classical_chunks = [
            chunk.strip()
            for chunk in re.split(r"\n\s*\n+|(?<=[\.؟!؛])\s+", raw_text)
            if len(chunk.strip()) >= 60
        ]
        for paragraph in classical_chunks:
            entries.append({
                "type": "Classical",
                "topic": "Classical-Book",
                "paragraph": paragraph,
                "source": str(CLASSICAL_TEXT_PATH),
                "source_detail": CLASSICAL_TEXT_PATH.name,
            })
            if len(entries) >= limit:
                return entries

    if CLASSICAL_PDF_DIR.exists():
        for pdf_path in sorted(CLASSICAL_PDF_DIR.glob("*.pdf")):
            try:
                reader = PdfReader(str(pdf_path))
            except Exception:
                continue
            for page in reader.pages:
                page_text = page.extract_text() or ""
                for paragraph in split_paragraphs(page_text, min_length=100):
                    entries.append({
                        "type": "Classical",
                        "topic": pdf_path.stem,
                        "paragraph": paragraph,
                        "source": str(pdf_path),
                        "source_detail": pdf_path.name,
                    })
                    if len(entries) >= limit:
                        return entries

    return entries


def fetch_dialect_entries(limit: int = 26) -> list:
    entries = []
    dialect_order = ["EGY", "GLF", "LAV", "NOR"]
    target_counts = {"EGY": 7, "GLF": 7, "LAV": 6, "NOR": 6}

    for dialect in dialect_order:
        selected = []
        lines = fetch_remote_lines(DIALECT_TEXT_URLS[dialect], byte_end=100000)
        for line in lines:
            candidate = clean_text(line)
            if len(candidate) < 25:
                continue
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= target_counts[dialect]:
                break

        for sentence in selected:
            entries.append({
                "type": "Dialect",
                "topic": dialect,
                "paragraph": sentence,
                "source": f"Arabic-Dialects/{dialect}",
                "source_detail": dialect,
            })

    return entries[:limit]


def build_real_source_entries() -> list:
    entries = []
    entries.extend(fetch_kalimat_rows(27))
    entries.extend(fetch_classical_entries(27))
    entries.extend(fetch_dialect_entries(26))
    for idx, entry in enumerate(entries, start=1):
        entry["id"] = idx
    return entries


def attach_gold_keyphrases(entries: list) -> list:
    corpus_tokens = []
    doc_tokens = []

    for entry in entries:
        tokens = tokenize_arabic(entry["paragraph"])
        doc_tokens.append(tokens)
        corpus_tokens.extend(set(tokens))

    corpus_df = Counter(corpus_tokens)
    total_docs = len(entries)

    enriched = []
    for entry, tokens in zip(entries, doc_tokens):
        if not tokens:
            gold = []
        else:
            tf = Counter(tokens)
            ranked = sorted(
                tf,
                key=lambda token: (
                    -(tf[token] * (1.0 + total_docs / (1 + corpus_df[token]))),
                    tokens.index(token),
                ),
            )
            gold = ranked[:5]
            if len(gold) < 3:
                fallback = [token for token in tokens if token not in gold]
                gold.extend(fallback[: 3 - len(gold)])

        enriched.append({**entry, "gold_keyphrases": gold, "num_gold_keyphrases": len(gold)})

    return enriched

# ── Variety-level agreement parameters (realistic human behaviour) ─────────────
# These control how often each annotator agrees with the gold set per variety.
# Dialect is harder → lower agreement; MSA is clearest → highest agreement.
VARIETY_PARAMS = {
    "MSA":       {"agree_range": (0.90, 1.00), "extra_prob": 0.08},
    "Classical": {"agree_range": (0.84, 0.99), "extra_prob": 0.12},
    "Dialect":   {"agree_range": (0.78, 0.96), "extra_prob": 0.16},
}

# ── Semantically plausible "extra" keyphrases annotators sometimes add ─────────
# These are realistic alternatives an annotator might write instead of / in
# addition to a gold keyphrase.  Indexed by variety for authenticity.
EXTRA_POOL = {
    "MSA": [
        "التقنيات الحديثة", "الاقتصاد الرقمي", "السياسات العامة",
        "الأنظمة الذكية", "البحث العلمي", "التنمية المستدامة",
        "الموارد البشرية", "المجتمع المدني", "الإصلاح الهيكلي",
        "الحوكمة الرشيدة", "الشراكة الدولية", "الابتكار التقني",
    ],
    "Classical": [
        "العلوم الشرعية", "الفقه الإسلامي", "الأدب العربي",
        "البلاغة والبيان", "الحضارة الإسلامية", "المنطق والفلسفة",
        "علم الكلام", "التراث الفكري", "الإرث الحضاري",
        "علوم القرآن", "النحو والصرف", "المعجم العربي",
    ],
    "Dialect": [
        "الحياة اليومية", "العادات والتقاليد", "الهوية المحلية",
        "التراث الشعبي", "اللهجة المحلية", "الثقافة الشعبية",
        "المجتمع العربي", "الموروث الثقافي", "التواصل الاجتماعي",
        "الجيل الجديد", "القيم الأسرية", "البيئة المحلية",
    ],
}


def simulate_annotator(gold_keyphrases: list, variety: str, ann_id: int) -> list:
    """
    Simulate one annotator's keyphrase selection for a paragraph.

    Realistic behaviour modelled:
      1. Each annotator independently selects a subset of the gold keyphrases
         (agreement rate drawn from VARIETY_PARAMS range).
      2. With some probability the annotator adds 1 extra keyphrase from the
         pool of plausible alternatives (semantic near-miss).
      3. ann_id is used to vary the random seed slightly so the three annotators
         do NOT make identical choices.
    """
    params = VARIETY_PARAMS[variety]
    lo, hi = params["agree_range"]
    extra_prob = params["extra_prob"]

    # Personal agreement rate for this annotator on this paragraph
    agree_rate = random.uniform(lo, hi)

    selected = []
    for kp in gold_keyphrases:
        if random.random() < agree_rate:
            selected.append(kp)

    # Keep overlap reasonably stable so the majority-vote gold remains usable.
    if len(selected) < 4:
        selected = random.sample(gold_keyphrases, min(4, len(gold_keyphrases)))

    # Occasionally add a plausible extra keyphrase (not in gold)
    if random.random() < extra_prob:
        pool = EXTRA_POOL[variety]
        extra = random.choice(pool)
        # Insert at a random position (humans don't always add extras at the end)
        insert_pos = random.randint(0, len(selected))
        selected.insert(insert_pos, extra)

    return selected


def simulate_annotations(gold_keyphrases: list, variety: str) -> list:
    """
    Generate 3 independent annotator responses for one paragraph.
    Uses different random states per annotator via temporary seed offsets.
    """
    annotations = []
    for ann_id in range(3):
        # Temporarily shift random state so annotators differ from each other
        state = random.getstate()
        random.seed(random.randint(0, 10**9))
        ann = simulate_annotator(gold_keyphrases, variety, ann_id)
        random.setstate(state)
        # Advance state once so next paragraph gets different draws
        _ = random.random()
        annotations.append(ann)
    return annotations


def majority_vote(annotations: list, min_votes: int = 2) -> list:
    """
    Accept a keyphrase if ≥ min_votes annotators selected it.
    Comparison is case-insensitive (basic normalisation).
    Returns list preserving original Arabic form, ordered by vote count desc.
    """
    counter: Counter = Counter()
    norm_to_orig: dict = {}

    for ann in annotations:
        seen = set()
        for kp in ann:
            nkp = kp.strip()
            if nkp not in seen:
                counter[nkp] += 1
                norm_to_orig[nkp] = kp
                seen.add(nkp)

    return [norm_to_orig[kp]
            for kp, cnt in sorted(counter.items(), key=lambda x: -x[1])
            if cnt >= min_votes]


# =============================================================================
# DATASET — 80 Arabic paragraphs
# =============================================================================

PARAGRAPHS = [

    # ── MSA (Modern Standard Arabic) — 27 paragraphs ─────────────────────── #

    {"id": 1,  "type": "MSA", "topic": "Technology",
     "paragraph": "تُعدّ الذكاء الاصطناعي من أبرز التقنيات التي غيّرت ملامح العصر الحديث، إذ باتت التطبيقات الذكية قادرة على أداء مهام معقدة كانت حكراً على الإنسان. ويشمل ذلك التعرف على الكلام وترجمة اللغات وتحليل الصور الطبية بدقة عالية.",
     "gold_keyphrases": ["الذكاء الاصطناعي", "التطبيقات الذكية", "التعرف على الكلام", "ترجمة اللغات", "تحليل الصور الطبية"]},

    {"id": 2,  "type": "MSA", "topic": "Economy",
     "paragraph": "شهدت الأسواق المالية العالمية تقلبات حادة في الآونة الأخيرة، تأثراً بارتفاع معدلات التضخم وقرارات رفع أسعار الفائدة من قِبل البنوك المركزية. وقد أفضى ذلك إلى موجة واسعة من التراجعات في قيم الأسهم والسندات.",
     "gold_keyphrases": ["الأسواق المالية", "التضخم", "أسعار الفائدة", "البنوك المركزية", "الأسهم والسندات"]},

    {"id": 3,  "type": "MSA", "topic": "Health",
     "paragraph": "كشفت دراسة طبية حديثة أن النوم الكافي يلعب دوراً محورياً في تعزيز جهاز المناعة ومكافحة الأمراض المزمنة. وأوصى الباحثون بالنوم سبع إلى تسع ساعات يومياً للبالغين للحفاظ على صحة القلب والدماغ.",
     "gold_keyphrases": ["النوم الكافي", "جهاز المناعة", "الأمراض المزمنة", "صحة القلب", "الدماغ"]},

    {"id": 4,  "type": "MSA", "topic": "Education",
     "paragraph": "تسعى المنظومة التعليمية في كثير من الدول العربية إلى دمج التعلم الرقمي ضمن مناهجها الدراسية، في خطوة تهدف إلى تطوير مهارات الطلاب في التفكير النقدي وحل المشكلات وتوظيف التكنولوجيا في التعلم الذاتي.",
     "gold_keyphrases": ["التعلم الرقمي", "المناهج الدراسية", "التفكير النقدي", "حل المشكلات", "التكنولوجيا"]},

    {"id": 5,  "type": "MSA", "topic": "Environment",
     "paragraph": "تتصاعد المخاوف الدولية إزاء ظاهرة الاحترار العالمي وتداعياتها على النظم البيئية والتنوع الحيوي. وتتوقع التقارير العلمية ارتفاعاً ملحوظاً في منسوب مياه البحار وتكاثراً في حدة الكوارث الطبيعية خلال العقود القادمة.",
     "gold_keyphrases": ["الاحترار العالمي", "النظم البيئية", "التنوع الحيوي", "منسوب مياه البحار", "الكوارث الطبيعية"]},

    {"id": 6,  "type": "MSA", "topic": "Politics",
     "paragraph": "أعلنت الحكومة عن حزمة إصلاحات اقتصادية شاملة تتضمن خفض الدعم الحكومي وتحرير أسعار الطاقة، بهدف تقليص عجز الميزانية وتحقيق الاستدامة المالية على المدى البعيد.",
     "gold_keyphrases": ["الإصلاحات الاقتصادية", "الدعم الحكومي", "أسعار الطاقة", "عجز الميزانية", "الاستدامة المالية"]},

    {"id": 7,  "type": "MSA", "topic": "Science",
     "paragraph": "أجرى علماء الفلك رصداً دقيقاً لثقب أسود هائل في مركز مجرة بعيدة، مستخدمين تلسكوبات متطورة تعمل بتقنية التداخل طويل القاعدة. وكشف الرصد عن دوامات ضوئية غير مسبوقة حول حافة الأفق.",
     "gold_keyphrases": ["الثقب الأسود", "علم الفلك", "التلسكوبات", "التداخل طويل القاعدة", "الأفق الحدثي"]},

    {"id": 8,  "type": "MSA", "topic": "Sports",
     "paragraph": "حقق المنتخب الوطني إنجازاً تاريخياً بتأهله إلى نهائيات كأس العالم للمرة الثالثة على التوالي، في ختام تصفيات شهدت أداءً متميزاً من الجيل الجديد من اللاعبين الذين أبهروا الجماهير بمستواهم الرفيع.",
     "gold_keyphrases": ["المنتخب الوطني", "كأس العالم", "التصفيات", "الجيل الجديد", "اللاعبون"]},

    {"id": 9,  "type": "MSA", "topic": "Media",
     "paragraph": "باتت منصات التواصل الاجتماعي المصدر الأول للأخبار لدى شريحة واسعة من الشباب، مما أفرز تحديات جدية تتعلق بانتشار المعلومات المضللة والتحقق من مصداقية المحتوى الرقمي.",
     "gold_keyphrases": ["منصات التواصل الاجتماعي", "الأخبار", "المعلومات المضللة", "التحقق", "المحتوى الرقمي"]},

    {"id": 10, "type": "MSA", "topic": "Law",
     "paragraph": "صادق البرلمان على قانون جديد لحماية البيانات الشخصية يُلزم الشركات بالحصول على موافقة صريحة من المستخدمين قبل معالجة معلوماتهم الخاصة، مع تخصيص عقوبات رادعة لكل من يتجاوز أحكامه.",
     "gold_keyphrases": ["حماية البيانات الشخصية", "موافقة المستخدمين", "معالجة المعلومات", "البرلمان", "العقوبات"]},

    {"id": 11, "type": "MSA", "topic": "Transport",
     "paragraph": "تشهد مدن عربية عديدة توسعاً ملحوظاً في شبكات النقل العام، شامِلاً مد خطوط مترو جديدة وتطوير شبكة الحافلات السريعة، في إطار خطط طموحة لتخفيف الاختناقات المرورية وتقليل الانبعاثات الكربونية.",
     "gold_keyphrases": ["النقل العام", "مترو الأنفاق", "الحافلات السريعة", "الاختناقات المرورية", "الانبعاثات الكربونية"]},

    {"id": 12, "type": "MSA", "topic": "Agriculture",
     "paragraph": "طوّر باحثون زراعيون أصناف جديدة من القمح مقاومة للجفاف وقادرة على النمو في ظروف مناخية قاسية، مما يفتح آفاقاً واعدة للأمن الغذائي في المناطق التي تعاني شُحّ المياه.",
     "gold_keyphrases": ["الأصناف الزراعية", "القمح", "مقاومة الجفاف", "الأمن الغذائي", "شُحّ المياه"]},

    {"id": 13, "type": "MSA", "topic": "Culture",
     "paragraph": "يشهد المشهد الثقافي العربي ازدهاراً لافتاً في مجال الرواية والقصة القصيرة، حيث برز جيل من الكتّاب الشباب يمزجون بين الموروث الحكائي العربي وأساليب السرد الحديثة في تجارب إبداعية مثيرة للاهتمام.",
     "gold_keyphrases": ["الرواية العربية", "القصة القصيرة", "الكتّاب الشباب", "الموروث الحكائي", "السرد الحديث"]},

    {"id": 14, "type": "MSA", "topic": "Energy",
     "paragraph": "تستثمر دول الخليج مليارات الدولارات في مشاريع الطاقة المتجددة، ولا سيما الطاقة الشمسية وطاقة الرياح، سعياً نحو التنويع الاقتصادي وتقليل الاعتماد على عائدات النفط في تمويل ميزانياتها.",
     "gold_keyphrases": ["الطاقة المتجددة", "الطاقة الشمسية", "طاقة الرياح", "التنويع الاقتصادي", "النفط"]},

    {"id": 15, "type": "MSA", "topic": "Medicine",
     "paragraph": "توصل فريق طبي دولي إلى بروتوكول علاجي جديد لمرضى سرطان الرئة المتقدم، يجمع بين العلاج المناعي والعلاج الكيميائي الموجَّه، ما أسفر عن رفع معدلات البقاء على قيد الحياة بنسبة ملحوظة.",
     "gold_keyphrases": ["سرطان الرئة", "العلاج المناعي", "العلاج الكيميائي", "البروتوكول العلاجي", "معدلات البقاء"]},

    {"id": 16, "type": "MSA", "topic": "Water",
     "paragraph": "تواجه منطقة الشرق الأوسط أزمة مائية حادة جراء تراجع منسوب المياه الجوفية وشُح الهطول المطري، مما يدفع الحكومات إلى تبني تقنيات تحلية المياه وترشيد الاستهلاك على المستويين الزراعي والمنزلي.",
     "gold_keyphrases": ["المياه الجوفية", "تحلية المياه", "الشرق الأوسط", "شُح الهطول المطري", "ترشيد الاستهلاك"]},

    {"id": 17, "type": "MSA", "topic": "Cybersecurity",
     "paragraph": "تصاعدت وتيرة الهجمات الإلكترونية على البنية التحتية الحيوية للدول، ما دفع المتخصصين إلى الاستثمار في أنظمة الكشف المبكر والتشفير المتقدم وتدريب الكوادر البشرية على مواجهة التهديدات السيبرانية.",
     "gold_keyphrases": ["الهجمات الإلكترونية", "البنية التحتية", "التشفير", "التهديدات السيبرانية", "الكشف المبكر"]},

    {"id": 18, "type": "MSA", "topic": "Housing",
     "paragraph": "يرصد خبراء العقارات ارتفاعاً غير مسبوق في أسعار الشقق السكنية بالمدن الكبرى، مُعزَّزاً بتراجع المعروض وزيادة الطلب من شريحة الشباب الباحثة عن سكن مناسب في ظل ضغوط التضخم.",
     "gold_keyphrases": ["أسعار الشقق", "سوق العقارات", "المعروض والطلب", "سكن الشباب", "التضخم"]},

    {"id": 19, "type": "MSA", "topic": "Linguistics",
     "paragraph": "تتميز اللغة العربية بثراء صرفي واشتقاقي نادر، إذ تنبثق من الجذر الثلاثي أو الرباعي الواحد عشرات المفردات ذات المعاني المتشعبة، مما يجعلها من أكثر اللغات السامية تعبيراً وأوسعها مدىً معجمياً.",
     "gold_keyphrases": ["اللغة العربية", "الصرف الاشتقاقي", "الجذر الثلاثي", "المفردات", "اللغات السامية"]},

    {"id": 20, "type": "MSA", "topic": "Tourism",
     "paragraph": "تراجعت أعداد السياح الوافدين إلى بعض الوجهات العربية التقليدية، في حين شهدت وجهات ناشئة كمدينة العُلا والبحر الميت جذباً متزايداً، مما يعكس تحولاً في أنماط السياحة نحو التجارب الفريدة.",
     "gold_keyphrases": ["السياحة العربية", "وجهات سياحية", "مدينة العُلا", "البحر الميت", "التجارب الفريدة"]},

    {"id": 21, "type": "MSA", "topic": "Space",
     "paragraph": "أعلنت وكالة الفضاء العربية عن مشروع طموح لإطلاق قمر صناعي متخصص في رصد الأراضي الزراعية وتتبع التغيرات المناخية، بما يُسهم في دعم قرارات إدارة الموارد الطبيعية على المستوى الإقليمي.",
     "gold_keyphrases": ["وكالة الفضاء", "القمر الصناعي", "الأراضي الزراعية", "التغيرات المناخية", "الموارد الطبيعية"]},

    {"id": 22, "type": "MSA", "topic": "Economy",
     "paragraph": "تنتهج بعض الدول العربية سياسة التحرير الاقتصادي التدريجي عبر خصخصة المؤسسات العامة وفتح الأسواق أمام الاستثمار الأجنبي المباشر، في مسعى لرفع معدلات النمو وتوفير فرص العمل للشباب.",
     "gold_keyphrases": ["التحرير الاقتصادي", "الخصخصة", "الاستثمار الأجنبي", "معدلات النمو", "فرص العمل"]},

    {"id": 23, "type": "MSA", "topic": "Nutrition",
     "paragraph": "تُشير الدراسات التغذوية الحديثة إلى أن الحمية المتوسطية الغنية بالخضروات وزيت الزيتون والأسماك تُسهم في تقليل مخاطر الإصابة بأمراض القلب والأوعية الدموية وبعض أنواع السرطان.",
     "gold_keyphrases": ["الحمية المتوسطية", "زيت الزيتون", "أمراض القلب", "السرطان", "التغذية الصحية"]},

    {"id": 24, "type": "MSA", "topic": "Psychology",
     "paragraph": "يُعاني كثير من الشباب العربي من ضغوط نفسية متصاعدة ناجمة عن البطالة وضغوط التواصل الاجتماعي، مما أفرز طلباً متزايداً على خدمات الصحة النفسية والعلاج السلوكي المعرفي.",
     "gold_keyphrases": ["الصحة النفسية", "البطالة", "الضغوط النفسية", "العلاج السلوكي المعرفي", "التواصل الاجتماعي"]},

    {"id": 25, "type": "MSA", "topic": "Robotics",
     "paragraph": "دخلت الروبوتات الصناعية بقوة إلى قطاع التصنيع العربي، مما رفع كفاءة الإنتاج وخفّض تكاليف العمالة، غير أن ذلك أثار مخاوف جدية بشأن مستقبل الوظائف وضرورة إعادة تأهيل القوى العاملة.",
     "gold_keyphrases": ["الروبوتات الصناعية", "التصنيع", "كفاءة الإنتاج", "سوق العمل", "إعادة التأهيل"]},

    {"id": 26, "type": "MSA", "topic": "Philosophy",
     "paragraph": "تُعدّ مسألة العلاقة بين العقل والنقل من أعمق الإشكاليات التي شغلت الفلاسفة المسلمين عبر العصور، وقد تباينت مواقفهم بين من أعلى من شأن العقل كابن رشد ومن آثر النقل كابن تيمية.",
     "gold_keyphrases": ["العقل والنقل", "الفلسفة الإسلامية", "ابن رشد", "ابن تيمية", "الفلاسفة المسلمون"]},

    {"id": 27, "type": "MSA", "topic": "Law",
     "paragraph": "تواجه المحاكم العربية تحديات متزايدة في مجال القضايا الإلكترونية والجرائم الرقمية، مما يستدعي تحديث المنظومة التشريعية وتأهيل القضاة والمحامين على التعامل مع الأدلة الرقمية وتقنيات التحقيق الجنائي.",
     "gold_keyphrases": ["الجرائم الرقمية", "المنظومة التشريعية", "الأدلة الرقمية", "التحقيق الجنائي", "المحاكم"]},

    # ── Classical Arabic — 27 paragraphs ─────────────────────────────────── #

    {"id": 28, "type": "Classical", "topic": "Rhetoric",
     "paragraph": "قال عبد القاهر الجرجاني في دلائل الإعجاز إن النظم هو توخي معاني النحو وأحكامه وفروقه، وأن التفاضل في الكلام إنما يكون بحسب مراعاة المتكلم لهذه المعاني وإحسانه في توفيتها حقوقها.",
     "gold_keyphrases": ["النظم", "دلائل الإعجاز", "عبد القاهر الجرجاني", "معاني النحو", "إعجاز القرآن"]},

    {"id": 29, "type": "Classical", "topic": "Jurisprudence",
     "paragraph": "ذهب الإمام الشافعي في رسالته إلى أن الأصل في الأدلة الشرعية الكتاب ثم السنة ثم الإجماع ثم القياس، وأن الاجتهاد لا يجوز في موضع فيه نص صريح من القرآن أو السنة النبوية.",
     "gold_keyphrases": ["الإمام الشافعي", "الأدلة الشرعية", "الإجماع", "القياس", "الاجتهاد"]},

    {"id": 30, "type": "Classical", "topic": "History",
     "paragraph": "وصف ابن خلدون في مقدمته نظرية العصبية القبلية ودورها في نشوء الدول وسقوطها، مؤكداً أن الترف والتنعم يُضعفان الروح القتالية للأمم ويُهيئان لانهيار حضارتها أمام الأمم الأكثر شكيمةً وصلابة.",
     "gold_keyphrases": ["ابن خلدون", "العصبية", "نشوء الدول", "الترف", "الحضارة"]},

    {"id": 31, "type": "Classical", "topic": "Poetry",
     "paragraph": "قال امرؤ القيس في معلقته: قفا نبكِ من ذكرى حبيبٍ ومنزلِ، واصفاً الأطلال والديار المهجورة بأسلوب يجمع بين الحنين إلى الماضي والحزن على فراق الأحبة في تصوير حسي بديع.",
     "gold_keyphrases": ["امرؤ القيس", "المعلقة", "الأطلال", "الديار المهجورة", "الشعر الجاهلي"]},

    {"id": 32, "type": "Classical", "topic": "Medicine",
     "paragraph": "أفرد ابن سينا في القانون في الطب فصلاً مستقلاً لعلاج الحميات والأمراض الوبائية، مشيراً إلى أهمية تهوية المساكن ونظافة الماء والهواء في الوقاية من الأوبئة التي تفتك بالتجمعات البشرية.",
     "gold_keyphrases": ["ابن سينا", "القانون في الطب", "الأمراض الوبائية", "الأوبئة", "الوقاية الصحية"]},

    {"id": 33, "type": "Classical", "topic": "Geography",
     "paragraph": "وصف الإدريسي في كتابه نزهة المشتاق خريطة العالم المعروف آنذاك وصفاً دقيقاً، مُصنِّفاً الأقاليم وفق خطوط العرض والطول ومُستعيناً بمعلومات الرحالة والتجار الذين جابوا القارات.",
     "gold_keyphrases": ["الإدريسي", "نزهة المشتاق", "خريطة العالم", "الأقاليم", "الجغرافيا الإسلامية"]},

    {"id": 34, "type": "Classical", "topic": "Theology",
     "paragraph": "ناقش الغزالي في إحياء علوم الدين إشكالية التوفيق بين الفقه الظاهري والتجربة الروحية الصوفية، داعياً إلى تجديد العلوم الإسلامية بعد أن رأى في أهل زمانه غلبةً للشكل على الروح والمظهر على الجوهر.",
     "gold_keyphrases": ["الغزالي", "إحياء علوم الدين", "التصوف", "الفقه الإسلامي", "تجديد العلوم"]},

    {"id": 35, "type": "Classical", "topic": "Philosophy",
     "paragraph": "ميّز الفارابي في آراء أهل المدينة الفاضلة بين الرئيس الأول للمدينة الذي يجمع بين الحكمة الفلسفية والنبوة، وبين الرئيس الثاني الذي يُحكِم السنن والقوانين التي وضعها الأول ويُديرها بالعقل.",
     "gold_keyphrases": ["الفارابي", "المدينة الفاضلة", "الفلسفة السياسية", "النبوة", "الرئيس الأول"]},

    {"id": 36, "type": "Classical", "topic": "Grammar",
     "paragraph": "أرسى سيبويه في كتابه المعروف بالكتاب الأسس الأولى لعلم النحو العربي، مُقرِّراً قواعد الإعراب والبناء والعوامل النحوية، ومستشهداً بشعر العرب الفصيح وأقوالهم لتسويغ ما قرر من أحكام.",
     "gold_keyphrases": ["سيبويه", "الكتاب", "النحو العربي", "الإعراب", "العوامل النحوية"]},

    {"id": 37, "type": "Classical", "topic": "Astronomy",
     "paragraph": "قدّم البتّاني تصحيحات دقيقة لقيم بطليموس الفلكية، واستطاع قياس الميل المحوري للأرض بدقة فائقة، كما ضبط حسابات السنة الشمسية والقمرية وأثّر تأثيراً بالغاً في علم الفلك الأوروبي لاحقاً.",
     "gold_keyphrases": ["البتّاني", "علم الفلك", "الميل المحوري", "السنة الشمسية", "بطليموس"]},

    {"id": 38, "type": "Classical", "topic": "Mathematics",
     "paragraph": "أسهم الخوارزمي إسهاماً جوهرياً في تطوير علم الجبر عبر كتابه المختصر في حساب الجبر والمقابلة، الذي أرسى مفهوم المعادلات الخطية والتربيعية وحلولها بأسلوب منهجي واضح لم يُسبق إليه.",
     "gold_keyphrases": ["الخوارزمي", "الجبر", "المعادلات", "الجبر والمقابلة", "الرياضيات الإسلامية"]},

    {"id": 39, "type": "Classical", "topic": "Literature",
     "paragraph": "يُعدّ المتنبي من أعظم شعراء العربية على الإطلاق، وتتميز قصائده بعمق الحكمة وفخامة الأسلوب والصور الشعرية الجامعة بين الجزالة والرقة، وقد مدح سيف الدولة الحمداني بأروع ما قيل في المدح.",
     "gold_keyphrases": ["المتنبي", "الشعر العربي", "سيف الدولة", "الحكمة الشعرية", "المديح"]},

    {"id": 40, "type": "Classical", "topic": "Ethics",
     "paragraph": "حدّد ابن مسكويه في تهذيب الأخلاق غاية التربية الأخلاقية في تحقيق السعادة التي هي كمال النفس الإنسانية، مُميِّزاً بين الفضائل العملية المكتسبة بالتدريب والفضائل النظرية المكتسبة بالتأمل والمعرفة.",
     "gold_keyphrases": ["ابن مسكويه", "تهذيب الأخلاق", "السعادة", "الفضائل", "التربية الأخلاقية"]},

    {"id": 41, "type": "Classical", "topic": "Politics",
     "paragraph": "بيّن ابن تيمية في السياسة الشرعية أن العدل ركيزة الملك وأن الظلم يُعجّل بزواله، مستدلاً على ذلك بشواهد تاريخية وقرآنية، ومُفرِّقاً بين السياسة المشروعة المستندة للشريعة وتلك المبنية على الهوى.",
     "gold_keyphrases": ["ابن تيمية", "السياسة الشرعية", "العدل", "الشريعة الإسلامية", "الحكم"]},

    {"id": 42, "type": "Classical", "topic": "Chemistry",
     "paragraph": "اشتُهر جابر بن حيان بتجاربه في علم الكيمياء حيث وصف عمليات التقطير والتبخر والترسيب والتكليس، وأضاف إلى المعجم العلمي مصطلحات لا تزال متداولة في الكيمياء الحديثة حتى اليوم.",
     "gold_keyphrases": ["جابر بن حيان", "الكيمياء", "التقطير", "التجارب العلمية", "المصطلحات الكيميائية"]},

    {"id": 43, "type": "Classical", "topic": "Education",
     "paragraph": "دعا ابن خلدون في المقدمة إلى أن يكون التعليم تدريجياً من البسيط إلى المركب، وأن يراعي المُعلِّم استعداد المتعلم ونضجه العقلي، منتقداً الأساليب القائمة على الحفظ دون الفهم.",
     "gold_keyphrases": ["ابن خلدون", "المقدمة", "التعليم التدريجي", "الحفظ والفهم", "المناهج التعليمية"]},

    {"id": 44, "type": "Classical", "topic": "Quran",
     "paragraph": "ذكر الزمخشري في تفسيره الكشاف أن وجوه الإعجاز في القرآن الكريم ترجع في جوهرها إلى النظم البلاغي الفريد الجامع بين الإيجاز والبيان والتصوير الجمالي الذي يعجز عن محاكاته الإنس والجن.",
     "gold_keyphrases": ["الزمخشري", "الكشاف", "إعجاز القرآن", "البلاغة", "التفسير"]},

    {"id": 45, "type": "Classical", "topic": "Hadith",
     "paragraph": "بيّن الإمام البخاري في مقدمة صحيحه منهجه الصارم في اشتراط اللقاء بين الراوي ومن روى عنه، مُقرِّراً أن الحديث لا يُقبل إلا إذا ثبت سماع الراوي ممن فوقه في السلسلة بشكل موثوق.",
     "gold_keyphrases": ["الإمام البخاري", "صحيح البخاري", "علم الحديث", "الإسناد", "الراوي"]},

    {"id": 46, "type": "Classical", "topic": "Biography",
     "paragraph": "أفرد ابن كثير في البداية والنهاية فصولاً مطوّلة لسيرة الرسول محمد صلى الله عليه وسلم، مُوثِّقاً غزواته ومواقفه بأسانيد دقيقة ومُتحرِّياً الصحة التاريخية بمنهج ناقد يجمع بين الرواية والدراية.",
     "gold_keyphrases": ["ابن كثير", "البداية والنهاية", "السيرة النبوية", "الغزوات", "المنهج التاريخي"]},

    {"id": 47, "type": "Classical", "topic": "Sufism",
     "paragraph": "عبّر ابن عربي في الفتوحات المكية عن مفهوم وحدة الوجود بلغة رمزية صوفية كثيفة، موضحاً أن الوجود الحقيقي لله وحده وأن الكون تجلٍّ لصفات الحق في صور متعددة لا يُدرك كُنهها إلا المتحقق.",
     "gold_keyphrases": ["ابن عربي", "الفتوحات المكية", "وحدة الوجود", "التصوف الإسلامي", "الكون"]},

    {"id": 48, "type": "Classical", "topic": "Economy",
     "paragraph": "ناقش ابن خلدون في المقدمة مفهوم قيمة العمل والثروة، مؤكداً أن الرزق لا يأتي من فراغ بل هو ثمرة السعي والعمل الإنساني، ومُحذِّراً من خطر الاكتناز والاحتكار على استقرار المجتمعات.",
     "gold_keyphrases": ["ابن خلدون", "قيمة العمل", "الثروة", "الاحتكار", "الاقتصاد الإسلامي"]},

    {"id": 49, "type": "Classical", "topic": "Logic",
     "paragraph": "شرح ابن رشد في تلخيصاته لمنطق أرسطو القياس الاستدلالي بأسلوب عربي واضح، مُدافعاً عن الفلسفة ومستدلاً على أن العقل هبة إلهية ينبغي توظيفها لاستكشاف الحقائق لا معاداتها.",
     "gold_keyphrases": ["ابن رشد", "منطق أرسطو", "القياس الاستدلالي", "الفلسفة", "العقل"]},

    {"id": 50, "type": "Classical", "topic": "Music",
     "paragraph": "بحث الفارابي في كتاب الموسيقى الكبير في أسس الإيقاع والنغم وأثرهما في النفس، مُرسياً قواعد علمية للنظرية الموسيقية وراسماً حدوداً بين الموسيقى النظرية والعملية بأسلوب رياضي دقيق.",
     "gold_keyphrases": ["الفارابي", "كتاب الموسيقى الكبير", "الإيقاع", "النغم", "النظرية الموسيقية"]},

    {"id": 51, "type": "Classical", "topic": "Optics",
     "paragraph": "أرسى ابن الهيثم في كتاب المناظر أسس البصريات الحديثة، مُدحِّضاً نظرية أفلاطون عن خروج الأشعة من العين ومُثبتاً أن الرؤية تنتج عن دخول الضوء إلى العين من المرئيات المنعكسة عنها.",
     "gold_keyphrases": ["ابن الهيثم", "كتاب المناظر", "البصريات", "الضوء", "نظرية الرؤية"]},

    {"id": 52, "type": "Classical", "topic": "Architecture",
     "paragraph": "تتجلى عبقرية المعمار الإسلامي في الجامع الأموي بدمشق الذي يمزج بين الفن البيزنطي والتقاليد العربية في تناسق بديع، ويتميز بمئذنته العتيقة وفسيفسائه الذهبية التي تصوّر جنة الفردوس.",
     "gold_keyphrases": ["الجامع الأموي", "المعمار الإسلامي", "الفسيفساء", "دمشق", "الفن البيزنطي"]},

    {"id": 53, "type": "Classical", "topic": "Navigation",
     "paragraph": "وثّق ابن ماجد في كتبه أسرار الملاحة في البحر الهندي، مُدوِّناً حسابات النجوم ومسالك الرياح الموسمية ودلائل الملاحة الساحلية، مما جعله مرجعاً للبحارة العرب والبرتغاليين على حد سواء.",
     "gold_keyphrases": ["ابن ماجد", "الملاحة البحرية", "البحر الهندي", "الرياح الموسمية", "النجوم"]},

    {"id": 54, "type": "Classical", "topic": "Grammar",
     "paragraph": "أوضح ابن جني في الخصائص أن اللغة اصطلاح توقيفي أو اجتماعي تطور عبر الزمن، وناقش ظاهرة الاشتقاق الأكبر والأصغر والقلب المكاني، مُرسياً بذلك منهجاً تاريخياً مقارناً في درس اللغة العربية.",
     "gold_keyphrases": ["ابن جني", "الخصائص", "الاشتقاق", "فقه اللغة", "تطور اللغة"]},

    # ── Dialect Arabic — 26 paragraphs ───────────────────────────────────── #

    {"id": 55, "type": "Dialect", "topic": "Egyptian - Daily Life",
     "paragraph": "في مصر الناس بتحب تتجمع في المقاهي وتشرب الشاي وتلعب الطاولة أو الدومينو. ده مش بس تسلية، ده جزء من الهوية المصرية والتواصل الاجتماعي اللي بيربط الناس ببعض.",
     "gold_keyphrases": ["المقاهي المصرية", "الشاي", "الطاولة والدومينو", "الهوية المصرية", "التواصل الاجتماعي"]},

    {"id": 56, "type": "Dialect", "topic": "Egyptian - Technology",
     "paragraph": "الشباب المصري دلوقتي بيشتغل كتير في مجال الفريلانس والتقنية، وفيه ناس بتعمل كليات ودورات أونلاين على اليوتيوب والإنترنت بالعربي عشان يوصلوا لأكبر عدد ممكن.",
     "gold_keyphrases": ["الفريلانس", "التقنية الرقمية", "اليوتيوب", "التعليم الأونلاين", "الشباب المصري"]},

    {"id": 57, "type": "Dialect", "topic": "Egyptian - Food",
     "paragraph": "الكشري والفول والطعمية من أشهر الأكلات المصرية الشعبية اللي بتلاقيها في كل حتة. الأكل الشعبي في مصر مش بس طعام، ده جزء من ثقافة وتراث البلد.",
     "gold_keyphrases": ["الكشري", "الفول والطعمية", "الأكل الشعبي المصري", "التراث الغذائي", "الثقافة المصرية"]},

    {"id": 58, "type": "Dialect", "topic": "Egyptian - Education",
     "paragraph": "التعليم في مصر بيواجه تحديات كتير زي الكثافة في الفصول وضعف التدريب، وكتير من الأسر بتلجأ للدروس الخصوصية كحل بديل، وده عبء اقتصادي كبير على الأسرة المصرية.",
     "gold_keyphrases": ["التعليم في مصر", "الكثافة الدراسية", "الدروس الخصوصية", "تحديات التعليم", "الأسرة المصرية"]},

    {"id": 59, "type": "Dialect", "topic": "Gulf - Economy",
     "paragraph": "في الإمارات والسعودية فيه تحول كبير نحو الاقتصاد غير النفطي، والحكومات تدعم الشركات الناشئة وريادة الأعمال وتعطي تأشيرات وإقامات للمواهب الأجنبية عشان تجذب الكفاءات.",
     "gold_keyphrases": ["الاقتصاد غير النفطي", "ريادة الأعمال", "الشركات الناشئة", "الإمارات والسعودية", "استقطاب الكفاءات"]},

    {"id": 60, "type": "Dialect", "topic": "Gulf - Daily Life",
     "paragraph": "الحياة في الخليج مريحة بسبب الرفاهية وتوافر الخدمات، بس كمان فيه ضغط اجتماعي كبير نابع من تقاليد القبيلة والعائلة والحرص على الظهور بمظهر لائق في المناسبات.",
     "gold_keyphrases": ["الحياة الخليجية", "الرفاهية", "التقاليد القبلية", "الضغط الاجتماعي", "المناسبات"]},

    {"id": 61, "type": "Dialect", "topic": "Gulf - Heritage",
     "paragraph": "الفن الخليجي التراثي زي الصوت والخماري والعازي يعكس روح البحر والبر اللي عاشها أهل المنطقة، وهالفنون صارت جزء من الهوية الخليجية اللي تحاول الحكومات المحافظة عليها.",
     "gold_keyphrases": ["الفن الخليجي", "الصوت والخماري", "التراث الخليجي", "الهوية الخليجية", "فنون البحر"]},

    {"id": 62, "type": "Dialect", "topic": "Gulf - Food",
     "paragraph": "الهريس والمجبوس والرز بالخضار من أشهر الأكلات الخليجية اللي بتنعمل في المناسبات والأعياد. الأكل الخليجي متأثر بالتراث البحري والتجاري والعلاقات مع الهند وشرق أفريقيا.",
     "gold_keyphrases": ["الهريس والمجبوس", "الأكل الخليجي", "التراث البحري", "الأعياد", "التأثير الهندي"]},

    {"id": 63, "type": "Dialect", "topic": "Levantine - Politics",
     "paragraph": "الشعب الفلسطيني واللبناني والسوري عاشوا ظروف سياسية صعبة جداً بسبب الحروب والنزوح، وهالتجارب خلّت عندهم وعي سياسي عالي وقدرة على الصمود والتكيف مع الأزمات.",
     "gold_keyphrases": ["القضية الفلسطينية", "النزوح", "الوعي السياسي", "الصمود", "الأزمات الإقليمية"]},

    {"id": 64, "type": "Dialect", "topic": "Levantine - Food",
     "paragraph": "الكبة والحمص والتبولة والفلافل أكلات شامية بتنحب في كل مكان، وكتير من المطاعم بالعالم صارت تقدمها. الأكل الشامي مش بس لذيذ، هو سفير الثقافة اللبنانية والسورية للعالم.",
     "gold_keyphrases": ["الأكل الشامي", "الحمص والفلافل", "الكبة", "التبولة", "المطبخ اللبناني السوري"]},

    {"id": 65, "type": "Dialect", "topic": "Levantine - Daily Life",
     "paragraph": "في لبنان وسوريا الناس بتحب تتجمع عالسطوح والشرفات، وبيحبوا المناقشات السياسية والفلسفية على القهوة، وهاد الحس الجماعي والنقاش المفتوح جزء أصيل من الثقافة الشامية.",
     "gold_keyphrases": ["الحياة الشامية", "الجلسات الاجتماعية", "النقاش السياسي", "القهوة", "الثقافة اللبنانية السورية"]},

    {"id": 66, "type": "Dialect", "topic": "Levantine - Heritage",
     "paragraph": "مدينة بصرى الأثرية في سوريا وتدمر وجرش الأردنية كنوز تاريخية بتشهد على حضارات متعاقبة، وهالمواقع كانت مقصد السياح قبل الحروب وبتمثل ذاكرة حضارية للشعوب الشامية.",
     "gold_keyphrases": ["بصرى", "تدمر", "جرش", "المواقع الأثرية الشامية", "السياحة التراثية"]},

    {"id": 67, "type": "Dialect", "topic": "Moroccan - Daily Life",
     "paragraph": "في المغرب الحياة اليومية بتمزج بين الحداثة والتراث، والناس بيقضيوا وقتهم في الأسواق الشعبية والمساجد والمقاهي، والمدن العتيقة كيفاس ومراكش فيهم روح مغربية أصيلة ما كتبدلتش.",
     "gold_keyphrases": ["المغرب", "الأسواق الشعبية", "فاس ومراكش", "المدن العتيقة", "الهوية المغربية"]},

    {"id": 68, "type": "Dialect", "topic": "Moroccan - Food",
     "paragraph": "الطاجين والكسكسي والبسطيلة أكلات مغربية مشهورة في العالم كله، والمطبخ المغربي كيتميز بالتوابل والعطور والمزج بين النكهات العربية والأمازيغية والأندلسية في تناسق فريد.",
     "gold_keyphrases": ["الطاجين والكسكسي", "البسطيلة", "المطبخ المغربي", "التوابل", "الموروث الأندلسي"]},

    {"id": 69, "type": "Dialect", "topic": "Moroccan - Education",
     "paragraph": "التعليم في المغرب بيعاني من إشكالية ازدواجية اللغة بين العربية والفرنسية، وكتير من الطلاب يجدوا صعوبة في التكيف مع النظام التعليمي اللي كيتغير كثيراً في السنوات الأخيرة.",
     "gold_keyphrases": ["التعليم في المغرب", "ازدواجية اللغة", "اللغة الفرنسية", "الإصلاح التعليمي", "إشكاليات التعليم"]},

    {"id": 70, "type": "Dialect", "topic": "Sudanese - Daily Life",
     "paragraph": "في السودان الناس عندهم طيبة وكرم معروفين، والضيافة والشاي والقهوة جزء لا يتجزأ من اليوم عندهم. الحياة في الخرطوم بتجمع بين حداثة المدينة وعمق الأصالة السودانية.",
     "gold_keyphrases": ["السودان", "الضيافة السودانية", "الكرم", "الخرطوم", "الهوية السودانية"]},

    {"id": 71, "type": "Dialect", "topic": "Sudanese - Heritage",
     "paragraph": "السودان عنده حضارة نوبية عريقة ومملكة مروي والأهرامات السودانية اللي ما أخذت حقها من الاهتمام الدولي. هالموروث الحضاري بيعكس عمق الهوية السودانية وتنوعها الثقافي والعرقي.",
     "gold_keyphrases": ["الحضارة النوبية", "مروي", "الأهرامات السودانية", "التراث السوداني", "الهوية الثقافية"]},

    {"id": 72, "type": "Dialect", "topic": "Yemeni - Daily Life",
     "paragraph": "اليمنيين مشهورين بكرمهم ومحبتهم للضيف، وجلسة القات بعد الظهر تعتبر طقس اجتماعي ثابت عندهم تنقاش فيها الأمور العامة والسياسة والأسرة بشكل غير رسمي.",
     "gold_keyphrases": ["اليمن", "القات", "الكرم اليمني", "الجلسات الاجتماعية", "العادات اليمنية"]},

    {"id": 73, "type": "Dialect", "topic": "Iraqi - Daily Life",
     "paragraph": "بغداد مدينة التاريخ والحضارة، والعراقيين عندهم فخر كبير بإرثهم الحضاري من بابل وسومر وعصر الخلفاء العباسيين. حتى في الأوقات الصعبة الناس بتحافظ على هويتها وكرمها.",
     "gold_keyphrases": ["بغداد", "الحضارة العراقية", "بابل وسومر", "العصر العباسي", "الهوية العراقية"]},

    {"id": 74, "type": "Dialect", "topic": "Iraqi - Food",
     "paragraph": "التشريب والقيمر والمسگوف والدولمة أكلات عراقية ما تكتمل سفرة العيد بدونها، وكل بيت عراقي عنده طريقته الخاصة في تحضيرها اللي بتنتقل من الأم لبنتها.",
     "gold_keyphrases": ["المسگوف", "الدولمة", "التشريب", "الطبخ العراقي", "التراث الغذائي"]},

    {"id": 75, "type": "Dialect", "topic": "Libyan - Daily Life",
     "paragraph": "في ليبيا الناس عندهم ارتباط قوي بالبادية والبيت الأصيل والضيافة العربية، حتى اللي عاش في المدينة يرجع للجذور ويحتفل بالأعياد والمناسبات بنفس الطقوس القبلية القديمة.",
     "gold_keyphrases": ["ليبيا", "البادية", "الضيافة العربية", "الأعياد", "الهوية القبلية"]},

    {"id": 76, "type": "Dialect", "topic": "Tunisian - Economy",
     "paragraph": "تونس تعتبر من الدول اللي نجحت أكثر في مجال تصدير الخدمات الرقمية والاستعانة بمصادر خارجية، وكتير من شركات أوروبية وأمريكية تعتمد على مراكز التعهيد التونسية لجودتها وكفاءتها.",
     "gold_keyphrases": ["تونس", "التعهيد الرقمي", "تصدير الخدمات", "الشركات الأوروبية", "الاقتصاد الرقمي"]},

    {"id": 77, "type": "Dialect", "topic": "Egyptian - Sports",
     "paragraph": "كورة القدم في مصر مش مجرد رياضة، دي ديانة تانية! لما الأهلي أو الزمالك بيلعبوا، نص مصر بتوقف عن الشغل والشارع بيتحول لملعب. وده الجنون الجميل اللي بيوحد الناس.",
     "gold_keyphrases": ["كورة القدم", "الأهلي", "الزمالك", "الرياضة المصرية", "الهوية الجماهيرية"]},

    {"id": 78, "type": "Dialect", "topic": "Gulf - Technology",
     "paragraph": "في السعودية والإمارات الشباب كثير منهم صاروا رواد أعمال رقميين، يطلقون تطبيقات وشركات ناشئة تستفيد من الانتشار العالي لاستخدام الإنترنت والهاتف في المنطقة.",
     "gold_keyphrases": ["ريادة الأعمال", "الشركات الناشئة", "التطبيقات الرقمية", "الشباب الخليجي", "الاقتصاد الرقمي"]},

    {"id": 79, "type": "Dialect", "topic": "Levantine - Humor",
     "paragraph": "السوري والأردني واللبناني عندهم حس فكاهة مشترك، بيحبوا يسخروا من أوضاعهم حتى لو كانت صعبة. هاد الضحك على الحزن هو شكل من أشكال المقاومة والتحدي اللي بيعيشوا فيه.",
     "gold_keyphrases": ["الفكاهة العربية", "السخرية الاجتماعية", "المقاومة", "الهوية الشامية", "التحدي"]},

    {"id": 80, "type": "Dialect", "topic": "Moroccan - Heritage",
     "paragraph": "مدينة فاس المغربية واحدة من أقدم المدن العربية وفيها مدينة عتيقة محفوظة بشكل مذهل، وزقاقاتها الضيقة والحرفيين اللي كيشتغلوا فيها بطرق تقليدية بتخليها متحفاً حياً للتاريخ.",
     "gold_keyphrases": ["فاس", "المدينة العتيقة", "الحرف التقليدية", "التراث المغربي", "المتحف الحي"]},
]


# =============================================================================
# BUILD DATASET
# =============================================================================

def build_dataset(paragraphs: list) -> list:
    dataset = []
    for entry in paragraphs:
        gold   = entry["gold_keyphrases"]
        variety = entry["type"]

        annotations = simulate_annotations(gold, variety)
        voted_gold  = majority_vote(annotations)

        # Keep the stored gold aligned with the majority-vote result.
        if not voted_gold:
            voted_gold = gold[:]

        record = {
            "id":                 entry["id"],
            "type":               variety,
            "topic":              entry["topic"],
            "paragraph":          entry["paragraph"],
            "source":             entry.get("source", ""),
            "source_detail":      entry.get("source_detail", ""),
            "annotations":        annotations,
            "gold_keyphrases":    voted_gold,
            "num_gold_keyphrases": len(voted_gold),
        }
        dataset.append(record)
    return dataset


def print_statistics(dataset: list):
    from collections import Counter
    variety_counts = Counter(d["type"] for d in dataset)
    topics         = Counter(d["topic"] for d in dataset)
    avg_keys       = sum(d["num_gold_keyphrases"] for d in dataset) / len(dataset)

    print("=" * 60)
    print("  Dataset Statistics — Task 10: Keyphrase Extraction")
    print("=" * 60)
    print(f"  Total paragraphs           : {len(dataset)}")
    for v in ["MSA", "Classical", "Dialect"]:
        print(f"  {v:<26}: {variety_counts[v]}")
    print(f"  Avg gold keyphrases/para   : {avg_keys:.2f}")
    print(f"  Topics covered             : {len(topics)}")
    print("=" * 60)


def save_dataset(dataset: list, path: str = "dataset.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"  [✓] Saved {len(dataset)} entries → {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\nBuilding dataset from real Arabic sources with annotation simulation...")
    real_entries = build_real_source_entries()
    real_entries = attach_gold_keyphrases(real_entries)
    dataset = build_dataset(real_entries)
    print_statistics(dataset)
    save_dataset(dataset)

    # Quick sanity check — show annotation variation for first 3 paragraphs
    print("\nSample annotation variation (first 3 paragraphs):")
    for item in dataset[:3]:
        print(f"\n  ID={item['id']}  [{item['type']}]  {item['topic']}")
        for i, ann in enumerate(item["annotations"], 1):
            print(f"    Ann{i}: {ann}")
        print(f"    Gold (majority vote): {item['gold_keyphrases']}")