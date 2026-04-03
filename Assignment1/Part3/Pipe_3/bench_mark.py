r"""
=============================================================================
PIPELINE 3 — STEP 1: LLM Benchmarking  (v7)
=============================================================================
Changes from v6:
  - Fixed Gemini model ID → google/gemini-2.5-flash-preview
  - MAX_OUTPUT_TOKENS_DEFAULT increased to 10 (fixes null responses)
  - QUICK_TEST_MODE = True by default (test 1 image before full run)
  - Paid models: gemini-2.5-flash, gpt-4o-mini
  - Free models: llama-3.2-11b-vision, qwen3-vl-8b, qwen3-vl-8b-thinking
  - Strong prompts with full visual guide for all 10 digit styles
  - Few-shot with real GT images (all 10 digits)
=============================================================================
HOW TO USE:
  1. Run with QUICK_TEST_MODE = True  → tests 1 image, checks all models work
  2. If all models return digits (not null/?) → set QUICK_TEST_MODE = False
  3. Run again for full 500-image benchmark
=============================================================================
"""

import os, re, json, time, base64, io, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from openai import OpenAI
from PIL import Image, ImageEnhance

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

API_KEY   = os.getenv("OPENROUTER_API_KEY", "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")

DATA_DIR   = Path(r"C:\Neural\Indian_Digits_Train")
OUTPUT_DIR = Path(r"C:\Neural\Pipeline3_Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUICK_TEST_MODE        = True    # ← set False after confirming all models work
QUICK_TEST_IMAGE_INDEX = 1
QUICK_TEST_FREE_ONLY   = False   # True = only test free models in quick test

USE_FEWSHOT    = True
FEWSHOT_DIGITS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)   # all 10 classes

GT_FILE_CANDIDATES = [
    OUTPUT_DIR / "ground_truth_labels.npy",
    DATA_DIR   / "ground_truth_labels.npy",
    Path(r"C:\Neural\Neural-Assignments\Assignment1\Part3\Pipe_3\ground_truth_labels.npy"),
]
EVAL_INDICES = None

# ─────────────────────────────────────────────────────────────────────────────
# MODELS  — PAID first, FREE after
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    # ── PAID (run first) ─────────────────────────────────────────────────────
    "gemini-2.5-flash"     : "google/gemini-2.5-flash-preview",
    "gpt-4o-mini"          : "openai/gpt-4o-mini",
    # ── FREE (run after) ─────────────────────────────────────────────────────
    "llama-3.2-11b-vision" : "meta-llama/llama-3.2-11b-vision-instruct",
    "qwen3-vl-8b"          : "qwen/qwen3-vl-8b-instruct",
    "qwen3-vl-8b-thinking" : "qwen/qwen3-vl-8b-thinking",
}

FREE_MODELS     = {"llama-3.2-11b-vision", "qwen3-vl-8b", "qwen3-vl-8b-thinking"}
THINKING_MODELS = {"qwen3-vl-8b-thinking"}

MAX_OUTPUT_TOKENS_DEFAULT  = 10   # increased from 2 — fixes null responses
MAX_OUTPUT_TOKENS_THINKING = 200  # thinking models need room for chain-of-thought

DELAY_BETWEEN_IMAGES = 0.25
DELAY_BETWEEN_MODELS = 1.5
MAX_RETRIES          = 2
RETRY_WAIT_BASE      = 4.0
EARLY_ABORT_4XX_STREAK = 5

IMG_SIZE = 224

# ─────────────────────────────────────────────────────────────────────────────
# COST TABLE  (USD per 1,000 images)
# ─────────────────────────────────────────────────────────────────────────────

COST_PER_1K_IMAGES = {
    "gemini-2.5-flash"     : 0.10,
    "gpt-4o-mini"          : 0.60,
    "llama-3.2-11b-vision" : 0.00,
    "qwen3-vl-8b"          : 0.00,
    "qwen3-vl-8b-thinking" : 0.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING  — LANCZOS + high contrast + sharpen
# ─────────────────────────────────────────────────────────────────────────────

def img_to_b64(path: Path, size: int = IMG_SIZE) -> str:
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = ImageEnhance.Sharpness(img).enhance(3.0)
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────────────────────────────────────
# STRONG PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert OCR system specializing in handwritten digit recognition \
across multiple numeral systems. You have deep knowledge of:
  - Western Arabic numerals (0-9) as written in South Asia and the Middle East
  - Eastern Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩) used in Arabic-speaking countries
  - Devanagari numerals (०१२३४५६७८९) used in Hindi/Sanskrit writing
  - Bengali/Assamese numerals (০১২৩৪৫৬৭৮৯)

Your task: look at the handwritten digit image and return EXACTLY ONE ASCII \
digit character (0-9). No words. No explanation. No punctuation. Just one digit."""

USER_PROMPT = """\
TASK: Identify the handwritten digit in this image.

VISUAL GUIDE — how each digit typically looks in Indian handwriting styles:

  0 → A closed oval or circle. May be slightly tilted or squashed.
      In Eastern Arabic it looks like a dot (·) or small oval (٠).

  1 → A vertical stroke, sometimes with a small flag or serif at the top.
      May lean left or right. In Devanagari it can look like a vertical bar.

  2 → Starts with a curve at the top (like a swan neck), then a flat base.
      The base may be a horizontal line or a looping curve.

  3 → Two bumps stacked vertically on the right side.
      The top bump is usually smaller. Open on the left.
      In Eastern Arabic (٣) it may look like a reversed epsilon (ε).

  4 → A sharp angular shape. Usually has a vertical stroke on the right
      and a crossbar. The top may be open or closed.

  5 → Flat top, curved belly to the right, and a small hat or flag at top.
      In Devanagari (५) it looks like a rounded shape with a top hook.

  6 → A curved stroke descending into a closed loop at the bottom.
      The loop is at the bottom-right. The top is open and curves left.

  7 → A horizontal stroke at the top with a diagonal stroke going down-right.
      May have a small crossbar through the middle.

  8 → Two stacked loops (top smaller, bottom larger). Looks like two circles.
      The crossing point is in the middle.

  9 → A closed loop at the top with a tail descending down (or curling).
      Mirror image of 6 in structure.

IMPORTANT NOTES:
  - The dataset mixes Western Arabic, Eastern Arabic, and Devanagari styles.
  - Strokes may be thick, thin, rotated, or noisy — focus on overall shape.
  - If unsure between two digits, pick the one whose LOOP STRUCTURE matches best.
  - Real example images with correct labels are shown above for reference.

Return ONLY one character: 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9."""

FEWSHOT_INTRO = """\
Below are REAL examples from this exact dataset with their correct labels.
Study the stroke style carefully — these are the same visual style you must classify."""

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT BUILDER  — uses real GT images, all 10 digits
# ─────────────────────────────────────────────────────────────────────────────

def build_fewshot_from_gt(gt_dict: dict, digits=FEWSHOT_DIGITS) -> list:
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in gt_dict.items():
        label_to_indices[lbl].append(idx)
    for lbl in label_to_indices:
        label_to_indices[lbl].sort()

    blocks = []
    found  = []
    for digit in digits:
        candidates = label_to_indices.get(digit, [])
        if not candidates:
            continue
        img_path = DATA_DIR / f"{candidates[0]}.bmp"
        if not img_path.exists():
            continue
        b64 = img_to_b64(img_path)
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
        blocks.append({
            "type": "text",
            "text": f"Correct answer for above image: {digit}"
        })
        found.append(digit)

    if found:
        print(f"  [OK] Few-shot: {len(found)} real GT examples — digits {found}")
    else:
        print("  [WARN] No GT images found for few-shot.")
    return blocks

# ─────────────────────────────────────────────────────────────────────────────
# UNICODE NORMALISER
# ─────────────────────────────────────────────────────────────────────────────

_UMAP = {
    "٠":"0","١":"1","٢":"2","٣":"3","٤":"4","٥":"5","٦":"6","٧":"7","٨":"8","٩":"9",
    "۰":"0","۱":"1","۲":"2","۳":"3","۴":"4","۵":"5","۶":"6","۷":"7","۸":"8","۹":"9",
    "०":"0","१":"1","२":"2","३":"3","४":"4","५":"5","६":"6","७":"7","८":"8","९":"9",
    "০":"0","১":"1","২":"2","৩":"3","৪":"4","৫":"5","৬":"6","৭":"7","৮":"8","৯":"9",
}

def normalize(text, is_thinking=False) -> str | None:
    if text is None:
        return None
    if isinstance(text, list):
        text = " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in text
        )
    text = str(text).strip()
    if not text or text.lower() in ("null", "none", ""):
        return None
    converted = "".join(_UMAP.get(c, c) for c in text)
    digits = re.findall(r"[0-9]", converted)
    if not digits:
        return None
    # thinking models: take last digit (after chain of thought ends)
    # normal models: take first digit
    return digits[-1] if is_thinking else digits[0]

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict(client, model_name, model_id, img_b64, fewshot_blocks):
    is_thinking = model_name in THINKING_MODELS
    max_tok     = MAX_OUTPUT_TOKENS_THINKING if is_thinking else MAX_OUTPUT_TOKENS_DEFAULT
    max_att     = 1 if QUICK_TEST_MODE else MAX_RETRIES

    # message order:
    # [few-shot intro + examples] → [visual guide + task] → [target image]
    user_content = []
    if fewshot_blocks:
        user_content.append({"type": "text", "text": FEWSHOT_INTRO})
        user_content.extend(fewshot_blocks)

    user_content.append({"type": "text", "text": USER_PROMPT})
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
    })

    for attempt in range(max_att):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=max_tok,
                temperature=0,
            )
            choices = getattr(resp, "choices", None)
            if not choices:
                return None, "ERROR:empty choices"

            message = getattr(choices[0], "message", None)
            if message is None:
                return None, "ERROR:no message"

            raw = getattr(message, "content", "")
            norm = normalize(raw, is_thinking=is_thinking)
            raw_text = (raw if isinstance(raw, str)
                        else json.dumps(raw, ensure_ascii=False))
            return norm, str(raw_text).strip()

        except Exception as e:
            wait = RETRY_WAIT_BASE * (attempt + 1)
            if attempt < max_att - 1:
                print(f"        [WARN] attempt {attempt+1} failed "
                      f"({str(e)[:60]}) — retry in {wait:.0f}s")
                time.sleep(wait)
            else:
                return None, f"ERROR:{str(e)}"

def is_hard_4xx(raw_text: str) -> bool:
    if not raw_text:
        return False
    m = re.search(r"^ERROR:Error code:\s*(\d{3})\b", str(raw_text))
    return bool(m) and 400 <= int(m.group(1)) < 500

# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth():
    gt_file = next((p for p in GT_FILE_CANDIDATES if Path(p).exists()), None)
    if gt_file is None:
        print("[ERROR] ground_truth_labels.npy not found. Searched:")
        for p in GT_FILE_CANDIDATES:
            print(f"    {p}")
        sys.exit(1)

    labels = np.load(gt_file).flatten().astype(int)
    print(f"[OK] Ground truth: {gt_file}  shape={labels.shape}")
    n = len(labels)

    if n == 10_000:
        if EVAL_INDICES is not None:
            eval_idx = list(EVAL_INDICES)
        else:
            import random
            random.seed(42)
            by_class = defaultdict(list)
            for i, lbl in enumerate(labels):
                by_class[int(lbl)].append(i + 1)
            eval_idx = []
            for cls in range(10):
                pool = by_class[cls]
                eval_idx.extend(random.sample(pool, min(50, len(pool))))
            eval_idx.sort()
        gt_dict = {idx: int(labels[idx - 1]) for idx in eval_idx}

    elif n == 500:
        eval_idx = list(EVAL_INDICES) if EVAL_INDICES else list(range(1, 501))
        gt_dict  = {idx: int(labels[i]) for i, idx in enumerate(eval_idx)}
    else:
        print(f"[ERROR] Unexpected .npy length {n}."); sys.exit(1)

    print(f"    Eval set: {len(eval_idx)} images")
    return eval_idx, gt_dict

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_path(name):
    return OUTPUT_DIR / f"ckpt_{name.replace('/','_').replace(':','_')}.json"

def load_ckpt(name):
    p = _ckpt_path(name)
    return json.load(open(p)) if p.exists() else {}

def save_ckpt(name, done):
    with open(_ckpt_path(name), "w") as f:
        json.dump(done, f)

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK ONE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_model(client, name, model_id, eval_idx, gt_dict, fewshot_blocks):
    typ = "FREE" if name in FREE_MODELS else "PAID"
    print(f"\n{'='*65}")
    print(f"  MODEL : {name}  [{typ}]")
    print(f"  ID    : {model_id}")
    if name in THINKING_MODELS:
        print(f"  NOTE  : thinking model — max_tokens={MAX_OUTPUT_TOKENS_THINKING}")
    print(f"{'='*65}")

    done       = load_ckpt(name)
    preds      = {}
    raws       = {}
    streak_4xx = 0

    if done:
        print(f"  [RESUME] {len(done)}/{len(eval_idx)} already done")

    for i, idx in enumerate(eval_idx):
        key = str(idx)
        if key in done:
            preds[idx] = done[key]["pred"]
            raws[idx]  = done[key]["raw"]
            continue

        img_path = DATA_DIR / f"{idx}.bmp"
        if not img_path.exists():
            preds[idx] = None
            raws[idx]  = "FILE_NOT_FOUND"
            done[key]  = {"pred": None, "raw": "FILE_NOT_FOUND"}
            save_ckpt(name, done)
            continue

        pred, raw = predict(
            client, name, model_id, img_to_b64(img_path), fewshot_blocks
        )
        preds[idx] = pred
        raws[idx]  = raw
        done[key]  = {"pred": pred, "raw": raw}

        if is_hard_4xx(raw):
            streak_4xx += 1
            if streak_4xx >= EARLY_ABORT_4XX_STREAK:
                save_ckpt(name, done)
                print(f"  [ABORT] {streak_4xx} consecutive 4xx — skipping model.")
                return {
                    "model_name" : name,
                    "model_id"   : model_id,
                    "skipped"    : True,
                    "skip_reason": f"{streak_4xx} consecutive 4xx errors",
                }
        else:
            streak_4xx = 0

        gt = gt_dict.get(idx)
        ok = ("OK" if (pred and gt is not None and int(pred) == gt)
              else ("?" if pred is None else "X"))
        print(f"  [{i+1:4d}/{len(eval_idx)}]  img={idx:5d}  "
              f"pred={pred or '?'}  gt={gt if gt is not None else '-'}  {ok}  "
              f"raw='{str(raw)[:25]}'")

        if (i + 1) % 50 == 0:
            save_ckpt(name, done)
            print(f"  [SAVE] checkpoint at {i+1}")

        time.sleep(DELAY_BETWEEN_IMAGES)

    save_ckpt(name, done)

    correct = total = null_ct = 0
    pc_c = defaultdict(int)
    pc_t = defaultdict(int)
    errors = []

    for idx in eval_idx:
        gt = gt_dict.get(idx)
        if gt is None:
            continue
        pred = preds.get(idx)
        pc_t[gt] += 1
        total    += 1
        if pred is None:
            null_ct += 1
            errors.append((idx, gt, None))
        elif int(pred) == gt:
            correct += 1
            pc_c[gt] += 1
        else:
            errors.append((idx, gt, pred))

    acc  = correct / total * 100 if total else 0.0
    null = null_ct / total * 100 if total else 0.0

    print(f"\n  ── {name} RESULTS ──")
    print(f"  Accuracy  : {acc:.2f}%  ({correct}/{total})")
    print(f"  Null rate : {null:.2f}%  ({null_ct} unparseable)")
    print(f"  Per-digit accuracy:")
    for d in range(10):
        t   = pc_t[d]
        c   = pc_c[d]
        pct = c / max(t, 1) * 100
        bar = "█" * int(pct / 5)
        print(f"    {d}: {c:3d}/{t:3d}  {bar:<20}  {pct:.0f}%")
    if errors:
        print(f"  First 5 errors:")
        for idx, gt, pred in errors[:5]:
            print(f"    img={idx}  gt={gt}  pred={pred}  "
                  f"raw='{str(raws.get(idx,''))[:30]}'")

    return {
        "model_name"        : name,
        "model_id"          : model_id,
        "accuracy"          : round(acc,  4),
        "null_rate"         : round(null, 4),
        "correct"           : correct,
        "total"             : total,
        "null_count"        : null_ct,
        "per_class_correct" : dict(pc_c),
        "per_class_total"   : dict(pc_t),
        "predictions"       : {str(k): v for k, v in preds.items()},
        "raw_outputs"       : {str(k): v for k, v in raws.items()},
    }

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(all_results):
    ranked = sorted(
        [(k, v) for k, v in all_results.items() if not v.get("skipped")],
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )

    lines = [
        "=" * 70,
        "  PIPELINE 3 - STEP 1: MODEL RANKING  (v7)",
        "=" * 70,
        f"  {'Rank':<5} {'Model':<26} {'Accuracy':>10} {'Null%':>7} "
        f"{'Type':>6}  {'Cost/10k':>10}",
        "  " + "-" * 65,
    ]

    for rank, (name, res) in enumerate(ranked, 1):
        typ  = "FREE" if name in FREE_MODELS else "PAID"
        cost = COST_PER_1K_IMAGES.get(name, 0.0) * 10
        sel  = " ← TOP" if rank <= 2 else ""
        lines.append(
            f"  {rank:<5} {name:<26} {res['accuracy']:>9.2f}%"
            f" {res['null_rate']:>6.2f}%  {typ:>6}  ${cost:>8.2f}{sel}"
        )

    lines += ["=" * 70, "", "  SELECTED FOR STEP 3 (top 2):"]
    for rank, (name, res) in enumerate(ranked[:2], 1):
        lines.append(f"    LLM-{rank}: {name}  ({res['accuracy']:.2f}%)")

    if len(ranked) >= 2:
        n1, n2 = ranked[0][0], ranked[1][0]
        c1 = COST_PER_1K_IMAGES.get(n1, 0.0) * 10
        c2 = COST_PER_1K_IMAGES.get(n2, 0.0) * 10
        lines += [
            "",
            "  ── STEP 3 API COST ESTIMATE (10,000 images × 2 LLMs) ──",
            f"    {n1:<30}  ${c1:.2f}",
            f"    {n2:<30}  ${c2:.2f}",
            f"    {'Total':<30}  ${c1+c2:.2f}",
            "    (Estimates from OpenRouter pricing — update if changed)",
        ]

    a1 = ranked[0][1]["accuracy"] if ranked else 0
    a2 = ranked[1][1]["accuracy"] if len(ranked) > 1 else 0
    if a1 >= 90 and a2 >= 90:
        verdict = "[OK]  Both >= 90% — proceed to Step 3."
    elif a1 >= 80:
        verdict = "[WARN] Top model 80-90% — consider prompt refinement."
    else:
        verdict = "[ERROR] < 80% — refine prompt or use stronger models."
    lines += ["", f"  {verdict}", "=" * 70]

    txt = "\n".join(lines)
    print("\n\n" + txt)
    (OUTPUT_DIR / "pipeline3_step1_summary.txt").write_text(txt, encoding="utf-8")

    sel = {}
    if len(ranked) >= 1:
        sel["llm1"] = {
            "name"     : ranked[0][0],
            "model_id" : ranked[0][1]["model_id"],
            "accuracy" : ranked[0][1]["accuracy"],
        }
    if len(ranked) >= 2:
        sel["llm2"] = {
            "name"     : ranked[1][0],
            "model_id" : ranked[1][1]["model_id"],
            "accuracy" : ranked[1][1]["accuracy"],
        }

    (OUTPUT_DIR / "pipeline3_step2_selection.json").write_text(
        json.dumps(sel, indent=2)
    )
    print(f"\n  [FILE] Summary   → {OUTPUT_DIR / 'pipeline3_step1_summary.txt'}")
    print(f"  [FILE] Selection → {OUTPUT_DIR / 'pipeline3_step2_selection.json'}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    paid_list  = [k for k in MODELS if k not in FREE_MODELS]
    free_list  = [k for k in MODELS if k in FREE_MODELS]

    print("\n" + "=" * 65)
    print("  PIPELINE 3 — STEP 1  (v7 fixed + strong-prompt + few-shot)")
    print(f"  PAID  models : {paid_list}")
    print(f"  FREE  models : {free_list}")
    fewshot_str = (f"enabled ({len(FEWSHOT_DIGITS)} digits)"
                   if USE_FEWSHOT else "disabled")
    print(f"  Few-shot     : {fewshot_str}")
    print(f"  Mode         : {'QUICK TEST (1 image)' if QUICK_TEST_MODE else 'FULL BENCHMARK (500 images)'}")
    print("=" * 65)

    client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

    # ── QUICK TEST ─────────────────────────────────────────────────────────────
    if QUICK_TEST_MODE:
        idx      = int(QUICK_TEST_IMAGE_INDEX)
        img_path = DATA_DIR / f"{idx}.bmp"
        if not img_path.exists():
            print(f"[ERROR] Image not found: {img_path}")
            sys.exit(1)

        print(f"\n  QUICK TEST — image {idx}.bmp")
        print(f"  (check each model returns a digit, not '?' or 'null')\n")

        if USE_FEWSHOT:
            eval_idx, gt_dict = load_ground_truth()
            fewshot_blocks    = build_fewshot_from_gt(gt_dict)
        else:
            fewshot_blocks = []

        img_b64 = img_to_b64(img_path)
        all_ok  = True

        for name, model_id in MODELS.items():
            if QUICK_TEST_FREE_ONLY and name not in FREE_MODELS:
                continue
            pred, raw = predict(client, name, model_id, img_b64, fewshot_blocks)
            typ    = "FREE" if name in FREE_MODELS else "PAID"
            think  = " [thinking]" if name in THINKING_MODELS else ""
            status = "[OK]  " if pred is not None else "[WARN]"
            if pred is None:
                all_ok = False
            print(f"  {status} [{typ}]{think} {name:<28} → {pred or '?'}"
                  f"   raw='{str(raw)[:50]}'")
            time.sleep(0.3)

        print()
        if all_ok:
            print("  ✓ All models returned digits.")
            print("  → Set QUICK_TEST_MODE = False and run again for full benchmark.")
        else:
            print("  ⚠  Some models returned '?' — check model IDs or increase MAX_OUTPUT_TOKENS_DEFAULT.")
        print()
        return

    # ── FULL BENCHMARK ─────────────────────────────────────────────────────────
    eval_idx, gt_dict = load_ground_truth()
    n = len(eval_idx)

    fewshot_blocks = build_fewshot_from_gt(gt_dict) if USE_FEWSHOT else []

    print(f"\n  Images      : {n}")
    print(f"  Few-shot    : {len(fewshot_blocks)//2} examples per query")
    est_min = len(MODELS) * n * DELAY_BETWEEN_IMAGES / 60
    print(f"  Est. time   : ~{est_min:.0f} min total")
    for name in MODELS:
        cost = COST_PER_1K_IMAGES.get(name, 0.0) * (n / 1000)
        typ  = "FREE" if name in FREE_MODELS else "PAID"
        print(f"  [{typ}] {name:<26}  est. cost: ${cost:.2f}")
    print()

    results_path = OUTPUT_DIR / "pipeline3_step1_results.json"
    all_results  = {}
    if results_path.exists():
        try:
            all_results = json.loads(results_path.read_text(encoding="utf-8"))
            print(f"  [RESUME] Existing results: {list(all_results.keys())}")
        except Exception:
            print("  [WARN] Could not load existing results — starting fresh.")

    for name, model_id in MODELS.items():
        if name in all_results:
            print(f"\n  [SKIP] {name} already complete.")
            continue

        res = benchmark_model(
            client, name, model_id, eval_idx, gt_dict, fewshot_blocks
        )

        if res.get("skipped"):
            print(f"  [SKIP] {name}: {res.get('skip_reason')}")
            continue

        all_results[name] = res
        results_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  [SAVE] Results saved → {results_path}")
        time.sleep(DELAY_BETWEEN_MODELS)

    save_summary(all_results)
    print("\n  [OK] Step 1 complete — next: pipeline3_step3_full_labelling.py\n")


if __name__ == "__main__":
    main()