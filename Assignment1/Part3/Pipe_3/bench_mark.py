r"""
=============================================================================
PIPELINE 3 — STEP 1: LLM Benchmarking  (v5 low-token)
=============================================================================
Budget-focused settings:
    - 3 paid models are first in run order.
    - Low-token prompt.
    - Few-shot disabled by default (removes extra image tokens).
    - Low output token cap for all models.
    - Single attempt per image to reduce spend.

Models:
    PAID : gpt-4o-mini, claude-3.5-sonnet, llama-3.2-90b-vision
    FREE : llama-3.2-11b-vision, qwen3-vl-8b, qwen3-vl-8b-thinking

Quick test  : QUICK_TEST_MODE = True   → 1 image, all 6 models
Full run    : QUICK_TEST_MODE = False  → 500 images with checkpointing
=============================================================================
"""

import os, re, json, time, base64, io, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from openai import OpenAI
from PIL import Image, ImageFilter, ImageEnhance

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

API_KEY    = os.getenv(
    "OPENROUTER_API_KEY",
    "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",
)
DATA_DIR   = Path(r"C:\Neural\Indian_Digits_Train")
OUTPUT_DIR = Path(r"C:\Neural\Pipeline3_Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUICK_TEST_MODE        = False
QUICK_TEST_IMAGE_INDEX = 1
QUICK_TEST_FREE_ONLY   = False

# Budget controls
LOW_TOKEN_MODE = True
USE_FEWSHOT    = False

GT_FILE_CANDIDATES = [
    OUTPUT_DIR / "ground_truth_labels.npy",
    DATA_DIR   / "ground_truth_labels.npy",
    Path(r"C:\Neural\Neural-Assignments\Assignment1\Part3\Pipe_3\ground_truth_labels.npy"),
]
EVAL_INDICES = None

# ─────────────────────────────────────────────────────────────────────────────
# MODELS  (paid first, then free)
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    # PAID
    "gpt-4o-mini"          : "openai/gpt-4o-mini",
    "claude-3.5-sonnet"    : "anthropic/claude-3.5-sonnet",
    "llama-3.2-90b-vision" : "meta-llama/llama-3.2-90b-vision-instruct",
    # FREE
    "llama-3.2-11b-vision" : "meta-llama/llama-3.2-11b-vision-instruct",
    "qwen3-vl-8b"          : "qwen/qwen3-vl-8b-instruct",
    "qwen3-vl-8b-thinking" : "qwen/qwen3-vl-8b-thinking",
}

FREE_MODELS    = {"llama-3.2-11b-vision", "qwen3-vl-8b", "qwen3-vl-8b-thinking"}
THINKING_MODELS = {"qwen3-vl-8b-thinking"}
MAX_OUTPUT_TOKENS_DEFAULT  = 2
MAX_OUTPUT_TOKENS_THINKING = 2

# ── Timing ────────────────────────────────────────────────────────────────────
DELAY_BETWEEN_IMAGES = 0.35
DELAY_BETWEEN_MODELS = 1.5
MAX_RETRIES          = 1
RETRY_WAIT_BASE      = 4.0

IMG_SIZE = 224

# ─────────────────────────────────────────────────────────────────────────────
# APPROXIMATE API COST TABLE  (USD per 1,000 images, vision input)
# Used only for the Step-3 cost estimate in the summary.
# Update these figures if OpenRouter pricing changes.
# ─────────────────────────────────────────────────────────────────────────────
COST_PER_1K_IMAGES = {
    "gpt-4o-mini"          : 0.60,   # ~$0.006 / call incl. image token
    "claude-3.5-sonnet"    : 3.00,
    "llama-3.2-90b-vision" : 0.90,
    "llama-3.2-11b-vision" : 0.20,
    "qwen3-vl-8b"          : 0.10,
    "qwen3-vl-8b-thinking" : 0.15,
}

# ─────────────────────────────────────────────────────────────────────────────
# UPSCALING  — M5: LANCZOS + Contrast x2.5 + Sharpen x3.0
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

def pil_to_b64(pil_img: Image.Image, size: int = IMG_SIZE) -> str:
    """Same pipeline but from a PIL image."""
    img = pil_img.convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = ImageEnhance.Sharpness(img).enhance(3.0)
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES  (FIX 2)
# ─────────────────────────────────────────────────────────────────────────────
# v3 used synthetic Arial-font digits as few-shot anchors.  This was wrong:
# clean printed digits anchor the LLM to Western style and can HURT accuracy
# on Indian handwritten images.
#
# v4 strategy: load one confirmed real image per digit class from the
# 500-image ground truth set.  This gives the LLM the actual visual style
# it will encounter.  We pick the FIRST confirmed match for each digit so
# the selection is deterministic and reproducible.
#
# If the ground truth set is not yet loaded when build_fewshot_examples()
# is called, the function returns an empty list and the prompt runs without
# few-shot examples — which is still better than misleading synthetic ones.
# ─────────────────────────────────────────────────────────────────────────────

def build_fewshot_from_gt(gt_dict: dict, digits=(0, 2, 4, 6, 9)) -> list:
    """
    Build few-shot content blocks using real images from the ground truth set.

    gt_dict : {image_index (int) -> label (int)}  — the 500-image GT mapping
    digits  : which digit classes to include as examples (5 keeps token cost low)

    Returns a list of OpenAI-style content blocks ready to embed in the prompt.
    """
    # Build reverse map: label -> sorted list of image indices
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
        blocks.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
        blocks.append({"type": "text", "text": f"Answer: {digit}"})
        found.append(digit)

    if found:
        print(f"  [OK]  Few-shot examples built from real GT images: digits {found}")
    else:
        print("  [WARN] No GT images found for few-shot - running without examples.")
    return blocks

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS  (v4 — real few-shot + Indian-digit-aware shape guide)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "Classify one handwritten Indian digit. Reply with exactly one ASCII digit 0-9."

FEWSHOT_INTRO = """\
Here are reference examples of real handwritten Indian digits with their correct labels.
Study the stroke style, loop shape, and overall form carefully — these images come from \
the same dataset you will be classifying."""

# FIX 3: shape guide is now script-agnostic and Indian-digit aware.
# Indian digit datasets (like this one) contain a mix of Western Arabic,
# Devanagari, and Eastern Arabic written forms.  We describe distinguishing
# structural features rather than assuming a single script convention.
USER_PROMPT = """\
Image has one handwritten digit from Indian styles (Western/Devanagari/Eastern Arabic).
Return ONLY one character from 0-9.
No words, no punctuation, no spaces."""

# ─────────────────────────────────────────────────────────────────────────────
# UNICODE NORMALISER
# ─────────────────────────────────────────────────────────────────────────────

_UMAP = {
    "٠":"0","١":"1","٢":"2","٣":"3","٤":"4","٥":"5","٦":"6","٧":"7","٨":"8","٩":"9",
    "۰":"0","۱":"1","۲":"2","۳":"3","۴":"4","۵":"5","۶":"6","۷":"7","۸":"8","۹":"9",
    "०":"0","१":"1","२":"2","३":"3","४":"4","५":"5","६":"6","७":"7","८":"8","९":"9",
    "০":"0","১":"1","২":"2","৩":"3","৪":"4","৫":"5","৬":"6","৭":"7","৮":"8","৯":"9",
}

def normalize(text) -> str | None:
    """
    Extract a single ASCII digit (0-9) from any model response.
    Handles:
      - Direct single-digit replies  ("7", "٧", "७")
      - Thinking-model responses     ("...therefore the answer is 7")
      - Unexpected verbose replies   ("The digit is 7.")
    For thinking models the last digit found is preferred (it follows the CoT).
    """
    if text is None:
        return None
    if isinstance(text, list):
        text = " ".join(p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in text)
    text = str(text).strip()
    if not text:
        return None
    converted = "".join(_UMAP.get(c, c) for c in text)
    # For thinking models, prefer the LAST digit in the response
    # (after the chain-of-thought concludes); for normal models both
    # first and last are the same since the reply is a single character.
    digits = re.findall(r"[0-9]", converted)
    return digits[-1] if digits else None

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict(client: OpenAI, model_name: str, model_id: str,
            img_b64: str, fewshot_blocks: list) -> tuple[str | None, str]:
    """
    Message structure:
      system : expert OCR role
      user   : [few-shot intro]
               [real GT few-shot: img0, label0, img2, label2, ...]
               [task prompt]
               [target image]

    FIX 1: thinking models get MAX_OUTPUT_TOKENS_THINKING tokens so their
           chain-of-thought is not truncated.  normalize() still extracts
           the final digit from the longer response.
    FIX 4: variable renamed from 'retries' to 'max_attempts' for clarity.
    """
    # Token budget
    max_tok = (MAX_OUTPUT_TOKENS_THINKING
               if model_name in THINKING_MODELS
               else MAX_OUTPUT_TOKENS_DEFAULT)

    max_attempts = 1 if QUICK_TEST_MODE else MAX_RETRIES

    user_content = (
        ([{"type": "text", "text": FEWSHOT_INTRO}] + fewshot_blocks
         if fewshot_blocks else [])
        + [{"type": "text", "text": USER_PROMPT}]
        + [{"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]
    )

    for attempt in range(max_attempts):
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
                return None, "ERROR:Provider returned empty choices"

            message = getattr(choices[0], "message", None)
            if message is None:
                return None, "ERROR:Provider returned no message"

            raw = getattr(message, "content", "")
            normalized = normalize(raw)
            raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            return normalized, str(raw_text).strip()

        except Exception as e:
            wait = RETRY_WAIT_BASE * (attempt + 1)
            if attempt < max_attempts - 1:
                print(f"        [WARN] attempt {attempt+1} failed "
                      f"({str(e)[:60]}) - wait {wait:.0f}s")
                time.sleep(wait)
            else:
                return None, f"ERROR:{str(e)[:80]}"

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
            import random; random.seed(42)
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

def benchmark_model(client, name, model_id, eval_idx, gt_dict,
                    fewshot_blocks) -> dict:
    print(f"\n{'='*65}\n  MODEL : {name}\n  ID    : {model_id}")
    if name in THINKING_MODELS:
        print(f"  NOTE  : thinking model - using {MAX_OUTPUT_TOKENS_THINKING} max_tokens")
    print(f"{'='*65}")

    done  = load_ckpt(name)
    if done:
        print(f"  [RESUME] {len(done)}/{len(eval_idx)} done")
    preds = {}; raws = {}

    for i, idx in enumerate(eval_idx):
        key = str(idx)
        if key in done:
            preds[idx] = done[key]["pred"]
            raws[idx]  = done[key]["raw"]
            continue

        img_path = DATA_DIR / f"{idx}.bmp"
        if not img_path.exists():
            preds[idx] = None; raws[idx] = "FILE_NOT_FOUND"
            done[key]  = {"pred": None, "raw": "FILE_NOT_FOUND"}
            save_ckpt(name, done); continue

        pred, raw = predict(
            client, name, model_id, img_to_b64(img_path), fewshot_blocks
        )
        preds[idx] = pred
        raws[idx] = raw
        done[key] = {"pred": pred, "raw": raw}

        gt = gt_dict.get(idx)
        ok = ("OK" if (pred and gt is not None and int(pred) == gt)
              else ("?" if pred is None else "X"))
        print(f"  [{i+1:4d}/{len(eval_idx)}] img={idx:5d}  "
              f"pred={pred or '?'}  gt={gt if gt is not None else '-'}  {ok}  "
              f"raw='{raw[:20]}'")

        if (i + 1) % 50 == 0:
            save_ckpt(name, done)
            print(f"  [SAVE] checkpoint ({i+1} done)")

        time.sleep(DELAY_BETWEEN_IMAGES)

    save_ckpt(name, done)

    correct = total = null_ct = 0
    pc_c = defaultdict(int); pc_t = defaultdict(int)
    errors = []

    for idx in eval_idx:
        gt   = gt_dict.get(idx)
        if gt is None: continue
        pred = preds.get(idx); pc_t[gt] += 1; total += 1
        if pred is None:
            null_ct += 1; errors.append((idx, gt, None))
        elif int(pred) == gt:
            correct += 1; pc_c[gt] += 1
        else:
            errors.append((idx, gt, pred))

    acc  = correct / total * 100 if total else 0.0
    null = null_ct / total * 100 if total else 0.0
    print(f"\n  -- {name} RESULTS --")
    print(f"  Accuracy : {acc:.2f}%  ({correct}/{total})")
    print(f"  Null rate: {null:.2f}%  ({null_ct} unparseable)")
    print(f"  Per-digit:")
    for d in range(10):
        t = pc_t[d]; c = pc_c[d]
        bar = "#" * int(c / max(t, 1) * 20)
        print(f"    {d}: {c:3d}/{t:3d}  {bar:<20}  {c/max(t,1)*100:.0f}%")
    if errors[:5]:
        print("  Sample errors:")
        for idx, gt, pred in errors[:5]:
            print(f"    img={idx}  gt={gt}  pred={pred}  "
                  f"raw='{raws.get(idx,'')[:25]}'")

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
# SUMMARY  (FIX 5: adds Step-3 API cost estimate)
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(all_results):
    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]["accuracy"], reverse=True)

    lines = ["="*70,
             "  PIPELINE 3 - STEP 1: MODEL RANKING  (v5)",
             "="*70,
             f"  {'Rank':<5} {'Model':<28} {'Accuracy':>10} {'Null%':>7} "
             f"{'Type':>6}  {'Est. cost/10k':>14}",
             "  " + "-"*65]

    for rank, (name, res) in enumerate(ranked, 1):
        typ  = "FREE" if name in FREE_MODELS else "PAID"
        sel  = "TOP-2" if rank <= 2 else ""
        cost = COST_PER_1K_IMAGES.get(name, 0.0) * 10   # cost for 10k images
        lines.append(
            f"  {rank:<5} {name:<28} {res['accuracy']:>9.2f}%"
            f" {res['null_rate']:>6.2f}%  {typ:>6}  ${cost:>12.2f}  {sel}"
        )

    lines += ["="*70, "", "  TOP-2 FOR STEP 3:"]
    for rank, (name, res) in enumerate(ranked[:2], 1):
        lines.append(f"    LLM-{rank}: {name}  ({res['accuracy']:.2f}%)")

    # FIX 5: Step-3 API cost estimate ─────────────────────────────────────────
    if len(ranked) >= 2:
        n1, n2  = ranked[0][0], ranked[1][0]
        c1 = COST_PER_1K_IMAGES.get(n1, 0.0) * 10
        c2 = COST_PER_1K_IMAGES.get(n2, 0.0) * 10
        lines += [
            "",
            "  -- STEP 3 API COST ESTIMATE (10,000 images x 2 LLMs) --",
            f"    LLM-1  {n1:<28}  ${c1:.2f}",
            f"    LLM-2  {n2:<28}  ${c2:.2f}",
            f"    Total  {'combined':<28}  ${c1+c2:.2f}",
            "    (Estimates based on approximate OpenRouter pricing.)",
            "    (Update COST_PER_1K_IMAGES dict if pricing has changed.)",
        ]
    # ──────────────────────────────────────────────────────────────────────────

    a1 = ranked[0][1]["accuracy"] if ranked else 0
    a2 = ranked[1][1]["accuracy"] if len(ranked) > 1 else 0
    msg = ("[OK] Both >= 90% - proceed to Step 3."
           if a1 >= 90 and a2 >= 90
           else "[WARN] 80-90% - consider prompt refinement."
           if a1 >= 80
           else "[ERROR] < 80% - refine prompt or try stronger models.")
    lines += ["", f"  {msg}", "="*70]

    txt = "\n".join(lines)
    print("\n\n" + txt)
    (OUTPUT_DIR / "pipeline3_step1_summary.txt").write_text(txt, encoding="utf-8")

    sel = {
        "llm1": {"name": ranked[0][0], "model_id": ranked[0][1]["model_id"],
                 "accuracy": ranked[0][1]["accuracy"]},
        "llm2": {"name": ranked[1][0], "model_id": ranked[1][1]["model_id"],
                 "accuracy": ranked[1][1]["accuracy"]},
    }
    (OUTPUT_DIR / "pipeline3_step2_selection.json").write_text(
        json.dumps(sel, indent=2))
    print(f"\n  [FILE] Summary  : {OUTPUT_DIR / 'pipeline3_step1_summary.txt'}")
    print(f"  [FILE] Selection: {OUTPUT_DIR / 'pipeline3_step2_selection.json'}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  PIPELINE 3 - STEP 1  (v5 low-token)")
    print(f"  Models: {len(FREE_MODELS)} free + {len(MODELS)-len(FREE_MODELS)} paid")
    print("="*65)

    client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

    # ── QUICK TEST (1 image) ──────────────────────────────────────────────────
    if QUICK_TEST_MODE:
        idx      = int(QUICK_TEST_IMAGE_INDEX)
        img_path = DATA_DIR / f"{idx}.bmp"
        if not img_path.exists():
            print(f"\n[ERROR] Image not found: {img_path}"); sys.exit(1)

        print(f"\n  QUICK TEST - image {idx}  ({img_path})")
        print(f"  Upscaling  : M5 (LANCZOS + Contrast x2.5 + Sharpen x3.0)")
        print(f"  Few-shot   : {'enabled' if USE_FEWSHOT else 'disabled'}")
        print(f"  Running    : {'free only' if QUICK_TEST_FREE_ONLY else 'all 6 models'}\n")

        # Load GT only if few-shot is enabled
        if USE_FEWSHOT:
            eval_idx, gt_dict = load_ground_truth()
            fewshot_blocks = build_fewshot_from_gt(gt_dict)
        else:
            fewshot_blocks = []
        img_b64 = img_to_b64(img_path)

        for name, model_id in MODELS.items():
            if QUICK_TEST_FREE_ONLY and name not in FREE_MODELS:
                continue
            pred, raw = predict(client, name, model_id, img_b64, fewshot_blocks)
            status = "[OK] " if pred is not None else "[WARN]"
            typ    = "FREE" if name in FREE_MODELS else "PAID"
            think  = " [thinking]" if name in THINKING_MODELS else ""
            print(f"  {status} [{typ}]{think} {name:<28} -> {pred or '?'}   "
                  f"raw='{raw[:40]}'")
            time.sleep(0.2)

        print("\n  [OK] Quick test done.")
        print("  -> Set QUICK_TEST_MODE = False for the full 500-image benchmark.\n")
        return

    # ── FULL 500-IMAGE BENCHMARK ──────────────────────────────────────────────
    eval_idx, gt_dict = load_ground_truth()
    n = len(eval_idx)

    # Build few-shot examples only when enabled (off by default for low-token mode)
    fewshot_blocks = build_fewshot_from_gt(gt_dict) if USE_FEWSHOT else []

    print(f"\n  Models      : {len(MODELS)}")
    print(f"  Images      : {n}")
    print(f"  Few-shot    : {len(fewshot_blocks)//2} examples per query")
    print(f"  Est. time   : ~{len(MODELS)*n*DELAY_BETWEEN_IMAGES/60:.0f} min")
    print(f"  gpt-4o-mini : ~${0.003*n:.2f} for {n} images")
    print(f"  claude-3.5  : ~${0.010*n:.2f} for {n} images\n")

    results_path = OUTPUT_DIR / "pipeline3_step1_results.json"
    all_results  = (json.loads(results_path.read_text())
                    if results_path.exists() else {})
    if all_results:
        print(f"  [RESUME] Existing results: {list(all_results.keys())}")

    for name, model_id in MODELS.items():
        if name in all_results:
            print(f"\n  [SKIP] Skipping {name} (already complete)"); continue
        res = benchmark_model(client, name, model_id, eval_idx, gt_dict,
                              fewshot_blocks)
        all_results[name] = res
        results_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"\n  [SAVE] Saved: {results_path}")
        time.sleep(DELAY_BETWEEN_MODELS)

    save_summary(all_results)
    print("\n  [OK] Step 1 complete - next: pipeline3_step3_full_labelling.py\n")


if __name__ == "__main__":
    main()