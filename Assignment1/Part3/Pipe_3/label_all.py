"""
Pipeline 3 - Step 3: Full Dataset Labelling by Both LLMs
=========================================================
LLM1 = gemini-2.5-flash  (~$1.50 for 10,000 images)
LLM2 = gpt-4o-mini       (~$0.75 for 10,000 images)
Total cost estimate: ~$2.25

Starts automatically — no keyboard input needed.
Auto-saves every 100 images. Safe to stop and resume.
"""

import re, json, time, base64, io, sys
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image, ImageEnhance

# ─────────────────────────────────────────────
# CONFIGURATION — only change these two lines
# ─────────────────────────────────────────────
API_KEY  = "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"
DATA_DIR = Path(r"C:\Neural\Indian_Digits_Train")

# ─────────────────────────────────────────────
# OUTPUT FOLDER
# ─────────────────────────────────────────────
OUTPUT_DIR = Path(r"C:\Neural\Pipeline3_Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
LLM1 = {
    "name":        "gemini-2.5-flash",
    "model_id":    "google/gemini-2.5-flash",
    "cost_per_1k": 0.15,
}

LLM2 = {
    "name":        "gpt-4o-mini",
    "model_id":    "openai/gpt-4o-mini",
    "cost_per_1k": 0.075,
}

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
DELAY       = 1.0   # seconds between API calls
MAX_RETRIES = 2     # retries on failure
RETRY_WAIT  = 5.0   # seconds before retry
IMG_SIZE    = 224   # upscale resolution

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert in recognizing handwritten Indian/Eastern Arabic digits. "
    "Reply with exactly one ASCII digit 0-9. Nothing else."
)

USER_PROMPT = """\
This image contains ONE handwritten Eastern Arabic/Indian numeral.
These digits look DIFFERENT from Western digits.

Shape guide:
- 0: tiny dot or small oval
- 1: simple vertical stroke
- 2: backward Z or hook curving right
- 3: reversed 3 or two humps on right
- 4: like 3 with a tail going down-left
- 5: circle or rounded oval
- 6: like L shape, horizontal then down
- 7: upside-down V or checkmark/tent shape
- 8: two bumps stacked like lambda
- 9: loop at top with tail curving right

Reply with ONLY one digit (0-9). No words. No explanation."""

# Unicode digit map
_UMAP = {
    "٠":"0","١":"1","٢":"2","٣":"3","٤":"4",
    "٥":"5","٦":"6","٧":"7","٨":"8","٩":"9",
    "۰":"0","۱":"1","۲":"2","۳":"3","۴":"4",
    "۵":"5","۶":"6","۷":"7","۸":"8","۹":"9",
    "०":"0","१":"1","२":"2","३":"3","४":"4",
    "५":"5","६":"6","७":"7","८":"8","९":"9",
}


def img_to_b64(image_index: int) -> str:
    """Load BMP → upscale → enhance → base64 PNG."""
    path = DATA_DIR / f"{image_index}.bmp"
    img  = Image.open(path).convert("L")
    img  = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img  = ImageEnhance.Contrast(img).enhance(2.5)
    img  = ImageEnhance.Sharpness(img).enhance(3.0)
    img  = img.convert("RGB")
    buf  = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def normalize(text) -> str | None:
    """Extract single digit (0-9) from any LLM response."""
    if not text:
        return None
    text      = str(text).strip()
    converted = "".join(_UMAP.get(c, c) for c in text)
    digits    = re.findall(r"[0-9]", converted)
    return digits[-1] if digits else None


def predict_one(client, model_id, image_index):
    """Send one image to one model. Returns (digit, raw_response)."""
    for attempt in range(MAX_RETRIES):
        try:
            img_b64  = img_to_b64(image_index)
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": USER_PROMPT}
                    ]}
                ],
                max_tokens=5,
                temperature=0,
            )
            raw   = response.choices[0].message.content
            digit = normalize(raw)
            return digit, str(raw).strip()

        except Exception as e:
            wait = RETRY_WAIT * (attempt + 1)
            print(f"    Retry {attempt+1}/{MAX_RETRIES} "
                  f"({str(e)[:50]}) waiting {wait:.0f}s...")
            time.sleep(wait)

    return None, "ERROR: max retries exceeded"


def label_all_images(client, llm_info, llm_num):
    """
    Label all 10,000 images with one LLM.
    Auto-saves every 100 images.
    Resumes automatically if stopped.
    """
    name     = llm_info["name"]
    model_id = llm_info["model_id"]
    est_cost = llm_info["cost_per_1k"] * 10

    print(f"\n{'='*55}")
    print(f"LLM{llm_num}: {name}")
    print(f"Model:       {model_id}")
    print(f"Est. cost:   ~${est_cost:.2f}")
    print(f"{'='*55}")

    # Checkpoint file for this model
    ckpt_file = OUTPUT_DIR / f"step3_llm{llm_num}_checkpoint.json"
    done      = {}

    # Load existing progress if resuming
    if ckpt_file.exists():
        done = json.loads(ckpt_file.read_text())
        print(f"Resuming: {len(done)}/10000 already done")
    else:
        print("Starting fresh...")

    total_images = 10000
    start_time   = time.time()

    for img_idx in range(1, total_images + 1):
        key = str(img_idx)

        # Skip already done
        if key in done:
            continue

        # Check image exists
        img_path = DATA_DIR / f"{img_idx}.bmp"
        if not img_path.exists():
            done[key] = None
            continue

        # Get prediction
        pred, raw = predict_one(client, model_id, img_idx)
        done[key] = pred

        # Show progress every 50 images
        if img_idx % 50 == 0:
            elapsed    = time.time() - start_time
            remaining  = total_images - img_idx
            eta_sec    = (elapsed / img_idx) * remaining if img_idx > 0 else 0
            eta_min    = eta_sec / 60
            done_count = len([v for v in done.values() if v is not None])
            print(f"  [{img_idx:5d}/10000] "
                  f"pred={pred or '?'} | "
                  f"labelled={done_count} | "
                  f"ETA={eta_min:.0f}min")

        # Auto-save every 100 images
        if img_idx % 100 == 0:
            ckpt_file.write_text(json.dumps(done))
            null_count = len([v for v in done.values() if v is None])
            print(f"  Checkpoint saved | "
                  f"{img_idx}/10000 ({img_idx/total_images*100:.1f}%) | "
                  f"nulls={null_count}")

        time.sleep(DELAY)

    # Final checkpoint save
    ckpt_file.write_text(json.dumps(done))

    # Convert to numpy array (0-based: index 0 = image 1.bmp)
    predictions = np.full(total_images, -1, dtype=int)
    null_count  = 0

    for i in range(1, total_images + 1):
        val = done.get(str(i))
        if val is not None:
            try:
                predictions[i - 1] = int(val)
            except Exception:
                predictions[i - 1] = -1
                null_count += 1
        else:
            predictions[i - 1] = -1
            null_count += 1

    print(f"\nLLM{llm_num} ({name}) DONE!")
    print(f"  Labelled:    {total_images - null_count}")
    print(f"  Failed:      {null_count}  ({null_count/total_images:.2%})")

    print(f"\nLabel distribution:")
    for d in range(10):
        count = int(np.sum(predictions == d))
        bar   = "█" * (count // 100)
        print(f"  Digit {d}: {count:5d}  {bar}")

    return predictions


def save_final_results(llm1_preds, llm2_preds):
    """Save both predictions and compute agreement statistics."""

    # Save arrays
    np.save(OUTPUT_DIR / "step3_llm1_predictions.npy", llm1_preds)
    np.save(OUTPUT_DIR / "step3_llm2_predictions.npy", llm2_preds)

    # Compute agreement
    total     = 10000
    agreed    = int(np.sum(llm1_preds == llm2_preds))
    disagreed = total - agreed
    llm1_null = int(np.sum(llm1_preds == -1))
    llm2_null = int(np.sum(llm2_preds == -1))
    cost1     = LLM1["cost_per_1k"] * 10
    cost2     = LLM2["cost_per_1k"] * 10

    summary = {
        "llm1_name":                    LLM1["name"],
        "llm1_model_id":                LLM1["model_id"],
        "llm2_name":                    LLM2["name"],
        "llm2_model_id":                LLM2["model_id"],
        "total_images":                 total,
        "agreed":                       agreed,
        "disagreed":                    disagreed,
        "agree_rate_pct":               round(agreed / total * 100, 2),
        "disagree_rate_pct":            round(disagreed / total * 100, 2),
        "llm1_null":                    llm1_null,
        "llm2_null":                    llm2_null,
        "est_cost_llm1_usd":            round(cost1, 2),
        "est_cost_llm2_usd":            round(cost2, 2),
        "est_cost_total_usd":           round(cost1 + cost2, 2),
        "manual_time_seconds":          disagreed * 10,
        "manual_time_minutes":          round(disagreed * 10 / 60, 1),
    }

    (OUTPUT_DIR / "step3_summary.json").write_text(
        json.dumps(summary, indent=2))

    print("\n" + "=" * 55)
    print("STEP 3 COMPLETE")
    print("=" * 55)
    print(f"Total images:      {total:,}")
    print(f"Both agreed:       {agreed:,}  ({agreed/total:.1%})")
    print(f"Disagreed:         {disagreed:,}  ({disagreed/total:.1%})")
    print(f"LLM1 nulls:        {llm1_null}")
    print(f"LLM2 nulls:        {llm2_null}")
    print(f"Manual time:       {disagreed} × 10s = "
          f"{disagreed*10/60:.1f} min")
    print(f"API cost used:     ~${cost1+cost2:.2f}")
    print(f"\nFiles saved:")
    print(f"  step3_llm1_predictions.npy")
    print(f"  step3_llm2_predictions.npy")
    print(f"  step3_summary.json")
    print(f"\nNext: Run pipe3_step4_agreement.py")


# ─────────────────────────────────────────────
# MAIN — starts automatically, no keyboard needed
# ─────────────────────────────────────────────
if __name__ == "__main__":

    cost1 = LLM1["cost_per_1k"] * 10
    cost2 = LLM2["cost_per_1k"] * 10

    print("=" * 55)
    print("Pipeline 3 - Step 3: Label All 10,000 Images")
    print("=" * 55)
    print(f"LLM1: {LLM1['name']:20s}  ~${cost1:.2f}")
    print(f"LLM2: {LLM2['name']:20s}  ~${cost2:.2f}")
    print(f"Total cost:                ~${cost1+cost2:.2f}")
    print(f"Budget remaining after:    ~${5.00-cost1-cost2:.2f}")
    print(f"Est. time: ~{10000*DELAY*2/3600:.1f} hours")
    print("Auto-saves every 100 images. Safe to stop/resume.")
    print("=" * 55)
    print("Starting in 5 seconds...")
    time.sleep(5)

    # Connect to OpenRouter
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    # LLM1 labels all 10,000 images
    print(f"\n{'#'*55}")
    print(f"# STARTING LLM1: {LLM1['name']}")
    print(f"{'#'*55}")
    llm1_predictions = label_all_images(client, LLM1, llm_num=1)

    print("\nLLM1 done! Waiting 5 seconds before LLM2...")
    time.sleep(5)

    # LLM2 labels all 10,000 images independently
    print(f"\n{'#'*55}")
    print(f"# STARTING LLM2: {LLM2['name']}")
    print(f"{'#'*55}")
    llm2_predictions = label_all_images(client, LLM2, llm_num=2)

    # Save everything for Step 4
    save_final_results(llm1_predictions, llm2_predictions)