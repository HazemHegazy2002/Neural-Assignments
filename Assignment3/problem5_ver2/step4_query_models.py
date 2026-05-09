"""
STEP 4 — MODEL QUERYING (Task 10: Keyphrase Extraction)

This script builds a unified prompt and prepares request/response files for
five models: ChatGPT, Gemini, ALLaM, Jais, Fanar.

Default behavior is DRY_RUN to avoid external calls. It generates:
  - step4_requests.json  (prompt per item)
  - step4_responses.json (empty slots to fill with model outputs)

To use real model outputs:
  1) Set DRY_RUN = False.
  2) Fill step4_responses.json manually, or load outputs from your tooling.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

DATASET_PATH = Path(__file__).with_name("dataset.json")
REQUESTS_PATH = Path(__file__).with_name("step4_requests.json")
RESPONSES_PATH = Path(__file__).with_name("step4_responses.json")

DRY_RUN = False
K = 5

# Fully local Colab mode: no API calls.
# Override with env var LOCAL_MODEL_ID if you want a different local checkpoint.
LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "google/mt5-small")
LOCAL_MAX_NEW_TOKENS = 96
LOCAL_TEMPERATURE = 0.2
LOCAL_TOP_P = 0.95
LOCAL_DEVICE = os.getenv("LOCAL_DEVICE", "auto")

# Only these model IDs will be queried when DRY_RUN is False.
RUN_MODEL_IDS = ["jais"]

MODELS = [
    {"id": "chatgpt", "name": "ChatGPT"},
    {"id": "gemini", "name": "Gemini"},
    {"id": "allam", "name": "ALLaM"},
    {"id": "jais", "name": "Jais"},
    {"id": "fanar", "name": "Fanar"},
]

PROMPT_TEMPLATE = (
    "You are given an Arabic paragraph. Extract exactly {k} keyphrases "
    "(1-3 words each) that best summarize the content. Return only a JSON "
    "array of {k} strings, no extra text.\n\n"
    "Paragraph:\n{paragraph}\n"
)


def build_prompt(paragraph: str, k: int = K) -> str:
    return PROMPT_TEMPLATE.format(k=k, paragraph=paragraph)


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError("dataset.json not found. Run step1 first.")
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def load_local_generator():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: transformers. Install it in Colab with `pip install transformers sentencepiece accelerate torch`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_ID)
    return tokenizer, model


def call_local_model(prompt: str, tokenizer, model) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: torch. Install it in Colab before running local inference."
        ) from exc

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    if LOCAL_DEVICE == "cuda" and torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        model = model.to("cuda")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=LOCAL_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=LOCAL_TEMPERATURE,
        top_p=LOCAL_TOP_P,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def parse_output(raw_text: str) -> list[str] | str:
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    return text


def main() -> None:
    dataset = load_dataset()
    tokenizer, model = load_local_generator()

    requests_payload = {
        "task": "Task 10 - Keyphrase Extraction",
        "k": K,
        "prompt_template": PROMPT_TEMPLATE,
        "items": [],
    }

    responses_payload = {
        "task": "Task 10 - Keyphrase Extraction",
        "k": K,
        "models": {model["id"]: {"name": model["name"], "outputs": []} for model in MODELS},
    }

    for item in dataset:
        prompt = build_prompt(item["paragraph"], K)
        requests_payload["items"].append({
            "id": item["id"],
            "type": item.get("type", ""),
            "topic": item.get("topic", ""),
            "source": item.get("source", ""),
            "source_detail": item.get("source_detail", ""),
            "prompt": prompt,
        })

        for model in MODELS:
            responses_payload["models"][model["id"]]["outputs"].append({
                "id": item["id"],
                "status": "pending",
                "output": "",
            })

    REQUESTS_PATH.write_text(
        json.dumps(requests_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    RESPONSES_PATH.write_text(
        json.dumps(responses_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if DRY_RUN:
        print("Dry run complete: wrote step4_requests.json and step4_responses.json")
        print("Fill step4_responses.json with model outputs to continue.")
        return

    print(f"\n[LOCAL] Using model: {LOCAL_MODEL_ID}")
    if LOCAL_DEVICE == "cuda":
        print("[LOCAL] CUDA requested; using GPU if available.")
    else:
        print("[LOCAL] Running on CPU unless you change LOCAL_DEVICE=cuda.")

    # Query only the configured model IDs (local mode uses a single local model).
    for model in MODELS:
        model_id = model["id"]
        if model_id not in RUN_MODEL_IDS:
            continue
        outputs = responses_payload["models"][model_id]["outputs"]
        for idx, item in enumerate(requests_payload["items"]):
            if outputs[idx]["status"] == "ok":
                continue
            try:
                raw = call_local_model(item["prompt"], tokenizer, model)
                outputs[idx]["output"] = parse_output(raw)
                outputs[idx]["status"] = "ok"
            except (TimeoutError, RuntimeError) as exc:
                outputs[idx]["status"] = "error"
                outputs[idx]["output"] = str(exc)
            time.sleep(0.2)

    RESPONSES_PATH.write_text(
        json.dumps(responses_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Model querying complete. Updated step4_responses.json")


if __name__ == "__main__":
    main()
