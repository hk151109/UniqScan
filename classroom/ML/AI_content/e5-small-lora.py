"""
Minimal Flask API exposing the E5-small LoRA AI-generated detector.

POST /classify
Request: { "text": "..." }
Response: { "ai_score": <float between 0 and 1> }

This loads the model once at startup. If a local model directory exists at
models/e5_small_lora_ai_detector, it will be used; otherwise it attempts to
download from Hugging Face (requires internet & credentials if needed).
"""

import os
from typing import Dict

from flask import Flask, jsonify, request
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "MayZhou/e5-small-lora-ai-generated-detector"
# Resolve models directory relative to repository root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LOCAL_DIR = os.path.join(ROOT_DIR, "models", "e5_small_lora_ai_detector")


def _ensure_model_dir() -> str:
    """Return a local directory containing the model; download if missing."""
    # If local dir already has the key files, use it
    expected = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ]
    if os.path.isdir(LOCAL_DIR) and all(
        os.path.isfile(os.path.join(LOCAL_DIR, f)) for f in expected
    ):
        return LOCAL_DIR

    # Otherwise, fetch using snapshot_download into LOCAL_DIR
    return snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        allow_patterns=expected,
    )


# Load model & tokenizer
_model_dir = _ensure_model_dir()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = AutoTokenizer.from_pretrained(_model_dir)
_model = AutoModelForSequenceClassification.from_pretrained(_model_dir).to(_device)
_model.eval()

# LABEL_0 -> Human-generated, LABEL_1 -> AI-generated
ID2LABEL: Dict[int, str] = {0: "Human-generated", 1: "AI-generated"}


def classify(text: str) -> float:
    """Return the AI-generated probability (float in [0,1])."""
    if not isinstance(text, str) or not text.strip():
        return 0.0

    enc = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(_device) for k, v in enc.items()}

    with torch.no_grad():
        out = _model(**enc)
        logits = out.logits.squeeze(0)

    probs = torch.softmax(logits, dim=-1).tolist()  # [p_human, p_ai]
    ai_score = float(probs[1])  # LABEL_1 is AI-generated
    return ai_score


# Flask API
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": _device,
        "model_dir": _model_dir,
    })


@app.route("/classify", methods=["POST"])
def classify_endpoint():
    try:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        score = classify(text)
        return jsonify({"ai_score": score})
    except Exception as e:
        # Don’t leak internals; return minimal error info
        return jsonify({"error": "classification_failed", "message": str(e)}), 500


if __name__ == "__main__":
    # Run on a separate port so the main Similarity API can call this service
    app.run(host="0.0.0.0", port=5001, debug=False)
