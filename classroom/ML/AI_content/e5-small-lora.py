# file: run_local_e5_detector.py
import os
import math
from typing import Dict

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "MayZhou/e5-small-lora-ai-generated-detector"

# 1) Download the model locally (cached). You can change local_dir.
local_dir = snapshot_download(
    repo_id=MODEL_ID,
    local_dir="models/e5_small_lora_ai_detector",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ],
)

# 2) Load model & tokenizer from the local folder
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForSequenceClassification.from_pretrained(local_dir).to(device)
model.eval()

# Label mapping (from the model card):
# LABEL_0 -> Human-generated, LABEL_1 -> AI-generated
ID2LABEL: Dict[int, str] = {0: "Human-generated", 1: "AI-generated"}

def classify(text: str):
    # Tokenize
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}

    # Forward
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits.squeeze(0)  # shape: [2]

    # Softmax probabilities
    probs = torch.softmax(logits, dim=-1).tolist()  # [p0, p1]
    # Build a raw, pipeline-like output (this is your "whole output")
    full_output = [
        {"label": f"LABEL_{i}", "pretty_label": ID2LABEL[i], "score": float(p)}
        for i, p in enumerate(probs)
    ]

    # Compute “AI score” and “Human score”
    human_score = probs[0]  # LABEL_0
    ai_score = probs[1]     # LABEL_1

    return full_output, ai_score, human_score

if __name__ == "__main__":
    text = (
        "Nice — you’ve built a very thorough training script. Overall it’s solid and mostly correct, but I found a few bugs, potential pitfalls, and suggestions that will make the script more robust, reproducible, and correct (especially metric handling and a couple small lifecycle issues). Below I’ll list the problems (with why they matter) and give minimal, copy-pasteable fixes. I’ll mark the critical ones first."
    )

    full_output, ai_score, human_score = classify(text)

    # 3) Print the full raw output FIRST (as requested)
    print("FULL OUTPUT:")
    for item in full_output:
        print(item)

    # 4) Then print clean scores
    print("\nCLEAN SCORES:")
    print(f"AI-generated score   : {ai_score:.4f} ({ai_score:.2%})")
    print(f"Human-generated score: {human_score:.4f} ({human_score:.2%})")
