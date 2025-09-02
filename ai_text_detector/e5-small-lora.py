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
    text = """--- Page 1 ---

Geography of Thane District 
Topography of Thane District includes several important physical divisions including Sahyadri hill 
ranges, forest areas, cultivated lands, and two major rivers. Ulhas River and Vaitarna River are the 
two main rivers of Thane District. Mainly three types of soils are found in Thane District - regur soil, 
red soil and brownish black soil. Regur soil, which is found in Dahanu, Palghar, Vasai and Thane 
tehsils, is fertile and useful for horticulture, paddy cultivation and vegetables. Whereas, red soil 
which is found in Mokhada, Talasari and some parts of other tehsils on the eastern slopes is useful 
for growing coarse millets.  

The third type of soil found in Bhiwandi, Kalyan and Shahapur tehsils is useful, particularly for paddy 
cultivation. Climate of Thane District is basically tropical. However, the climate of coastal plains 
differs from the climate on the eastern slopes. July is the rainiest month. Minimum temperature 
recorded here is 17.5 degree Celsius and maximum recorded temperature is 34.4 degree Celsius. 
Average annual rainfall is 2,576 mm. 

--- Page 1, Image 1 (OCR) ---

In this lab, you will learn how to configure a network to record traffic to and from an Apache web 
server using VPC Flow Logs. You will then export the logs to BigQuery for analysis. There are multiple 
use cases for VPC Flow Logs. For example, you might use VPC Flow Logs to determine where your 
applications are being accessed from to optimize network traffic expense, to create HTTP Load 
Balancers to balance traffic globally, or to denylist unwanted IP addresses with Cloud Armor.

--- Page 1, Image 2 (OCR) ---

Before you click the Start Lab button Read these instructions. Labs are timed and you cannot pause 
them. The timer, which starts when you click Start Lab, shows how long Google Cloud resources are 
made available to you. This hands-on lab lets you do the lab activities in a real cloud environment, 
not in a simulation or demo environment. It does so by giving you new, temporary credentials you use 
to sign in and access Google Cloud for the duration of the lab."""


    full_output, ai_score, human_score = classify(text)

    # 3) Print the full raw output FIRST (as requested)
    print("FULL OUTPUT:")
    for item in full_output:
        print(item)

    # 4) Then print clean scores
    print("\nCLEAN SCORES:")
    print(f"AI-generated score   : {ai_score:.4f} ({ai_score:.2%})")
    print(f"Human-generated score: {human_score:.4f} ({human_score:.2%})")
