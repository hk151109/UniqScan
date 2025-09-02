# AI Text Detector

Lightweight utilities to detect AI-generated content using different providers:

- `e5-small-lora.py`: run a local Hugging Face sequence-classification model (MayZhou/e5-small-lora-ai-generated-detector) and print AI/Human scores.
- `SzegedAI.py`: sample client that calls the hosted Gradio Space "SzegedAI/AI_Detector".
- `content-detector2.py`: sample client for RapidAPI "ai-content-detector2" endpoint.

## Environment and prerequisites

- Python 3.9+ recommended.
- GPU (CUDA) optional but supported by PyTorch.
- For RapidAPI client, you need an API key.

Install packages:

```bash
pip install -r requirements.txt
# Plus utilities used by the scripts:
pip install huggingface_hub requests
```

Optional environment variables:

- `HUGGINGFACE_HUB_CACHE` — override HF cache directory (optional).
- `CUDA_VISIBLE_DEVICES` — choose GPU device (optional).
- `RAPIDAPI_KEY` — for convenience; if set, use it in `content-detector2.py` instead of the hardcoded placeholder.

## Scripts and how they work

### 1) Local e5-small LoRA detector (`e5-small-lora.py`)

What it does
- Downloads the model snapshot into `models/e5_small_lora_ai_detector/` (cached).
- Loads tokenizer and model, classifies text, and prints both raw probabilities and clean AI/Human scores.

Run
```bash
python e5-small-lora.py
```

Output
- Prints a "FULL OUTPUT" list of label/score pairs, then
- "CLEAN SCORES" with `AI-generated score` and `Human-generated score` (0..1 and %).

### 2) Gradio Space client (`SzegedAI.py`)

What it does
- Uses `gradio_client.Client("SzegedAI/AI_Detector")` to call `/classify_text` for a few sample texts and prints the result returned by the Space.

Run
```bash
python SzegedAI.py
```

Notes
- Requires internet access; response schema is owned by the Space and may change.

### 3) RapidAPI client (`content-detector2.py`)

What it does
- Sends a POST request to `https://ai-content-detector2.p.rapidapi.com/analyzePatterns` with a `text` payload.
- Interprets the `aggregated` result as an AI-vs-Human confidence and prints AI/Human percentages.

Configure
- Replace `"api_key"` header in the file with your actual key, or load from `os.environ["RAPIDAPI_KEY"]`.

Run
```bash
python content-detector2.py
```

## Tips

- If the HF model is private, set `HF_TOKEN` in your environment and be logged into `huggingface_hub`.
- If PyTorch cannot find CUDA, it will fall back to CPU automatically.
