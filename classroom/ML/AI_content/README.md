# AI Content Detection Service (E5 Small LoRA)

Flask microservice exposing MayZhou/e5-small-lora-ai-generated-detector via a simple HTTP API.

Files
- `e5-small-lora.py`: the Flask app exposing `/health` and `/classify`.
- `test_e5_client.py`: small client to send text and print AI/Human scores.

## Environment

Python packages
```bash
pip install flask torch transformers huggingface_hub requests
```

Optional environment variables
- `HF_TOKEN` — if the model requires authentication.
- `AI_SCORE_API_URL` — not used directly here; consumed by the Similarity service.

Model cache
- First run downloads to `models/e5_small_lora_ai_detector/` at repository root.

## API

- `GET /health`
  - Response: `{ "status": "ok", "device": "cpu|cuda", "model_dir": "..." }`

- `POST /classify`
  - Request: `{ "text": "..." }`
  - Response: `{ "ai_score": 0..1 }` where 1.0 = very likely AI-generated.

## Run

```bash
python e5-small-lora.py
# Service listens on 0.0.0.0:5001
```

## Test

```bash
python test_e5_client.py --url http://localhost:5001/classify --text "Sample text"
```
