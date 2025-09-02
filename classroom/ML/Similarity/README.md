# Similarity and Plagiarism Analysis API (with PDF/OCR)

Flask service that:
- Downloads a submitted file via HTTP URL, extracts text (PDF/DOCX/PPTX/Images/CSV/TXT) with OCR.
- Computes similarity vs a local corpus of extracted texts.
- Calls the AI content detection microservice to estimate AI probability.
- Produces an overall plagiarism score and returns an HTML report along with structured scores.

Main file: `app.py`

## Environment

Python packages
```bash
pip install flask requests nltk pytesseract pillow pymupdf opencv-python numpy python-docx python-pptx
```

System dependency
- Tesseract OCR installed and on PATH. On Windows the app defaults to `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`.

Environment variables
- `AI_SCORE_API_URL` (default `http://localhost:8001`): Base URL of the AI detector service; the code uses `${AI_SCORE_API_URL}/classify`.

## Folders

- `downloads/` — per-request subfolders with original files.
- `extracted_text/` — saved `.txt` of extracted content, used as the comparison corpus.
- `reports/` — JSON and HTML outputs.
- `logs/` — service logs.

## API

- `POST /grade/analyze`
  - Request JSON:
    ```json
    {
      "student_id": "string",
      "student_name": "string",
      "file_url": "http://.../uploads/homeworks/file.pdf",
      "assignment_id": "string",
      "classroom_name": "string"
    }
    ```
  - Response JSON (abridged):
    ```json
    {
      "status": "completed|partial|failed",
      "similarity_analysis": {
        "similarity_score": 0.0,
        "total_comparisons": 0,
        "detailed_results": [ {"source":"...","similarity": 0.0, "matched_text": "..."} ],
        "report_path": "reports/.../similarity_report.json"
      },
      "ai_analysis": {
        "ai_percentage": 0.0,
        "interpretation": "...",
        "chunks_analyzed": 1,
        "confidence": 0.8
      },
      "plagiarism_analysis": {
        "plagiarism_score": 0.0,
        "risk_level": "low|moderate|high|critical",
        "contributing_factors": ["..."]
      },
      "report_html": "<!DOCTYPE html>..."
    }
    ```

- `GET /health`
  - Service status and config.

- `POST /test-processing`
  - Body: `{ "file_url": "http://..." }`
  - Returns extracted content preview and metrics.

## Flow

1. Receive analyze request from Node backend with `file_url` pointing to the uploaded file on the Node server.
2. Download file, determine content type, and extract text via PDF/OCR pipeline.
3. Save extracted text into `extracted_text/` to grow a local corpus.
4. Compare against other texts using an n-gram matcher with healing/extension.
5. Call AI detector service `/classify` and get `ai_score` -> `%`.
6. Combine into an overall plagiarism score and render a detailed HTML report.

## Run

```bash
set AI_SCORE_API_URL=http://localhost:5001
python app.py
# Service listens on Flask default (e.g., 5000 via flask run or use app.run in a wrapper)
```

Tip: Start the AI service first on port 5001 (see `../AI_content`). Set `AI_SCORE_API_URL` accordingly.
