# ML NLP-OCR Document Processor

Automated document processor that watches an upload folder, extracts text and images from PDFs/DOCX/PPTX/images/CSV/TXT, performs OCR where needed, and writes structured Markdown with image references. It also tracks processed files and logs operations.

Main entry point: `app.py`

## Environment and prerequisites

Required software
- Python 3.9+
- Tesseract OCR
  - Windows default location is assumed: `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
  - Otherwise, set `pytesseract.pytesseract.tesseract_cmd` accordingly or add to PATH.

Python packages (common, not exhaustive)
- See imports in `app.py`: `pytesseract`, `Pillow`, `fitz` (PyMuPDF), `docling`, `watchdog`, `opencv-python`, `numpy`, `tqdm`, `python-docx`, `python-pptx`.

Example install
```bash
pip install pytesseract pillow pymupdf watchdog opencv-python numpy tqdm python-docx python-pptx docling
```

Environment variables (optional)
- None strictly required; behavior is configured by `processor_config.json`.

## Folder structure

- `uploaded_files/` — drop input files here (PDF, DOCX, PPTX, TXT, CSV, images).
- `processed_docs/` — generated Markdown files with metadata frontmatter.
- `extracted_images/` — saved images per document (subfolder per source file).
- `failed_processing/` — files that failed processing are copied here.
- `processed_files.json` — tracking DB of processed files and metadata.
- `processor_config.json` — runtime configuration (created/updated automatically).
- `file_processing.log` — rotating log file.

## Configuration (`processor_config.json`)

Keys
- `scan_interval_minutes`: periodic rescan interval.
- `ocr_languages`: Tesseract language codes (e.g., `eng` or `eng+deu`).
- `max_workers`: threads for parallel processing.
- `monitor_mode`: `watch`, `scan`, or `both`.
- `image_quality_factor`: multiplier for contrast/sharpness enhancement.

The app reads and updates this file automatically.

## How `app.py` works

- Starts with Tesseract check and directory setup.
- Watches `uploaded_files/` for new files (Watchdog) and/or periodically scans based on `monitor_mode`.
- For each file:
  - Chooses an extractor:
    - PDF: PyMuPDF + Docling to produce Markdown with `<IMAGE_PLACEHOLDER>` anchors; embedded images are extracted and OCR’d.
    - DOCX/PPTX: Docling for text + python-docx/python-pptx for image extraction + OCR.
    - Images: OCR directly (with preprocessing using OpenCV/Pillow).
    - TXT/CSV: read and format intelligently (CSV as Markdown table, capped to 100 rows).
  - Saves a Markdown file in `processed_docs/` with YAML frontmatter and references to saved images plus OCR text.
  - Records success/failure and metadata in `processed_files.json`.

Edge handling
- Skips tiny images (< 50px).
- Deduplicates by absolute path; skips already successful entries.
- Moves unsupported or failed files to `failed_processing/`.

## Run

Local run:
```bash
python app.py
```
Then drop files into `uploaded_files/`. The app will process them and create Markdown outputs in `processed_docs/`.

## Notes

- On Windows, ensure Tesseract is installed and accessible.
- If Docling fails to convert a document, the app falls back to basic text extraction and annotates the Markdown.
