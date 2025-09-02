# Matcher Algorithm and Plagiarism Detector

Two components:
- `matcher.py`: a reusable text matcher that finds extended n-gram matches between two texts.
- `plag-detect.py`: a CLI tool that scans a folder for text-like files, compares them pairwise, and generates HTML/TXT plagiarism reports with highlighted matches and citation mapping.

## Environment and prerequisites

- Python 3.9+
- NLTK with `punkt` and `stopwords` corpora (the CLI downloads them on first run if missing).

Install packages:
```bash
pip install nltk termcolor reportlab watchdog
```

Optional environment variables: none required.

## matcher.py

Core concepts
- Tokenization with NLTK and stemming (LancasterStemmer), optional stopword removal.
- N-gram indexing per text; initial matches via `difflib.SequenceMatcher` on n-grams.
- Healing nearby matches (`minDistance`) and extending boundaries using edit distance heuristics.

Key classes
- `Text(raw_text, label, removeStopwords=True)` — prepares tokens, spans, and n-grams.
- `Matcher(textA, textB, threshold=3, cutoff=5, ngramSize=3, minDistance=8, silent=False)` — computes matches.
- `match()` returns number of matches and span locations.

## plag-detect.py

What it does
- Monitors a target folder for text-like files (txt, md, code, etc.).
- On new or changed files, compares against known files and generates:
  - HTML report with highlighted sections and citation numbers.
  - Plain-text summary report.
- Maintains checksums to avoid reprocessing unchanged content.

Run
```bash
python plag-detect.py files --output plagiarism_reports
```

Arguments
- `folder` (positional): directory to monitor.
- `--output, -o`: reports directory (default `plagiarism_reports`).
- `--threshold, -t`: initial matching threshold (default 3).
- `--cutoff, -c`: minimum kept match size after extending (default 5).
- `--ngram, -n`: n-gram size (default 3).
- `--distance, -d`: healing distance (default 8).
- `--silent, -s`: suppress detailed logs.
- `--scan-only`: run once and exit.

Outputs
- `plagiarism_reports/reports/<file>_report.html` and `_report.txt`.
- `plagiarism_reports/plagiarism_detector.log` for run logs.

Notes
- The HTML report includes similarity percentages per source and inline highlighted citations `[n]` in the analyzed content.
