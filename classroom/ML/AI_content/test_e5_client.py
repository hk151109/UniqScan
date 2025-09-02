import argparse
import json
import sys
import requests


def classify(url: str, text: str):
    resp = requests.post(url, json={"text": text}, timeout=60)
    resp.raise_for_status()
    data = resp.json()  # expected: { "ai_score": <float 0..1> }
    ai_score = float(data.get("ai_score", 0.0))
    human_score = 1.0 - ai_score

    # Print results similar to prior test format
    print("CLEAN SCORES:")
    print(f"AI-generated score   : {ai_score:.4f} ({ai_score:.2%})")
    print(f"Human-generated score: {human_score:.4f} ({human_score:.2%})")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Test client for e5-small-lora AI score API")
    parser.add_argument(
        "--url",
        default="http://localhost:5001/classify",
        help="Classifier API endpoint (default: http://localhost:5001/classify)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="Inline text to classify")
    group.add_argument("--file", help="Path to a text file to classify")
    args = parser.parse_args(argv)

    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        # Fallback sample text
        text = (
            "This is a short sample for testing the e5-small-lora AI detection API. "
            "You can pass --text or --file to customize the input."
        )

    classify(args.url, text)


if __name__ == "__main__":
    main()
