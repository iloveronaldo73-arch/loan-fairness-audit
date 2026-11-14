"""Plain-English AI summary of the audit results, checked against the real numbers.

Setup (free key from Google AI Studio; NEVER commit the key to GitHub):
    export GEMINI_API_KEY=your_key        # in Colab: add it under Secrets instead
Run:
    python -m audit.ai_summary            # reads results/summary.md, writes results/ai_summary.md
    python -m audit.ai_summary --dry-run  # prints the prompt, makes no API call

Every number in the AI text is compared with the source results; any number
that does not appear there is flagged in the output file.

This project is a prototype and is not financial advice.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = """You are explaining a fairness audit of a loan-approval model to a non-technical loan officer.
Write at most 180 words, as short paragraphs (no numbered lists), covering:
what was tested, the main finding on age, the main finding on sex, what the fix cost, and the biggest caveat.
Rules:
- Use ONLY numbers that appear in the RESULTS below. Do not calculate new numbers or add outside facts.
- If a range includes zero, say the result is uncertain.
- Do not recommend that a lender use this model. Plain language, no jargon.

RESULTS:
{results}
"""

_NUM = re.compile(r"\d+(?:\.\d+)?")
_LIST_MARK = re.compile(r"^\s*\(?\d+[.)]\s+", re.M)


def _values(text):
    return {round(float(x), 4) for x in _NUM.findall(text)}


def ungrounded_numbers(summary, source):
    """Numbers in the AI text that do not appear anywhere in the source results."""
    summary = _LIST_MARK.sub("", summary)
    return sorted(_values(summary) - _values(source))


def call_gemini(prompt, model, key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["candidates"][0]["content"]["parts"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/summary.md")
    ap.add_argument("--out", default="results/ai_summary.md")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                    help="check Google's docs for the current free model name")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    source = Path(a.results).read_text()[:20000]
    prompt = PROMPT.format(results=source)
    if a.dry_run:
        print(prompt); return
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY first (see the top of this file).")
    try:
        text = call_gemini(prompt, a.model, key)
    except urllib.error.HTTPError as e:
        sys.exit(f"API error {e.code}: {e.read().decode()[:300]}")

    bad = ungrounded_numbers(text, source)
    out = [f"# AI-generated summary\n\n*Written by {a.model} from `{a.results}`. "
           "It can contain mistakes: the tables in the report are the source of truth. "
           "This project is a prototype and is not financial advice.*\n", text.strip(), ""]
    if bad:
        out.append(f"> **Check these numbers by hand:** {bad} do not appear in the source results.")
    else:
        out.append("> Number check passed: every number above appears in the source results.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
