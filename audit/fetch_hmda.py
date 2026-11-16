"""Download HMDA loan-level data for one state and year from the CFPB Data Browser API.

    python -m audit.fetch_hmda --state WA --year 2025

Writes data/hmda_raw.csv, which is large and listed in .gitignore. The audit
saves a small filtered sample to data/hmda_sample.csv that you can commit.
"""
import argparse
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states={state}&years={year}&actions_taken=1,3&total_units=1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True, help="two-letter code, for example WA")
    p.add_argument("--year", default="2025")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "hmda_raw.csv"))
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    url = URL.format(state=args.state.upper(), year=args.year)
    print("downloading", url)
    try:
        with urllib.request.urlopen(url) as resp, open(out, "wb") as f:
            total = 0
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                total += len(chunk)
                print(f"\r{total / 1e6:.0f} MB", end="", flush=True)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"\nThe API answered {e.code}. Download a CSV by hand from https://ffiec.cfpb.gov/data-browser/ "
                         "and pass it with: python -m audit.run_audit --dataset hmda --csv FILE")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
