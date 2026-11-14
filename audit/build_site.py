"""Render the page: run the audit, then inject the results into the template.

    python -m audit.build_site              # 50 splits, seed 0
    python -m audit.build_site --splits 10  # quicker run for a smoke test

The page is a single self-contained HTML file, so the template is the whole site
and the only thing rewritten is the DATA block. Output defaults to
results/index.html.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .audit import run

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "template.html"
DATA_BLOCK = re.compile(r"/\*DATA\*/.*?/\*END\*/;", re.S)


def render(data: dict, template: Path = TEMPLATE) -> str:
    html = template.read_text()
    if not DATA_BLOCK.search(html):
        raise SystemExit(f"{template} has no /*DATA*/.../*END*/ block")
    payload = "/*DATA*/" + json.dumps(data) + "/*END*/;"
    return DATA_BLOCK.sub(lambda _: payload, html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "results" / "index.html"))
    parser.add_argument("--refresh", action="store_true", help="re-download the dataset")
    args = parser.parse_args()

    data = run(splits=args.splits, seed=args.seed, refresh=args.refresh)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data))
    print(f"wrote {out} — {data['splits']} splits, seed {args.seed}")
    for trait, strategies in data["res"].items():
        for strategy, values in strategies.items():
            accuracy = "  n/a" if values[2] is None else f"{values[2]:5.1f}"
            print(f"  {trait:4} {strategy:22} {values[0]:5.1f} / {values[1]:5.1f}  "
                  f"acc {accuracy}  gap {values[0] - values[1]:+6.1f}")
    print("  proxy AUC:", data["proxy"])


if __name__ == "__main__":
    main()
