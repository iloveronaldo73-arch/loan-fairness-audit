"""Fair-lending audit: build the page and result tables, then serve them locally.

    python app.py            # audit + comparison, writes results/
    python app.py build      # audit only
    python app.py compare    # mitigation comparison only
    python app.py serve      # serve results/ on http://127.0.0.1:4173

The numbers come from audit/audit.py and audit/compare_cutoffs.py; this script
only decides where the artifacts go.
"""

from __future__ import annotations

import argparse
import csv
import functools
import http.server
import json
from pathlib import Path
from typing import Optional

from audit.audit import run as run_audit
from audit.build_site import render
from audit.compare_cutoffs import run as run_comparison

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _write_audit(data: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "audit.json").write_text(json.dumps(data, indent=2) + "\n")
    (RESULTS / "index.html").write_text(render(data))

    with (RESULTS / "audit.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "trait", "strategy", "other_group", "other_rate", "ref_group", "ref_rate",
            "accuracy", "gap", "gap_p2_5", "gap_p97_5",
        ])
        for trait, strategies in data["res"].items():
            groups = data["groups"][trait]
            for name, values in strategies.items():
                writer.writerow([
                    trait, name, groups["other"], f"{values[0]:.1f}",
                    groups["ref"], f"{values[1]:.1f}",
                    "" if values[2] is None else f"{values[2]:.1f}",
                    f"{values[0] - values[1]:.1f}", f"{values[3]:.1f}", f"{values[4]:.1f}",
                ])


def _write_cutoffs(summary: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "cutoffs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "trait", "rule", "other_rate", "ref_rate", "approval_gap", "gap_p2_5",
            "gap_p97_5", "tpr_gap", "fpr_gap", "accuracy",
        ])
        for trait, rules in summary.items():
            for name, values in rules.items():
                writer.writerow([
                    trait, name, f"{values['other']:.1f}", f"{values['ref']:.1f}",
                    f"{values['gap']:.1f}", f"{values['gap_low']:.1f}",
                    f"{values['gap_high']:.1f}", f"{values['tpr_gap']:.1f}",
                    f"{values['fpr_gap']:.1f}", f"{values['accuracy']:.1f}",
                ])


def _write_summary(data: dict, summary: Optional[dict]) -> None:
    lines = [
        f"# {data['dataset']}",
        "",
        f"{data['rows']:,} applications · {data['base']}% {data['label']} · "
        f"{data['splits']} random 60/20/20 splits · model {data['model']}.",
        "",
    ]
    for trait, strategies in data["res"].items():
        groups = data["groups"][trait]
        lines += [
            f"## {trait}: {groups['other']} vs {groups['ref']}",
            "",
            f"| strategy | {groups['other']} | {groups['ref']} | accuracy | gap | 95% range |",
            "|---|---|---|---|---|---|",
        ]
        for name, values in strategies.items():
            accuracy = "—" if values[2] is None else f"{values[2]:.1f}"
            lines.append(
                f"| {name} | {values[0]:.1f} | {values[1]:.1f} | {accuracy} | "
                f"{values[0] - values[1]:+.1f} | {values[3]:+.1f} to {values[4]:+.1f} |"
            )
        lines.append("")

    if summary:
        lines += [
            "## Cutoff rules on the blind model",
            "",
            "| trait | rule | approval gap | 95% range | TPR gap | FPR gap | accuracy |",
            "|---|---|---|---|---|---|---|",
        ]
        for trait, rules in summary.items():
            for name, values in rules.items():
                lines.append(
                    f"| {trait} | {name} | {values['gap']:+.1f} | {values['gap_low']:+.1f} to "
                    f"{values['gap_high']:+.1f} | {values['tpr_gap']:+.1f} | "
                    f"{values['fpr_gap']:+.1f} | {values['accuracy']:.1f} |"
                )
        lines.append("")

    (RESULTS / "summary.md").write_text("\n".join(lines))


def serve(port: int = 4173) -> None:
    page = RESULTS / "index.html"
    if not page.exists():
        raise SystemExit("results/index.html is missing — run `python app.py build` first")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(RESULTS))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"serving results/ at http://127.0.0.1:{port}  (ctrl-c to stop)")
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", nargs="?", default="all",
                        choices=["all", "build", "compare", "serve"])
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()

    if args.command == "serve":
        serve(args.port)
        return

    data = None
    summary = None
    if args.command in ("all", "build"):
        data = run_audit(splits=args.splits, seed=args.seed)
        _write_audit(data)
        print(f"wrote results/audit.json, results/audit.csv, results/index.html "
              f"({data['rows']:,} applications, {data['splits']} splits)")
    if args.command in ("all", "compare"):
        summary = run_comparison(splits=args.splits, seed=args.seed)
        _write_cutoffs(summary)
        print("wrote results/cutoffs.csv")
    if data is not None:
        _write_summary(data, summary)
        print("wrote results/summary.md")


if __name__ == "__main__":
    main()
