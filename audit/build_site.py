"""Write the audit numbers from results/ into the data block of docs/index.html.

    python -m audit.build_site
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def build(results=ROOT / "results", page=ROOT / "docs" / "index.html"):
    results, page = Path(results), Path(page)
    meta = json.loads((results / "meta.json").read_text())
    summary = pd.read_csv(results / "summary.csv")
    proxy = pd.read_csv(results / "proxy_check.csv")

    res = {}
    for _, r in summary.iterrows():
        acc = None if pd.isna(r["accuracy_mean"]) else round(r["accuracy_mean"] * 100, 1)
        res.setdefault(r["attribute"], {})[r["strategy"]] = [
            round(r["approval_other_mean"] * 100, 1), round(r["approval_ref_mean"] * 100, 1), acc,
            round(r["approval_gap_lo"] * 100, 1), round(r["approval_gap_hi"] * 100, 1),
            None if pd.isna(r["cost_mean"]) else round(r["cost_mean"], 3)]  # gap range across splits, cost per applicant
    site = {**meta, "res": res, "proxy": {r["attribute"]: round(r["auc"], 2) for _, r in proxy.iterrows()}}

    extra = results / "extra.json"
    if extra.exists():
        site["extra"] = json.loads(extra.read_text())

    html = page.read_text()
    block = "/*DATA*/" + json.dumps(site) + "/*END*/"
    new, n = re.subn(r"/\*DATA\*/.*?/\*END\*/", lambda m: block, html, flags=re.S)
    if n != 1:
        raise SystemExit("data markers not found in " + str(page))
    page.write_text(new)
    print("updated", page)


if __name__ == "__main__":
    build()
