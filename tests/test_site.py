import json
import re
import unittest
from pathlib import Path

from audit.build_site import TEMPLATE, render

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "results" / "index.html"
RESULTS = ROOT / "results" / "audit.json"
DATA_BLOCK = re.compile(r"/\*DATA\*/(.*?)/\*END\*/;", re.S)


class TemplateTest(unittest.TestCase):
    def test_template_has_exactly_one_data_block(self):
        self.assertEqual(len(DATA_BLOCK.findall(TEMPLATE.read_text())), 1)

    def test_render_round_trips_the_payload(self):
        payload = {
            "rows": 10,
            "base": 90.0,
            "res": {"age": {"approve everyone": [100.0, 100.0, 90.0, 0.0, 0.0]}},
        }
        html = render(payload)
        match = DATA_BLOCK.search(html)
        self.assertIsNotNone(match)
        self.assertEqual(json.loads(match.group(1)), payload)


class BuiltPageTest(unittest.TestCase):
    def test_page_and_results_json_agree(self):
        if not PAGE.exists() or not RESULTS.exists():
            self.skipTest("run `python app.py build` first")
        page_payload = json.loads(DATA_BLOCK.search(PAGE.read_text()).group(1))
        self.assertEqual(page_payload, json.loads(RESULTS.read_text()))


if __name__ == "__main__":
    unittest.main()
