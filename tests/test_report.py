from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from musaprobe.report import render_report


class ReportTests(unittest.TestCase):
    def test_gradient_layout_and_stage_are_visible_in_details(self):
        layout = {"gradient_scope": "optimizer_owned_shard", "original_parameter_shape": [4],
                  "parameter_flat_range": [0, 2], "gradient_stage": "post_reduce_scatter_pre_unscale_pre_clip",
                  "gradient_parallel_coordinates": {"tensor": 0, "pipeline": 0, "data": 0}}
        record = {"rank": 0, "identity": {}, "status": "unaligned", "metrics": {},
                  "reference": layout, "candidate": {**layout, "parameter_flat_range": [2, 4]},
                  "gradient_layout": {"reference": layout, "candidate": {**layout, "parameter_flat_range": [2, 4]}}}
        comparison = {"overall": "fail", "summary": {"statuses": {"unaligned": 1}}, "records": [record], "first_difference": None}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            render_report(comparison, output)
            html = output.read_text()
            self.assertIn("梯度分片与采集阶段", html)
            self.assertIn("gradientLayout(r)", html)
            payload = re.search(r'<script id="comparison-data" type="application/json">(.*?)</script>', html, re.S).group(1)
            self.assertEqual(json.loads(payload)["records"][0]["gradient_layout"], record["gradient_layout"])

    def test_int64_metadata_is_exact_and_source_is_not_mutated(self):
        large = 2 ** 62 + 1
        comparison = {"overall": "pass", "summary": {}, "records": [],
                      "coverage": {"min": -large, "max": large, "bounds": [2 ** 53 - 1, 2 ** 53], "enabled": True}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            render_report(comparison, output)
            payload = re.search(r'<script id="comparison-data" type="application/json">(.*?)</script>', output.read_text(), re.S).group(1)
            parsed = json.loads(payload)
            self.assertEqual(parsed["coverage"]["min"], str(-large))
            self.assertEqual(parsed["coverage"]["max"], str(large))
            self.assertEqual(parsed["coverage"]["bounds"], [2 ** 53 - 1, str(2 ** 53)])
            self.assertIs(parsed["coverage"]["enabled"], True)
            self.assertEqual(comparison["coverage"]["max"], large)
            self.assertIsInstance(comparison["coverage"]["max"], int)

    def test_standalone_report_escapes_embedded_data(self):
        attack = '</script><script>alert("unsafe")</script>&\u2028\u2029'
        comparison = {"schema_version": 1, "overall": "incomplete", "reference": attack,
                      "candidate": "候选", "summary": {"statuses": {"statistics_only": 1}},
                      "coverage": {}, "first_difference": None,
                      "records": [{"rank": 0, "identity": {"module": attack}, "status": "statistics_only",
                                   "reason": attack, "reference": {"stack": [{"file": attack}]},
                                   "candidate": None, "metrics": {}}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "report.html"
            self.assertEqual(render_report(comparison, output), str(output.resolve()))
            html = output.read_text()
            self.assertNotIn(attack, html)
            payload = re.search(r'<script id="comparison-data" type="application/json">(.*?)</script>', html, re.S).group(1)
            self.assertEqual(json.loads(payload), comparison)
            self.assertIn("\\u003c/script", payload)
            self.assertNotIn("innerHTML", html)
            self.assertNotIn("fetch(", html)
            self.assertNotRegex(html, r'<(?:script|link)[^>]+(?:src|href)=')
            for feature in ('id="tree"', 'id="search"', 'id="status"', 'id="rank"', 'id="coverage"', 'id="details"'):
                self.assertIn(feature, html)
            self.assertIn("比较记录", html)
            self.assertIn("含NaN/Inf记录", html)
            self.assertNotIn("对齐记录", html)
            self.assertIn("else if(data.overall==='fail')", html)
            self.assertIn("不能据此认定模型数值异常", html)

    @unittest.skipUnless(shutil.which("node"), "Node is optional for JavaScript syntax validation")
    def test_report_script_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            render_report({"overall": "pass", "records": [], "summary": {}}, output)
            script = re.findall(r'<script>(.*?)</script>', output.read_text(), re.S)[0]
            script_file = Path(directory) / "report.js"
            script_file.write_text(script)
            subprocess.run(["node", "--check", str(script_file)], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
