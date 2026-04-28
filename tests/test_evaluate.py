import tempfile
import unittest
from pathlib import Path

from src.common.io import write_jsonl
from src.experiments.evaluate import evaluate


class EvaluateTest(unittest.TestCase):
    def test_evaluate_writes_summary_and_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            predictions = tmp_path / "predictions.jsonl"
            write_jsonl(
                predictions,
                [
                    {
                        "model": "m",
                        "condition": "useful",
                        "gold_label": "support",
                        "metric_label": "support",
                        "predicted_label": "support",
                    },
                    {
                        "model": "m",
                        "condition": "useful",
                        "gold_label": "deny",
                        "metric_label": "deny",
                        "predicted_label": "deny",
                    },
                    {
                        "model": "m",
                        "condition": "conflicting",
                        "gold_label": "support",
                        "metric_label": "comment",
                        "predicted_label": "comment",
                    },
                    {
                        "model": "m",
                        "condition": "conflicting",
                        "gold_label": "deny",
                        "metric_label": "deny",
                        "predicted_label": "deny",
                    },
                ],
            )
            payload = evaluate(predictions, tmp_path / "tables")
            self.assertTrue((tmp_path / "tables" / "summary_metrics.csv").exists())
            self.assertTrue((tmp_path / "tables" / "context_gaps.csv").exists())
            useful = next(row for row in payload["summary"] if row["condition"] == "useful")
            self.assertEqual(useful["accuracy"], 1.0)
            self.assertGreater(payload["context_gaps"][0]["macro_f1_drop"], 0)


if __name__ == "__main__":
    unittest.main()

