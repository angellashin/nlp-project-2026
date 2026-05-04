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
                        "target_id": "t1",
                        "platform": "twitter",
                        "depth": 2,
                        "depth_bucket": "depth_2plus",
                        "parent_available": True,
                        "context_source": "same_thread",
                        "mixed_valid": False,
                        "condition": "useful",
                        "gold_label": "support",
                        "metric_label": "support",
                        "predicted_label": "support",
                    },
                    {
                        "model": "m",
                        "target_id": "t2",
                        "platform": "twitter",
                        "depth": 2,
                        "depth_bucket": "depth_2plus",
                        "parent_available": True,
                        "context_source": "same_thread",
                        "mixed_valid": False,
                        "condition": "useful",
                        "gold_label": "deny",
                        "metric_label": "deny",
                        "predicted_label": "deny",
                    },
                    {
                        "model": "m",
                        "target_id": "t1",
                        "platform": "twitter",
                        "depth": 2,
                        "depth_bucket": "depth_2plus",
                        "parent_available": True,
                        "context_source": "same_thread",
                        "mixed_valid": False,
                        "condition": "conflicting",
                        "gold_label": "support",
                        "metric_label": "comment",
                        "predicted_label": "comment",
                    },
                    {
                        "model": "m",
                        "target_id": "t2",
                        "platform": "twitter",
                        "depth": 2,
                        "depth_bucket": "depth_2plus",
                        "parent_available": True,
                        "context_source": "same_thread",
                        "mixed_valid": False,
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
            self.assertTrue((tmp_path / "tables" / "summary_by_platform.csv").exists())
            self.assertTrue((tmp_path / "tables" / "context_gaps_by_platform.csv").exists())
            self.assertTrue((tmp_path / "tables" / "summary_by_depth_bucket.csv").exists())
            self.assertTrue((tmp_path / "tables" / "summary_by_context_source.csv").exists())
            self.assertTrue((tmp_path / "tables" / "summary_by_parent_available.csv").exists())
            self.assertTrue((tmp_path / "tables" / "summary_by_validity_subset.csv").exists())
            self.assertTrue((tmp_path / "tables" / "predicted_label_distribution.csv").exists())
            self.assertTrue((tmp_path / "tables" / "paired_flip_rates.csv").exists())
            self.assertTrue((tmp_path / "tables" / "paired_flip_cases.csv").exists())
            useful = next(row for row in payload["summary"] if row["condition"] == "useful")
            self.assertEqual(useful["accuracy"], 1.0)
            self.assertGreater(payload["context_gaps"][0]["macro_f1_drop"], 0)
            self.assertIn("platform", payload)
            distribution_count = sum(
                int(row["count"])
                for row in payload["predicted_label_distribution"]
                if row["model"] == "m" and row["condition"] == "useful"
            )
            self.assertEqual(distribution_count, 2)


if __name__ == "__main__":
    unittest.main()
