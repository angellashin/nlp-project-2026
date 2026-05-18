import tempfile
import unittest
from pathlib import Path

from src.common.io import write_jsonl
from src.experiments.evaluate_sensitivity import evaluate_sensitivity


def prediction(target_id, condition, gold, pred, gold_score):
    scores = {
        "support": {"mean_logprob": -3.0, "sum_logprob": -3.0, "token_count": 1},
        "deny": {"mean_logprob": -3.0, "sum_logprob": -3.0, "token_count": 1},
        "query": {"mean_logprob": -3.0, "sum_logprob": -3.0, "token_count": 1},
        "comment": {"mean_logprob": -3.0, "sum_logprob": -3.0, "token_count": 1},
    }
    scores[gold] = {"mean_logprob": gold_score, "sum_logprob": gold_score, "token_count": 1}
    return {
        "model": "m",
        "example_id": f"dev:{target_id}:{condition}",
        "target_id": target_id,
        "condition": condition,
        "platform": "twitter",
        "depth": 2,
        "depth_bucket": "depth_2plus",
        "parent_available": True,
        "mixed_valid": condition == "mixed",
        "context_source": "same_thread" if condition != "reply_only" else "none",
        "gold_label": gold,
        "predicted_label": pred,
        "metric_label": pred,
        "label_scores": scores,
    }


class SensitivityEvaluationTest(unittest.TestCase):
    def test_writes_score_summary_and_pair_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            predictions = tmp_path / "predictions.jsonl"
            write_jsonl(
                predictions,
                [
                    prediction("t1", "useful", "support", "support", -0.1),
                    prediction("t1", "conflicting", "support", "deny", -2.0),
                    prediction("t2", "useful", "deny", "deny", -0.2),
                    prediction("t2", "conflicting", "deny", "deny", -0.8),
                ],
            )
            payload = evaluate_sensitivity(predictions, tmp_path / "tables", "mean_logprob", 100, 461)
            self.assertTrue((tmp_path / "tables" / "score_summary.csv").exists())
            self.assertTrue((tmp_path / "tables" / "score_gaps.csv").exists())
            useful_to_conflicting = next(
                row
                for row in payload["gaps"]
                if row["from_condition"] == "useful" and row["to_condition"] == "conflicting"
            )
            self.assertEqual(useful_to_conflicting["n"], 2)
            self.assertGreater(useful_to_conflicting["mean_gold_score_drop"], 0)
            self.assertGreater(useful_to_conflicting["mean_margin_drop"], 0)


if __name__ == "__main__":
    unittest.main()
