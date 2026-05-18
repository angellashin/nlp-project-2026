import tempfile
import unittest
from pathlib import Path

from src.common.io import read_jsonl, write_jsonl
from src.data.build_challenge_sets import build_challenge_sets


def variant(target_id, condition, label="support", context_source="same_thread", parent=True, conflict=True):
    return {
        "example_id": f"dev:{target_id}:{condition}",
        "split": "dev",
        "target_id": target_id,
        "condition": condition,
        "label": label,
        "context_source": "none" if condition == "reply_only" else context_source,
        "parent_available": parent,
        "mixed_valid": condition == "mixed" and parent,
        "has_conflicting_reply": condition in {"conflicting", "mixed"} and conflict,
        "depth_bucket": "depth_2plus" if parent else "depth_1",
        "platform": "twitter",
    }


class ChallengeSetTest(unittest.TestCase):
    def test_builds_strict_and_balanced_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            variants_path = tmp_path / "dev_context_variants.jsonl"
            rows = []
            for condition in ["reply_only", "useful", "irrelevant", "conflicting", "mixed"]:
                rows.append(variant("t1", condition, label="support"))
                rows.append(variant("t2", condition, label="deny", parent=False))
                rows.append(variant("t3", condition, label="query", context_source="same_event_fallback"))
            write_jsonl(variants_path, rows)

            summary = build_challenge_sets(
                variants_path,
                tmp_path / "challenge",
                "dev",
                ["complete", "strict", "strict_balanced"],
                ["reply_only", "useful", "irrelevant", "conflicting", "mixed"],
                None,
            )

            self.assertEqual(summary["presets"]["complete"]["targets"], 3)
            self.assertEqual(summary["presets"]["strict"]["targets"], 1)
            strict_rows = read_jsonl(tmp_path / "challenge" / "dev_strict_variants.jsonl")
            self.assertEqual(len(strict_rows), 5)
            self.assertEqual({row["target_id"] for row in strict_rows}, {"t1"})
            balanced_rows = read_jsonl(tmp_path / "challenge" / "dev_strict_balanced_variants.jsonl")
            self.assertEqual(len(balanced_rows), 5)


if __name__ == "__main__":
    unittest.main()
