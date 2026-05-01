import unittest

from src.data.build_context_variants import build_indexes, build_variants_for_target


class ContextVariantTest(unittest.TestCase):
    def test_builds_all_conditions_without_target_leakage(self):
        nodes = [
            {
                "split": "dev",
                "platform": "twitter-english",
                "event": "event",
                "thread_id": "t1",
                "source_id": "s",
                "node_id": "s",
                "parent_id": None,
                "depth": 0,
                "text": "Source claim",
                "label": "support",
                "is_source_target": True,
                "children_ids": ["p", "c1", "d1"],
                "ancestor_ids": [],
                "author": "a",
            },
            {
                "split": "dev",
                "platform": "twitter-english",
                "event": "event",
                "thread_id": "t1",
                "source_id": "s",
                "node_id": "p",
                "parent_id": "s",
                "depth": 1,
                "text": "Parent asks for source",
                "label": "query",
                "is_source_target": False,
                "children_ids": ["target"],
                "ancestor_ids": ["s"],
                "author": "b",
            },
            {
                "split": "dev",
                "platform": "twitter-english",
                "event": "event",
                "thread_id": "t1",
                "source_id": "s",
                "node_id": "target",
                "parent_id": "p",
                "depth": 2,
                "text": "I agree this happened",
                "label": "support",
                "is_source_target": False,
                "children_ids": [],
                "ancestor_ids": ["s", "p"],
                "author": "c",
            },
            {
                "split": "dev",
                "platform": "twitter-english",
                "event": "event",
                "thread_id": "t1",
                "source_id": "s",
                "node_id": "c1",
                "parent_id": "s",
                "depth": 1,
                "text": "Random side comment",
                "label": "comment",
                "is_source_target": False,
                "children_ids": [],
                "ancestor_ids": ["s"],
                "author": "d",
            },
            {
                "split": "dev",
                "platform": "twitter-english",
                "event": "event",
                "thread_id": "t1",
                "source_id": "s",
                "node_id": "d1",
                "parent_id": "s",
                "depth": 1,
                "text": "No, this is false",
                "label": "deny",
                "is_source_target": False,
                "children_ids": [],
                "ancestor_ids": ["s"],
                "author": "e",
            },
        ]
        target = {
            "split": "dev",
            "platform": "twitter-english",
            "event": "event",
            "thread_id": "t1",
            "source_id": "s",
            "target_id": "target",
            "parent_id": "p",
            "depth": 2,
            "label": "support",
            "target_text": "I agree this happened",
            "source_text": "Source claim",
            "parent_text": "Parent asks for source",
            "ancestor_ids": ["s", "p"],
            "is_source_target": False,
            "author": "c",
        }
        variants = build_variants_for_target(target, build_indexes(nodes), seed=461)
        self.assertEqual({row["condition"] for row in variants}, {
            "reply_only",
            "useful",
            "irrelevant",
            "conflicting",
            "mixed",
        })
        for variant in variants:
            self.assertNotIn("target", [item["node_id"] for item in variant["context_items"]])
            self.assertIn("Target reply:", variant["prompt_text"])
            self.assertEqual(variant["platform"], "twitter-english")
            self.assertEqual(variant["depth"], 2)
        conflict = next(row for row in variants if row["condition"] == "conflicting")
        conflict_labels = [
            item["label_if_available"]
            for item in conflict["context_items"]
            if item["role"] == "conflicting_reply"
        ]
        self.assertEqual(conflict_labels, ["deny"])


if __name__ == "__main__":
    unittest.main()
