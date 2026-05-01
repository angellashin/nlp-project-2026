from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple

from src.common.io import ensure_dir, read_jsonl, write_json, write_jsonl
from src.common.text import truncate_words


LABELS = ["support", "deny", "query", "comment"]
CONDITIONS = ["reply_only", "useful", "irrelevant", "conflicting", "mixed"]
PREFERRED_CONFLICTS = {
    "support": ["deny", "query", "comment"],
    "deny": ["support", "query", "comment"],
    "query": ["support", "deny", "comment"],
    "comment": ["deny", "support", "query"],
}


def stable_sample(items: list[dict[str, Any]], k: int, seed: int, key: str) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{key}")
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled[:k]


def descendants(node_id: str, child_map: dict[str, list[str]]) -> set[str]:
    todo = list(child_map.get(node_id, []))
    seen: set[str] = set()
    while todo:
        child = todo.pop()
        if child in seen:
            continue
        seen.add(child)
        todo.extend(child_map.get(child, []))
    return seen


def node_context_item(node: dict[str, Any], role: str, relation: str) -> dict[str, Any]:
    return {
        "role": role,
        "node_id": node["node_id"],
        "text": node["text"],
        "label_if_available": node.get("label"),
        "relation": relation,
    }


def render_prompt(target: dict[str, Any], condition: str, context_items: list[dict[str, Any]]) -> str:
    lines = [
        "Classify the stance of the target reply toward the source rumour.",
        "Choose exactly one label: support, deny, query, comment.",
        "",
        "Label meanings:",
        "- support: the target supports or agrees with the rumour.",
        "- deny: the target rejects, refutes, or disagrees with the rumour.",
        "- query: the target asks for clarification, evidence, or more information.",
        "- comment: the target is related but does not clearly support, deny, or query.",
        "",
        f"Context condition: {condition}",
    ]
    if context_items:
        lines.append("Context:")
        for item in context_items:
            role = item["role"].replace("_", " ").title()
            text = truncate_words(item["text"], 120)
            lines.append(f"[{role}] {text}")
    else:
        lines.append("Context: none")

    lines.extend(
        [
            "",
            f"Target reply: {truncate_words(target['target_text'], 120)}",
            "",
            "Answer with one label only.",
        ]
    )
    return "\n".join(lines)


def build_indexes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {node["node_id"]: node for node in nodes}
    by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_event: dict[Tuple[str, Optional[str]], list[dict[str, Any]]] = defaultdict(list)
    child_map: dict[str, list[str]] = {}
    for node in nodes:
        by_thread[node["thread_id"]].append(node)
        by_event[(node["platform"], node.get("event"))].append(node)
        child_map[node["node_id"]] = node.get("children_ids", [])
    return {"by_id": by_id, "by_thread": by_thread, "by_event": by_event, "child_map": child_map}


def target_path_ids(target: dict[str, Any]) -> set[str]:
    ids = set(target.get("ancestor_ids", []))
    ids.add(target["target_id"])
    if target.get("parent_id"):
        ids.add(target["parent_id"])
    return ids


def useful_items(target: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    source = by_id.get(target["source_id"])
    if source:
        items.append(node_context_item(source, "source", "source"))
    parent_id = target.get("parent_id")
    if parent_id and parent_id != target["source_id"] and parent_id in by_id:
        items.append(node_context_item(by_id[parent_id], "parent", "direct_parent"))
    return items


def irrelevant_candidates(
    target: dict[str, Any],
    by_thread: dict[str, list[dict[str, Any]]],
    by_event: dict[Tuple[str, Optional[str]], list[dict[str, Any]]],
    child_map: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], str]:
    path_ids = target_path_ids(target) | descendants(target["target_id"], child_map)
    candidates = [
        node
        for node in by_thread[target["thread_id"]]
        if not node["is_source_target"]
        and node["node_id"] not in path_ids
        and node.get("text")
        and node.get("author") != target.get("author")
    ]
    comments = [node for node in candidates if node.get("label") == "comment"]
    if comments:
        return comments, "same_thread_comment_non_path"
    if candidates:
        return candidates, "same_thread_non_path_no_comment_available"

    event_key = (target["platform"], target.get("event"))
    if target.get("event") is not None:
        same_event_comments = [
            node
            for node in by_event.get(event_key, [])
            if not node["is_source_target"]
            and node["thread_id"] != target["thread_id"]
            and node.get("label") == "comment"
            and node.get("text")
        ]
        if same_event_comments:
            return same_event_comments, "same_event_fallback_comment"

    return [], "no_irrelevant_candidate_available"


def conflict_candidates(
    target: dict[str, Any],
    by_thread: dict[str, list[dict[str, Any]]],
    by_event: dict[Tuple[str, Optional[str]], list[dict[str, Any]]],
    child_map: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], str]:
    preferred = PREFERRED_CONFLICTS.get(target["label"], [label for label in LABELS if label != target["label"]])
    path_ids = target_path_ids(target) | descendants(target["target_id"], child_map)

    def usable(node: dict[str, Any], allow_path: bool, same_thread: bool) -> bool:
        if node["is_source_target"] or node["node_id"] == target["target_id"] or not node.get("text"):
            return False
        if same_thread and not allow_path and node["node_id"] in path_ids:
            return False
        return node.get("label") in preferred

    same_thread_non_path = [
        node for node in by_thread[target["thread_id"]] if usable(node, allow_path=False, same_thread=True)
    ]
    if same_thread_non_path:
        return same_thread_non_path, "same_thread_non_path_conflict"

    same_thread_any = [
        node for node in by_thread[target["thread_id"]] if usable(node, allow_path=True, same_thread=True)
    ]
    if same_thread_any:
        return same_thread_any, "same_thread_any_conflict"

    event_key = (target["platform"], target.get("event"))
    same_event = [
        node
        for node in by_event.get(event_key, [])
        if node["thread_id"] != target["thread_id"] and usable(node, allow_path=True, same_thread=False)
    ]
    if same_event:
        return same_event, "same_event_fallback_conflict"

    same_platform, global_pool = cross_thread_conflict_candidates(target, by_event, usable)
    if same_platform:
        return same_platform, "same_platform_fallback_conflict"
    if global_pool:
        return global_pool, "global_fallback_conflict"

    any_different = [
        node
        for node in by_thread[target["thread_id"]]
        if not node["is_source_target"]
        and node["node_id"] != target["target_id"]
        and node.get("label")
        and node.get("label") != target["label"]
        and node.get("text")
    ]
    return any_different, "same_thread_any_different_label"


def cross_thread_conflict_candidates(
    target: dict[str, Any],
    by_event: dict[Tuple[str, Optional[str]], list[dict[str, Any]]],
    usable,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen_ids: set[str] = set()
    same_platform: list[dict[str, Any]] = []
    global_pool: list[dict[str, Any]] = []
    for (platform, _event), nodes in by_event.items():
        for node in nodes:
            if node["node_id"] in seen_ids or node["thread_id"] == target["thread_id"]:
                continue
            seen_ids.add(node["node_id"])
            if not usable(node, allow_path=True, same_thread=False):
                continue
            if platform == target["platform"]:
                same_platform.append(node)
            global_pool.append(node)
    return same_platform, global_pool


def mixed_conflict_candidates(
    target: dict[str, Any],
    conflict_pool: list[dict[str, Any]],
    conflict_note: str,
    by_event: dict[Tuple[str, Optional[str]], list[dict[str, Any]]],
    used_ids: set[str],
) -> tuple[list[dict[str, Any]], str]:
    preferred = PREFERRED_CONFLICTS.get(target["label"], [label for label in LABELS if label != target["label"]])

    non_overlap = [node for node in conflict_pool if node["node_id"] not in used_ids]
    if non_overlap:
        return non_overlap, f"{conflict_note}_not_in_useful"

    def usable(node: dict[str, Any], allow_path: bool, same_thread: bool) -> bool:
        return (
            not node["is_source_target"]
            and node["node_id"] != target["target_id"]
            and node["node_id"] not in used_ids
            and node.get("text")
            and node.get("label") in preferred
        )

    same_platform, global_pool = cross_thread_conflict_candidates(target, by_event, usable)
    if same_platform:
        return same_platform, "same_platform_fallback_conflict_not_in_useful"
    if global_pool:
        return global_pool, "global_fallback_conflict_not_in_useful"
    return [], "no_non_overlapping_conflict_available"


def source_from_note(note: str) -> str:
    if note.startswith("same_event"):
        return "same_event_fallback"
    if note.startswith("same_platform") or note.startswith("global"):
        return "cross_thread_fallback"
    if note.startswith("same_thread"):
        return "same_thread"
    return "none"


def make_variant(
    target: dict[str, Any],
    condition: str,
    context_items: list[dict[str, Any]],
    context_source: str,
    notes: list[str],
) -> dict[str, Any]:
    prompt = render_prompt(target, condition, context_items)
    return {
        "example_id": f"{target['split']}:{target['target_id']}:{condition}",
        "split": target["split"],
        "platform": target.get("platform"),
        "event": target.get("event"),
        "target_id": target["target_id"],
        "thread_id": target["thread_id"],
        "source_id": target["source_id"],
        "parent_id": target.get("parent_id"),
        "depth": target.get("depth"),
        "label": target["label"],
        "condition": condition,
        "context_items": context_items,
        "context_source": context_source,
        "construction_notes": notes,
        "uses_gold_labels_for_stress_test": condition in {"conflicting", "mixed"},
        "prompt_text": prompt,
        "target_text": target["target_text"],
    }


def build_variants_for_target(target: dict[str, Any], indexes: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    by_id = indexes["by_id"]
    by_thread = indexes["by_thread"]
    by_event = indexes["by_event"]
    child_map = indexes["child_map"]
    variants: list[dict[str, Any]] = []

    variants.append(make_variant(target, "reply_only", [], "none", ["target_reply_only"]))

    useful = useful_items(target, by_id)
    useful_notes = ["source_included"]
    if target.get("parent_id") and target["parent_id"] != target["source_id"]:
        useful_notes.append("parent_included")
    variants.append(make_variant(target, "useful", useful, "same_thread", useful_notes))

    irrelevant_pool, irrelevant_note = irrelevant_candidates(target, by_thread, by_event, child_map)
    irrelevant = stable_sample(irrelevant_pool, 2, seed, f"{target['target_id']}:irrelevant")
    irrelevant_items = useful_items(target, by_id)[:1] + [
        node_context_item(node, "irrelevant_reply", irrelevant_note) for node in irrelevant
    ]
    variants.append(
        make_variant(
            target,
            "irrelevant",
            irrelevant_items,
            source_from_note(irrelevant_note) if irrelevant else "none",
            [irrelevant_note, f"irrelevant_count={len(irrelevant)}"],
        )
    )

    conflict_pool, conflict_note = conflict_candidates(target, by_thread, by_event, child_map)
    conflict = stable_sample(conflict_pool, 1, seed, f"{target['target_id']}:conflicting")
    conflict_items = useful_items(target, by_id)[:1] + [
        node_context_item(node, "conflicting_reply", conflict_note) for node in conflict
    ]
    variants.append(
        make_variant(
            target,
            "conflicting",
            conflict_items,
            source_from_note(conflict_note) if conflict else "none",
            [conflict_note, f"conflict_count={len(conflict)}"],
        )
    )

    mixed_items = useful_items(target, by_id)
    seen = {item["node_id"] for item in mixed_items}
    mixed_pool, mixed_note = mixed_conflict_candidates(target, conflict_pool, conflict_note, by_event, seen)
    mixed_conflict = stable_sample(mixed_pool, 1, seed, f"{target['target_id']}:mixed_conflicting")
    for node in mixed_conflict:
        if node["node_id"] not in seen:
            mixed_items.append(node_context_item(node, "conflicting_reply", mixed_note))
    variants.append(
        make_variant(
            target,
            "mixed",
            mixed_items,
            source_from_note(mixed_note) if mixed_conflict else "same_thread",
            [mixed_note, f"conflict_count={len(mixed_conflict)}", "useful_plus_conflict"],
        )
    )

    return variants


def validate_variant(variant: dict[str, Any]) -> None:
    target_id = variant["target_id"]
    for item in variant["context_items"]:
        if item["node_id"] == target_id:
            raise AssertionError(f"Target leaked into context: {variant['example_id']}")
    if variant["condition"] == "useful":
        roles = {item["role"] for item in variant["context_items"]}
        if "source" not in roles:
            raise AssertionError(f"Useful context missing source: {variant['example_id']}")
    if variant["condition"] in {"conflicting", "mixed"}:
        conflict_items = [item for item in variant["context_items"] if item["role"] == "conflicting_reply"]
        if not conflict_items:
            raise AssertionError(f"Missing conflicting context: {variant['example_id']}")
        for item in conflict_items:
            if item.get("label_if_available") == variant["label"]:
                raise AssertionError(f"Conflicting context has same label: {variant['example_id']}")


def build_split(processed_dir: Path, split: str, out_dir: Path, seed: int) -> dict[str, Any]:
    nodes = read_jsonl(processed_dir / f"nodes_{split}.jsonl")
    targets = read_jsonl(processed_dir / f"reply_targets_{split}.jsonl")
    indexes = build_indexes(nodes)
    variants: list[dict[str, Any]] = []
    for target in targets:
        variants.extend(build_variants_for_target(target, indexes, seed))

    for variant in variants:
        validate_variant(variant)

    write_jsonl(out_dir / f"{split}_context_variants.jsonl", variants)
    write_jsonl(out_dir / f"{split}_prompts_qwen.jsonl", variants)

    coverage: dict[str, Any] = {
        "split": split,
        "targets": len(targets),
        "variants": len(variants),
        "condition_counts": dict(Counter(v["condition"] for v in variants)),
        "condition_label_counts": {},
        "context_source_counts": {},
        "notes_counts": {},
    }
    for condition in CONDITIONS:
        condition_rows = [v for v in variants if v["condition"] == condition]
        coverage["condition_label_counts"][condition] = dict(Counter(v["label"] for v in condition_rows))
        coverage["context_source_counts"][condition] = dict(Counter(v["context_source"] for v in condition_rows))
        note_counter: Counter[str] = Counter()
        for row in condition_rows:
            note_counter.update(row["construction_notes"])
        coverage["notes_counts"][condition] = dict(note_counter)

    write_json(out_dir / f"{split}_coverage.json", coverage)
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled context variants for RumourEval.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/variants")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--seed", type=int, default=461)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    all_coverage = {}
    for split in args.splits:
        coverage = build_split(Path(args.processed_dir), split, out_dir, args.seed)
        all_coverage[split] = coverage
        print(f"{split}: {coverage['variants']} variants from {coverage['targets']} targets")
    write_json(out_dir / "coverage_summary.json", all_coverage)


if __name__ == "__main__":
    main()
