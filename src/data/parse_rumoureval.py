from __future__ import annotations

import argparse
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from src.common.io import ensure_dir, read_json, write_json, write_jsonl
from src.common.text import clean_text


EXPECTED_REPLY_COUNTS = {"train": 4890, "dev": 1447, "test": 1746}


def find_dataset_root(raw_dir: Path) -> Path:
    raw_dir = raw_dir.resolve()
    direct = raw_dir / "rumoureval2019"
    if (direct / "rumoureval-2019-training-data.zip").exists():
        return direct
    if (raw_dir / "rumoureval-2019-training-data.zip").exists():
        return raw_dir

    archive = raw_dir / "rumoureval2019.tar.bz2"
    if archive.exists():
        with tarfile.open(archive, "r:bz2") as tar:
            safe_extract_tar(tar, raw_dir)
        if (direct / "rumoureval-2019-training-data.zip").exists():
            return direct

    raise FileNotFoundError(
        "Could not find RumourEval files. Expected rumoureval2019.tar.bz2 "
        "or extracted rumoureval2019/ under "
        f"{raw_dir}"
    )


def safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if destination not in [member_path, *member_path.parents]:
            raise ValueError(f"Unsafe path in tar archive: {member.name}")
    tar.extractall(destination)


def extract_zip_once(zip_path: Path, extract_dir: Path) -> Path:
    ensure_dir(extract_dir)
    marker = extract_dir / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        marker.write_text(zip_path.name, encoding="utf-8")
    return extract_dir


def prepare_dataset(raw_dir: Path) -> dict[str, Path]:
    root = find_dataset_root(raw_dir)
    train_extract = extract_zip_once(
        root / "rumoureval-2019-training-data.zip", root / "extracted-training"
    )
    test_extract = extract_zip_once(
        root / "rumoureval-2019-test-data.zip", root / "extracted-test"
    )
    return {
        "root": root,
        "train_data": train_extract / "rumoureval-2019-training-data",
        "test_data": test_extract / "rumoureval-2019-test-data",
        "train_key": train_extract / "rumoureval-2019-training-data" / "train-key.json",
        "dev_key": train_extract / "rumoureval-2019-training-data" / "dev-key.json",
        "test_key": root / "final-eval-key.json",
    }


def walk_structure(
    subtree: dict[str, Any],
    source_id: str,
    parent_id: Optional[str],
    depth: int,
    rows: dict[str, dict[str, Any]],
) -> None:
    for node_id, children in subtree.items():
        node_id = str(node_id)
        child_ids = list(children.keys()) if isinstance(children, dict) else []
        rows[node_id] = {
            "source_id": source_id,
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": depth,
            "children_ids": [str(child) for child in child_ids],
        }
        if isinstance(children, dict):
            walk_structure(children, source_id, node_id, depth + 1, rows)


def parse_structure(path: Path) -> dict[str, dict[str, Any]]:
    structure = read_json(path)
    rows: dict[str, dict[str, Any]] = {}
    for source_id, children in structure.items():
        source_id = str(source_id)
        child_ids = list(children.keys()) if isinstance(children, dict) else []
        rows[source_id] = {
            "source_id": source_id,
            "node_id": source_id,
            "parent_id": None,
            "depth": 0,
            "children_ids": [str(child) for child in child_ids],
        }
        if isinstance(children, dict):
            walk_structure(children, source_id, source_id, 1, rows)
    return rows


def lineage(node_id: str, parent_by_id: dict[str, Optional[str]]) -> list[str]:
    ancestors: list[str] = []
    parent = parent_by_id.get(node_id)
    while parent:
        ancestors.append(parent)
        parent = parent_by_id.get(parent)
    ancestors.reverse()
    return ancestors


def platform_event_thread(split_root: Path, thread_dir: Path) -> Tuple[str, Optional[str], str]:
    rel_parts = thread_dir.relative_to(split_root).parts
    if rel_parts[0].startswith("twitter"):
        event = rel_parts[1] if len(rel_parts) >= 3 else None
        thread_id = rel_parts[2] if len(rel_parts) >= 3 else rel_parts[-1]
        return rel_parts[0], event, thread_id
    return rel_parts[0], None, rel_parts[-1]


def read_node_json(thread_dir: Path, node_id: str, is_source: bool) -> dict[str, Any]:
    subdir = "source-tweet" if is_source else "replies"
    path = thread_dir / subdir / f"{node_id}.json"
    if not path.exists():
        return {}
    return read_json(path)


def extract_text(payload: dict[str, Any], is_source: bool) -> str:
    if not payload:
        return ""

    if "text" in payload:
        return clean_text(payload.get("text"))

    data = payload.get("data")
    if isinstance(data, dict):
        children = data.get("children")
        if isinstance(children, list) and children:
            first_child = children[0]
            if isinstance(first_child, dict):
                child_data = first_child.get("data", {})
                if isinstance(child_data, dict):
                    title = clean_text(child_data.get("title"))
                    selftext = clean_text(child_data.get("selftext"))
                    body = clean_text(child_data.get("body"))
                    return clean_text(f"{title} {selftext} {body}")

    data = payload.get("data", payload)
    if is_source:
        return clean_text(f"{data.get('title', '')} {data.get('selftext', '')}")
    return clean_text(data.get("body") or data.get("text") or "")


def extract_author(payload: dict[str, Any]) -> Optional[str]:
    data = payload.get("data", payload)
    if isinstance(data, dict) and "children" in data:
        children = data.get("children")
        if isinstance(children, list) and children:
            first_child = children[0]
            if isinstance(first_child, dict):
                data = first_child.get("data", {})
    author = data.get("author") if isinstance(data, dict) else None
    return clean_text(author) or None


def extract_created_at(payload: dict[str, Any]) -> Optional[Union[str, float]]:
    data = payload.get("data", payload)
    if isinstance(data, dict) and "children" in data:
        children = data.get("children")
        if isinstance(children, list) and children:
            first_child = children[0]
            if isinstance(first_child, dict):
                data = first_child.get("data", {})
    if not isinstance(data, dict):
        return None
    return data.get("created_at") or data.get("created_utc") or data.get("created")


def parse_split(split: str, split_root: Path, labels: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_nodes: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []

    for structure_path in sorted(split_root.rglob("structure.json")):
        thread_dir = structure_path.parent
        platform, event, thread_id = platform_event_thread(split_root, thread_dir)
        structure_rows = parse_structure(structure_path)
        parent_by_id = {node_id: row["parent_id"] for node_id, row in structure_rows.items()}
        text_by_id: dict[str, str] = {}
        author_by_id: dict[str, Optional[str]] = {}
        source_id = next(iter(structure_rows.values()))["source_id"]

        for node_id, row in structure_rows.items():
            is_source = node_id == source_id
            payload = read_node_json(thread_dir, node_id, is_source)
            extracted_text = extract_text(payload, is_source)
            text_by_id[node_id] = extracted_text or "[missing text]"
            author_by_id[node_id] = extract_author(payload)
            node = {
                "split": split,
                "platform": platform,
                "event": event,
                "thread_id": thread_id,
                "source_id": source_id,
                "node_id": node_id,
                "parent_id": row["parent_id"],
                "depth": row["depth"],
                "text": text_by_id[node_id],
                "text_missing": not bool(extracted_text),
                "label": labels.get(node_id),
                "is_source_target": is_source,
                "children_ids": row["children_ids"],
                "ancestor_ids": lineage(node_id, parent_by_id),
                "author": author_by_id[node_id],
                "created_at": extract_created_at(payload),
            }
            all_nodes.append(node)

        for node_id, row in structure_rows.items():
            if node_id == source_id or node_id not in labels:
                continue
            parent_id = row["parent_id"]
            siblings = [
                sibling
                for sibling, sibling_row in structure_rows.items()
                if sibling != node_id and sibling_row["parent_id"] == parent_id
            ]
            non_source_thread_ids = [
                candidate for candidate in structure_rows if candidate != source_id and candidate != node_id
            ]
            targets.append(
                {
                    "split": split,
                    "platform": platform,
                    "event": event,
                    "thread_id": thread_id,
                    "source_id": source_id,
                    "target_id": node_id,
                    "parent_id": parent_id,
                    "depth": row["depth"],
                    "label": labels[node_id],
                    "target_text": text_by_id[node_id],
                    "source_text": text_by_id.get(source_id, ""),
                    "parent_text": text_by_id.get(parent_id, "") if parent_id else "",
                    "ancestor_ids": lineage(node_id, parent_by_id),
                    "sibling_ids": siblings,
                    "same_thread_candidate_ids": non_source_thread_ids,
                    "is_source_target": False,
                    "author": author_by_id.get(node_id),
                }
            )

    return all_nodes, targets


def label_map(path: Path) -> dict[str, str]:
    data = read_json(path)
    return {str(k): v for k, v in data["subtaskaenglish"].items()}


def validate_targets(split: str, targets: list[dict[str, Any]]) -> None:
    for target in targets:
        if target["is_source_target"]:
            raise AssertionError(f"Source target leaked into {split}: {target['target_id']}")
        for field in ["target_text", "label", "source_text", "thread_id"]:
            if not target.get(field):
                raise AssertionError(f"Missing {field} for {split}:{target['target_id']}")


def write_outputs(raw_dir: Path, out_dir: Path) -> dict[str, Any]:
    paths = prepare_dataset(raw_dir)
    ensure_dir(out_dir)
    split_specs = {
        "train": (paths["train_data"], label_map(paths["train_key"])),
        "dev": (paths["train_data"], label_map(paths["dev_key"])),
        "test": (paths["test_data"], label_map(paths["test_key"])),
    }
    summary: dict[str, Any] = {"splits": {}}

    for split, (split_root, labels) in split_specs.items():
        nodes, targets = parse_split(split, split_root, labels)
        validate_targets(split, targets)
        write_jsonl(out_dir / f"nodes_{split}.jsonl", nodes)
        write_jsonl(out_dir / f"reply_targets_{split}.jsonl", targets)
        counts = Counter(target["label"] for target in targets)
        summary["splits"][split] = {
            "nodes": len(nodes),
            "reply_targets": len(targets),
            "expected_reply_targets": EXPECTED_REPLY_COUNTS.get(split),
            "label_counts": dict(sorted(counts.items())),
            "platform_counts": dict(Counter(target["platform"] for target in targets)),
            "thread_count": len({target["thread_id"] for target in targets}),
        }
        if split in EXPECTED_REPLY_COUNTS and len(targets) != EXPECTED_REPLY_COUNTS[split]:
            raise AssertionError(
                f"{split} reply count mismatch: {len(targets)} != {EXPECTED_REPLY_COUNTS[split]}"
            )

    write_json(out_dir / "dataset_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse RumourEval into reply-level JSONL tables.")
    parser.add_argument("--raw-dir", default="data/raw/rumoureval2019")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    summary = write_outputs(Path(args.raw_dir), Path(args.out_dir))
    for split, stats in summary["splits"].items():
        print(f"{split}: {stats['reply_targets']} replies {stats['label_counts']}")


if __name__ == "__main__":
    main()
