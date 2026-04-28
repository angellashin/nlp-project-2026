from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Union


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, data: Any) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: PathLike) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: PathLike, rows: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    ensure_dir(output.parent)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count
