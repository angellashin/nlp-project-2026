from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load JSON or YAML config.

    PyYAML is optional. The committed YAML file is intentionally simple, but if
    PyYAML is unavailable we still support JSON-formatted config files.
    """
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for YAML configs. Install dependencies from "
            "requirements.txt or pass a JSON config instead."
        ) from exc

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded
