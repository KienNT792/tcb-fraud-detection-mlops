from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REGISTRY_METADATA_FILENAME = "registry_metadata.json"


def registry_metadata_path(models_dir: str | Path) -> Path:
    return Path(models_dir) / REGISTRY_METADATA_FILENAME


def write_registry_metadata(
    models_dir: str | Path,
    payload: dict[str, Any],
) -> Path:
    path = registry_metadata_path(models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return path


def read_registry_metadata(
    models_dir: str | Path,
) -> dict[str, Any]:
    path = registry_metadata_path(models_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
