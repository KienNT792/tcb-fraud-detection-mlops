from __future__ import annotations

import sys
from pathlib import Path

import yaml


REQUIRED_KEYS = {
    "Deployment": {"apiVersion", "kind", "metadata", "spec"},
    "Service": {"apiVersion", "kind", "metadata", "spec"},
    "Ingress": {"apiVersion", "kind", "metadata", "spec"},
    "ConfigMap": {"apiVersion", "kind", "metadata", "data"},
    "Secret": {"apiVersion", "kind", "metadata", "stringData"},
    "HorizontalPodAutoscaler": {"apiVersion", "kind", "metadata", "spec"},
}


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for doc_index, doc in enumerate(yaml.safe_load_all(handle), start=1):
            if not doc:
                continue
            kind = doc.get("kind")
            if kind not in REQUIRED_KEYS:
                errors.append(f"{path}: document {doc_index} has unsupported kind '{kind}'")
                continue
            missing = REQUIRED_KEYS[kind] - set(doc.keys())
            if missing:
                errors.append(f"{path}: document {doc_index} missing keys {sorted(missing)}")
    return errors


def main() -> None:
    k8s_dir = Path("k8s")
    files = sorted(k8s_dir.glob("*.yaml"))
    errors: list[str] = []
    for file_path in files:
        errors.extend(validate_file(file_path))

    if errors:
        print("\n".join(errors))
        raise SystemExit(1)

    print(f"Validated {len(files)} Kubernetes manifest file(s).")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc}. Install PyYAML to validate manifests.", file=sys.stderr)
        raise
