import hashlib
import json
from pathlib import Path


def cache_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class DiskCache:
    """One JSON file per key under outputs/cache/. The same frames + prompt never cost a second call."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        path = self._dir / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, key: str, value: dict) -> None:
        (self._dir / f"{key}.json").write_text(json.dumps(value), encoding="utf-8")
