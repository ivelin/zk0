"""Record writing: eval records and state logs (spec §3, §5).

Every record the package emits is JSON, carries the runtime manifest
hash (so a result traces to the exact stack that produced it), and is
content-hashed (SHA-256 over the canonical form). Node-key signing of
records lands with the Stage 1 identity work; the content hash is
the tamper-evident anchor until then, and the record says so.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def package_root() -> Path:
    """The node/ package directory (this file lives in src/zk0_node/)."""
    return Path(__file__).resolve().parents[2]


def runtime_manifest_path() -> Path:
    return package_root() / "runtime-manifest.json"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonable(obj):
    """Recursively convert numpy scalars/arrays and tuples to JSON values."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover — numpy is a runtime pin
        np = None
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def content_sha256(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "content_sha256"}
    canonical = json.dumps(jsonable(body), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_record(record: dict, path: Path) -> Path:
    record = dict(record)
    record["content_sha256"] = content_sha256(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(record), indent=1, sort_keys=True))
    return path


def manifest_sha256(manifest_path: Path) -> str:
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
