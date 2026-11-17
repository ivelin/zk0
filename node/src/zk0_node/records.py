"""Record writing: eval records and state logs (spec §3, §5).

Every record the package emits is JSON, carries the runtime manifest
hash (so a result traces to the exact stack that produced it), and is
content-hashed (SHA-256 over the canonical form). Build slice 2 adds
node-key signing on top: the signature signs the content hash with
the node's Ed25519 key (identity.py), so a record is both
tamper-evident (hash) and attributable (signature). The content
hash excludes the `content_sha256` and `signature` keys themselves;
slice-1 records carry no signature key, so their hashes recompute
identically under this rule.
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
    body = {k: v for k, v in record.items()
            if k not in ("content_sha256", "signature")}
    canonical = json.dumps(jsonable(body), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_record(record: dict, path: Path) -> Path:
    record = dict(record)
    record["content_sha256"] = content_sha256(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(record), indent=1, sort_keys=True))
    return path


def sign_record(record: dict, key) -> dict:
    """Attach the node-key signature over the record's content hash.

    The record must already carry its content_sha256 (write_record
    computes it); signing is a separate, explicit act so unsigned
    records remain possible where the spec allows them.
    """
    digest = record.get("content_sha256")
    if not digest:
        raise ValueError("record has no content_sha256 — hash it first "
                         "(records.write_record)")
    record = dict(record)
    record["signature"] = {
        "alg": "ed25519",
        "signed_field": "content_sha256",
        "public_key_hex": key.public_key_hex,
        "key_fingerprint": key.fingerprint,
        "sig_hex": key.sign_hex(bytes.fromhex(digest)),
    }
    return record


def write_signed_record(record: dict, path: Path, key) -> Path:
    """Hash, sign, and write a record in one step (the slice-2 default)."""
    record = dict(record)
    record["content_sha256"] = content_sha256(record)
    record = sign_record(record, key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(record), indent=1, sort_keys=True))
    return path


def verify_record(record: dict) -> dict:
    """Verify a record's content hash and (if present) its signature.

    Returns {hash_ok, signature_present, signature_ok, signer}.
    Verification needs no key material beyond what the record
    itself carries (the signer's public key rides in the record).
    """
    from .identity import NodeKey

    out = {"hash_ok": False, "signature_present": False,
           "signature_ok": False, "signer": None}
    stored = record.get("content_sha256")
    if stored and content_sha256(record) == stored:
        out["hash_ok"] = True
    sig = record.get("signature")
    if isinstance(sig, dict) and sig.get("alg") == "ed25519":
        out["signature_present"] = True
        out["signer"] = sig.get("key_fingerprint")
        if out["hash_ok"]:
            out["signature_ok"] = NodeKey.verify(
                sig["public_key_hex"], bytes.fromhex(stored),
                sig["sig_hex"])
    return out


def manifest_sha256(manifest_path: Path) -> str:
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
