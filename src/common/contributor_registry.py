"""Contributor registry utilities for zk0 federated learning."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

REGISTRY_FILENAME = "contributor_registry.jsonl"


def build_contributor_record(
    node_id: Union[str, int],
    dataset_uri: str,
    *,
    timestamp: Optional[str] = None,
    source: str = "client",
    server_round: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a contributor registry record from node identity and dataset URI."""
    if not dataset_uri or not str(dataset_uri).strip():
        raise ValueError("dataset_uri is required for contributor registry records")

    record: Dict[str, Any] = {
        "node_id": str(node_id),
        "dataset_uri": str(dataset_uri).strip(),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "source": source,
    }
    if server_round is not None:
        record["server_round"] = int(server_round)
    return record


def get_registry_path(save_path: Union[str, Path]) -> Path:
    """Return the append-only contributor registry path for a run."""
    return Path(save_path) / REGISTRY_FILENAME


def append_contributor_record(
    registry_path: Union[str, Path], record: Dict[str, Any]
) -> None:
    """Append one contributor record to the registry JSONL artifact."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    logger.info(
        "Contributor registry: recorded node_id={} dataset_uri={} source={}",
        record.get("node_id"),
        record.get("dataset_uri"),
        record.get("source"),
    )


def extract_dataset_uri_from_context(context: Any) -> str:
    """Resolve dataset-uri from a Flower Context or context-like object."""
    node_config = getattr(context, "node_config", {}) or {}
    if node_config.get("dataset-uri"):
        return str(node_config["dataset-uri"])

    run_config = getattr(context, "run_config", {}) or {}
    if run_config.get("dataset.repo_id"):
        return str(run_config["dataset.repo_id"])
    if run_config.get("dataset.root"):
        return str(Path(run_config["dataset.root"]).name)

    import os

    if os.environ.get("DATASET_NAME"):
        return os.environ["DATASET_NAME"]

    raise ValueError(
        "No dataset source found for contributor registry. "
        f"node_config keys: {list(node_config.keys())}"
    )


def record_contributor_from_context(
    context: Any,
    *,
    source: str = "client",
    server_round: Optional[int] = None,
) -> Dict[str, Any]:
    """Build and optionally persist a contributor record from Flower Context."""
    node_id = getattr(context, "node_id", "unknown")
    dataset_uri = extract_dataset_uri_from_context(context)
    record = build_contributor_record(
        node_id=node_id,
        dataset_uri=dataset_uri,
        source=source,
        server_round=server_round,
    )

    run_config = getattr(context, "run_config", {}) or {}
    save_path = run_config.get("save_path")
    if save_path:
        append_contributor_record(get_registry_path(save_path), record)
    else:
        logger.info(
            "Contributor registry: built record without save_path (not persisted yet): {}",
            record,
        )
    return record


def lookup_dataset_from_client_round(
    save_path: Union[str, Path],
    server_round: int,
    node_id: Union[str, int],
) -> Optional[str]:
    """Resolve dataset_uri from per-client round JSON written during fit."""
    clients_dir = Path(save_path) / "clients"
    if not clients_dir.is_dir():
        return None

    node_id_str = str(node_id)
    for round_file in clients_dir.glob(f"*/round_{server_round}.json"):
        try:
            data = json.loads(round_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if str(data.get("client_id")) == node_id_str and data.get("dataset_name"):
            return str(data["dataset_name"])
    return None


def record_contributors_from_fit_results(
    save_path: Union[str, Path],
    server_round: int,
    validated_results: List[Any],
) -> List[Dict[str, Any]]:
    """Record contributor entries from validated Flower fit results."""
    registry_path = get_registry_path(save_path)
    records: List[Dict[str, Any]] = []

    for client_proxy, fit_res in validated_results:
        metrics = getattr(fit_res, "metrics", {}) or {}
        node_id = getattr(client_proxy, "cid", None) or metrics.get("client_id", "unknown")
        dataset_uri = metrics.get("dataset_name") or metrics.get("dataset_uri")
        if not dataset_uri:
            dataset_uri = lookup_dataset_from_client_round(
                save_path, server_round, node_id
            )
        if not dataset_uri:
            logger.warning(
                "Contributor registry: skipping fit result without dataset_uri (round {}, node_id={})",
                server_round,
                node_id,
            )
            continue

        record = build_contributor_record(
            node_id=node_id,
            dataset_uri=str(dataset_uri),
            source="server_fit",
            server_round=server_round,
        )
        append_contributor_record(registry_path, record)
        records.append(record)

    return records


def sync_contributor_registry_from_client_rounds(
    save_path: Union[str, Path],
    server_round: int,
) -> List[Dict[str, Any]]:
    """Sync contributor registry from client round JSON artifacts."""
    clients_dir = Path(save_path) / "clients"
    records: List[Dict[str, Any]] = []
    if not clients_dir.is_dir():
        return records

    for round_file in clients_dir.glob(f"*/round_{server_round}.json"):
        try:
            data = json.loads(round_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        dataset_uri = data.get("dataset_name")
        node_id = data.get("client_id")
        if not dataset_uri or node_id is None:
            continue
        record = build_contributor_record(
            node_id=node_id,
            dataset_uri=str(dataset_uri),
            source="client_round_metrics",
            server_round=server_round,
        )
        append_contributor_record(get_registry_path(save_path), record)
        records.append(record)
    return records