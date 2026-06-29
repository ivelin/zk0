"""Unit tests for contributor registry utilities."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from flwr.common import Context
from flwr.common.record import RecordDict

from src.common.contributor_registry import (
    REGISTRY_FILENAME,
    append_contributor_record,
    build_contributor_record,
    extract_dataset_uri_from_context,
    get_registry_path,
    record_contributor_from_context,
    record_contributors_from_fit_results,
)
from src.common.utils import get_dataset_slug


def _prod_context(
    node_id: int = 7,
    dataset_uri: str = "shaunkirby/record-test",
    save_path: str = "/tmp/zk0-test-run",
) -> Context:
    return Context(
        run_id=1,
        node_id=node_id,
        node_config={"dataset-uri": dataset_uri},
        state=RecordDict(),
        run_config={
            "save_path": save_path,
            "model-name": "lerobot/smolvla_base",
            "local-epochs": 1,
            "num-server-rounds": 1,
        },
    )


class TestBuildContributorRecord:
    def test_builds_required_fields(self):
        record = build_contributor_record("node-abc", "user/private-dataset", source="client")

        assert record["node_id"] == "node-abc"
        assert record["dataset_uri"] == "user/private-dataset"
        assert record["source"] == "client"
        assert "timestamp" in record

    def test_rejects_empty_dataset_uri(self):
        with pytest.raises(ValueError, match="dataset_uri"):
            build_contributor_record("node-abc", "  ")


class TestRegistryPersistence:
    def test_append_writes_jsonl(self, tmp_path: Path):
        registry_path = tmp_path / REGISTRY_FILENAME
        record = build_contributor_record(42, "ethanCSL/direction_test")

        append_contributor_record(registry_path, record)

        lines = registry_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert '"dataset_uri": "ethanCSL/direction_test"' in lines[0]
        assert '"node_id": "42"' in lines[0]


class TestContextIntegration:
    def test_extract_dataset_uri_from_node_config(self):
        context = _prod_context(dataset_uri="local:/data/episodes")

        assert extract_dataset_uri_from_context(context) == "local:/data/episodes"

    def test_get_dataset_slug_matches_node_config(self):
        context = _prod_context(dataset_uri="giovipeg/record-test")

        assert get_dataset_slug(context) == "giovipeg/record-test"

    def test_record_contributor_from_context_persists(self, tmp_path: Path):
        save_path = tmp_path / "run-output"
        context = _prod_context(node_id=99, dataset_uri="VoicAndrei/so100_cubes_three_cameras", save_path=str(save_path))

        record = record_contributor_from_context(context, source="client_registration")

        assert record["node_id"] == "99"
        assert record["dataset_uri"] == "VoicAndrei/so100_cubes_three_cameras"
        registry_file = get_registry_path(save_path)
        assert registry_file.exists()
        assert "VoicAndrei/so100_cubes_three_cameras" in registry_file.read_text(encoding="utf-8")


class TestFitResultRecording:
    def test_record_contributors_from_fit_results(self, tmp_path: Path):
        save_path = tmp_path / "server-run"
        client_proxy = SimpleNamespace(cid="client-proxy-1")
        fit_res = SimpleNamespace(metrics={"dataset_name": "shaunkirby/record-test", "client_id": "client-proxy-1"})

        records = record_contributors_from_fit_results(
            save_path, server_round=3, validated_results=[(client_proxy, fit_res)]
        )

        assert len(records) == 1
        assert records[0]["dataset_uri"] == "shaunkirby/record-test"
        assert records[0]["node_id"] == "client-proxy-1"
        assert records[0]["server_round"] == 3
        registry_text = get_registry_path(save_path).read_text(encoding="utf-8")
        assert "server_fit" in registry_text
        assert "shaunkirby/record-test" in registry_text