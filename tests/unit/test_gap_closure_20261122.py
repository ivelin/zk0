"""Regression tests for the Study 001 gap closures of 2026-11-22.

Founder directive 2026-11-22 16:41 CT: close the gaps for all future
runs and protect each closure in test automation. Every test here is
tied to a gap-register entry (queue doc section 8) and fails against
the pre-fix behavior:

- G-10 (prep G1): per-round metrics.json carried the round-0
  server_composite_eval_loss verbatim into every round.
- G-4: the training loop's batch diagnostic read a nonexistent key
  and logged batch_img=N/A for all 15,000 steps.
- G-5: server-eval landmine - the server template and the initial
  model were built from a leftover server evaluation dataset, and
  the substrate aborted unless one was configured.
- G-13 (prep G4): client->task assignment existed only in log lines.
- G-11 (prep G2): the canonical eval gate report must cover the set.
- G-6: stale records must not sit in the eval active set.
- G-12 (prep G3): eval admission requires a standard-load proof, not
  existence of policy_preprocessor.json.
- G-14 (prep G5): packaging verification must work without CUDA.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- G-10
from src.server.server_utils import prepare_server_eval_metrics


def _strategy(losses, rounds, per_dataset=None, per_dataset_round=None):
    return SimpleNamespace(
        server_eval_losses=losses,
        server_eval_rounds=rounds,
        last_aggregated_metrics={},
        last_client_metrics=[],
        last_per_dataset_results=per_dataset or [],
        last_per_dataset_results_round=per_dataset_round,
    )


def test_g10_stale_round0_composite_not_reported_for_later_rounds():
    """The Study 001 defect itself: one round-0 measurement (0.5773),
    thirty round checkpoints claiming it. After the fix, a round
    that was not evaluated reports null - and no per-dataset rows."""
    per_dataset = [{"dataset_name": "d", "loss": 0.5773}]
    s = _strategy([0.5773], [0], per_dataset, per_dataset_round=0)
    for rnd in (1, 10, 30):
        out = prepare_server_eval_metrics(s, rnd)
        assert out["server_composite_eval_loss"] is None
        assert out["server_per_dataset_eval_results"] == []
        assert out["num_server_eval_datasets"] == 0


def test_g10_measured_round_reports_its_own_loss():
    s = _strategy([0.5773, 0.41], [0, 7],
                  [{"dataset_name": "d", "loss": 0.41}], per_dataset_round=7)
    out = prepare_server_eval_metrics(s, 7)
    assert out["server_composite_eval_loss"] == 0.41
    assert out["num_server_eval_datasets"] == 1
    # an earlier round may not claim the latest measurement either
    out0 = prepare_server_eval_metrics(s, 0)
    assert out0["server_composite_eval_loss"] is None


def test_g10_unattributable_losses_report_null():
    """Losses without round metadata cannot be claimed for any round."""
    s = _strategy([0.5], None)
    assert prepare_server_eval_metrics(s, 1)["server_composite_eval_loss"] is None


def test_g10_process_evaluation_metrics_records_rounds():
    from src.server.evaluation import process_evaluation_metrics

    s = SimpleNamespace(federated_metrics_history=[])
    process_evaluation_metrics(s, 0, 0.5773, {"policy_loss": 0.5}, {}, [])
    process_evaluation_metrics(s, 3, 0.41, {"policy_loss": 0.4}, {}, [])
    assert s.server_eval_losses == [0.5773, 0.41]
    assert s.server_eval_rounds == [0, 3]


# ----------------------------------------------------------------- G-4
from src.training.training_loop import batch_image_shapes


def test_g4_batch_image_shapes_reports_real_keys():
    import torch

    batch = {
        "observation.images.camera1": torch.zeros((2, 3, 256, 256)),
        "observation.images.camera2": torch.zeros((2, 3, 256, 256)),
        "observation.images.camera3": torch.zeros((2, 3, 256, 256)),
        "observation.state": torch.zeros((2, 6)),
    }
    shapes = batch_image_shapes(batch)
    assert shapes == {
        "observation.images.camera1": (2, 3, 256, 256),
        "observation.images.camera2": (2, 3, 256, 256),
        "observation.images.camera3": (2, 3, 256, 256),
    }


def test_g4_batch_image_shapes_na_only_when_no_images():
    import torch

    assert batch_image_shapes({"observation.state": torch.zeros((2, 6))}) == "N/A"


def test_g4_training_loop_has_no_singular_image_key_lookup():
    src = (REPO / "src/training/training_loop.py").read_text()
    assert "batch.get('observation.image')" not in src
    assert "'observation.image' in batch" not in src


# ----------------------------------------------------------------- G-5
def test_g5_pyproject_has_no_active_nonstudy_server_dataset():
    import tomllib

    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    datasets = cfg["tool"]["zk0"]["datasets"]
    client_names = {c["name"] for c in datasets["clients"]}
    for entry in datasets.get("server", []):
        assert entry["name"] in client_names, (
            f"active server dataset {entry['name']!r} is not the study "
            "dataset - the Incident 1 landmine pattern"
        )


def test_g5_template_meta_comes_from_first_client(monkeypatch):
    import src.common.utils as common_utils
    from src.configs.datasets import ClientConfig, DatasetConfig
    from src.server import server_utils

    cfg = DatasetConfig(
        clients=[
            ClientConfig(name="study/ds", description="node 1",
                         client_id=1, root="/data/tree",
                         episodes_file="/x/subsets/ring_insert/episodes.json"),
            ClientConfig(name="study/ds", description="node 0",
                         client_id=0, root="/data/tree",
                         episodes_file="/x/subsets/block_stack/episodes.json"),
        ],
        server=[],
    )
    monkeypatch.setattr(DatasetConfig, "load", classmethod(lambda cls: cfg))
    calls = {}

    def fake_load(repo_id, root=None, **kw):
        calls["repo_id"] = repo_id
        calls["root"] = root
        return SimpleNamespace(meta="CLIENT_META")

    monkeypatch.setattr(common_utils, "load_lerobot_dataset", fake_load)
    meta = server_utils.load_template_dataset_meta()
    assert meta == "CLIENT_META"
    assert calls == {"repo_id": "study/ds", "root": "/data/tree"}


def test_g5_initialize_global_model_needs_no_server_dataset(monkeypatch):
    import numpy as np

    import src.common.parameter_utils as param_utils
    import src.training.model_utils as training_model_utils
    from src.server import server_utils

    monkeypatch.setattr(
        server_utils, "load_template_dataset_meta", lambda: "CLIENT_META")
    monkeypatch.setattr(
        training_model_utils, "get_model", lambda dataset_meta: object())
    monkeypatch.setattr(
        training_model_utils, "get_params", lambda m: [np.zeros((2,))])
    monkeypatch.setattr(
        param_utils, "validate_and_log_parameters", lambda *a, **k: None)
    params, meta = server_utils.initialize_global_model()
    assert meta == "CLIENT_META"
    assert params is not None


# ---------------------------------------------------------------- G-13
def test_g13_client_task_map_records_tasks():
    from src.configs.datasets import ClientConfig, DatasetConfig

    cfg = DatasetConfig(
        clients=[
            ClientConfig(name="study/ds", description="n1", client_id=1,
                         root="/data/tree",
                         episodes_file="/x/subsets/ring_insert/episodes.json"),
            ClientConfig(name="study/ds", description="n0", client_id=0,
                         root="/data/tree",
                         episodes_file="/x/subsets/block_stack/episodes.json"),
        ],
        server=[],
    )
    rows = cfg.client_task_map()
    assert [r["client_id"] for r in rows] == [0, 1]
    assert rows[0]["task"] == "block_stack"
    assert rows[1]["task"] == "ring_insert"


def test_g13_config_snapshot_embeds_client_task_map(tmp_path, monkeypatch):
    import datetime

    from src.configs.datasets import ClientConfig, DatasetConfig
    from src.server import server_utils

    cfg = DatasetConfig(
        clients=[ClientConfig(
            name="study/ds", description="n0", client_id=0, root="/data/tree",
            episodes_file="/x/subsets/tool_insert/episodes.json")],
        server=[],
    )
    monkeypatch.setattr(DatasetConfig, "load", classmethod(lambda cls: cfg))
    context = SimpleNamespace(run_config={"federation": "test"})
    server_utils.save_config_snapshot(
        context, tmp_path, datetime.datetime(2026, 11, 22), "0.0.0")
    snap = json.loads((tmp_path / "config.json").read_text())
    assert snap["client_task_map"][0]["task"] == "tool_insert"
    assert snap["client_task_map"][0]["client_id"] == 0


# ---------------------------------------------------------------- G-11
eg = _load_module("eval_gate_check", REPO / "study_prep" / "eval_gate_check.py")


def test_g11_gate_report_coverage():
    report = {
        "verdict": "PASS",
        "checkpoints": {
            "fl_r30_3cam": {"config_sha256": "abc"},
            "cexp": {"config_sha256": "def"},
        },
    }
    assert eg.assert_gate_report_covers(report, ["fl_r30_3cam", "cexp"]) == []
    problems = eg.assert_gate_report_covers(report, ["fl_r30_3cam", "sequential"])
    assert any("sequential" in p for p in problems)
    bad = dict(report, verdict="FAIL")
    assert eg.assert_gate_report_covers(bad, ["cexp"]) != []
    nohash = {"verdict": "PASS", "checkpoints": {"cexp": {}}}
    assert eg.assert_gate_report_covers(nohash, ["cexp"]) != []


# ----------------------------------------------------------------- G-6
def test_g6_stale_active_records_detected(tmp_path):
    (tmp_path / "fl_r30_3cam__block_stack__ep1.npz").write_bytes(b"x")
    (tmp_path / "fl_r30__block_stack__ep2.npz").write_bytes(b"x")
    stale = eg.stale_active_records(tmp_path, ["fl_r30_3cam", "cexp"])
    assert stale == ["fl_r30__block_stack__ep2.npz"]
    # per-checkpoint summaries are named summary__<ckpt>.json
    (tmp_path / "summary__cexp.json").write_text("{}")
    (tmp_path / "summary__fl_r30.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("not a record")
    stale = eg.stale_active_records(tmp_path, ["fl_r30_3cam", "cexp"])
    assert stale == ["fl_r30__block_stack__ep2.npz", "summary__fl_r30.json"]


def test_g6_gate_wires_active_records_hygiene():
    src = (REPO / "study_prep" / "eval_gate_check.py").read_text()
    assert "eval.active_records_hygiene" in src
    assert "stale_active_records(records_dir" in src


# ---------------------------------------------------------------- G-12
def test_g12_rewrite_device_steps_cpu_is_pure():
    pre = {"steps": [
        {"registry_name": "device_processor", "config": {"device": "cuda"}},
        {"registry_name": "rename_observations_processor", "config": {}},
    ]}
    out = eg.rewrite_device_steps_cpu(pre)
    assert out["steps"][0]["config"]["device"] == "cpu"
    assert pre["steps"][0]["config"]["device"] == "cuda"  # original untouched


def _build_processor_fixture(d: Path, rename_map: dict):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.processor import (
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
    )

    feats = {
        "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.state": {"type": "STATE", "shape": [6]},
    }
    cfg = SmolVLAConfig(
        input_features=feats,
        output_features={"action": {"type": "ACTION", "shape": [6]}},
    )
    cfg.save_pretrained(d)
    pre = PolicyProcessorPipeline(
        steps=[RenameObservationsProcessorStep(rename_map=rename_map)])
    pre.save_pretrained(d)
    (d / "dataprocessorpipeline.json").rename(d / "policy_preprocessor.json")
    scratch = d / "_post_scratch"
    scratch.mkdir()
    PolicyProcessorPipeline(steps=[]).save_pretrained(scratch)
    (scratch / "dataprocessorpipeline.json").rename(d / "policy_postprocessor.json")
    scratch.rmdir()


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None, reason="lerobot unavailable")
def test_g12_standard_load_proof_discriminates(tmp_path):
    raw = ["observation.images.front", "observation.images.top",
           "observation.images.wrist"]
    decl = ["observation.images.camera1", "observation.images.camera2",
            "observation.images.camera3"]
    good = tmp_path / "good"
    good.mkdir()
    _build_processor_fixture(good, {
        "observation.images.front": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
        "observation.images.wrist": "observation.images.camera3",
    })
    ok, _detail = eg.standard_load_proof(good, decl)
    assert ok
    # A preprocessor file that EXISTS but whose pipeline cannot be
    # constructed: the old existence-only gate check passed this;
    # the proof must not. (The real-world instance - a raw round
    # checkpoint whose normalizer state file is absent - is covered
    # by the gate drill on the actual artifacts.)
    bad = tmp_path / "bad"
    bad.mkdir()
    _build_processor_fixture(bad, {})
    pre_path = bad / "policy_preprocessor.json"
    pre = json.loads(pre_path.read_text())
    pre["steps"] = [{"registry_name": "no_such_processor_step",
                     "config": {}}]
    pre_path.write_text(json.dumps(pre))
    ok, detail = eg.standard_load_proof(bad, decl)
    assert not ok


# ---------------------------------------------------------------- G-14
pkg = _load_module(
    "package_final_checkpoint", REPO / "study_prep" / "package_final_checkpoint.py")


def test_g14_packaging_rewrite_device_steps_cpu():
    pre = {"steps": [
        {"registry_name": "device_processor",
         "config": {"device": "cuda", "float_dtype": None}},
        {"registry_name": "normalizer_processor", "config": {}},
    ]}
    out = pkg.rewrite_device_steps_cpu(pre)
    assert out["steps"][0]["config"]["device"] == "cpu"
    assert out["steps"][0]["config"]["float_dtype"] is None
    assert pre["steps"][0]["config"]["device"] == "cuda"


def test_g14_packaging_cli_accepts_cpu_verify(tmp_path):
    # flags must parse and reach the checkpoint assertion (the dir
    # has no model.safetensors, so packaging refuses it as before)
    with pytest.raises(AssertionError):
        pkg.main(["--verify-only", "--cpu-verify", str(tmp_path)])
