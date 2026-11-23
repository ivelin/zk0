"""Study 001 packaging step (D-11/G-8): write the serialized D-9
pooled preprocessor into a finished FL checkpoint directory,
replacing the base-stats preprocessor files save_model_checkpoint
copied there, plus the tokenizer artifact and the postprocessor
unnormalizer state the substrate does not write. Then verify:
(a) the dir loads via SmolVLAPolicy.from_pretrained +
make_pre_post_processors (the eval stage's own construction);
(b) the packaged preprocessor's stats equal the pooled D-9 stats
at F32 - the serialization form every arm's checkpoint carries
(bit-equality with the F64 D-9 hash is impossible in this form by
construction; the F32 round-trip hash is a5557279...); (c) the
unnormalizer's action stats equal the pooled action stats (F32).

Usage: package_final_checkpoint.py <checkpoint_dir> [--verify-only]
       [--cpu-verify]

--verify-only   skip the packaging copies; run only the verification
                half (used by tests and re-checks).
--cpu-verify    run the verification against a temp copy of the
                checkpoint whose preprocessor device step is set to
                CPU (Study 001 gap G-14: the verification used to
                hard-fail whenever CUDA was unavailable - e.g. on a
                busy training box - because processor construction
                instantiates the artifact's cuda device step. The
                artifact of record is never modified by this mode.)
                CPU verification is also selected automatically when
                CUDA is unavailable.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-20261119/run/scripts")

EXPECTED = "a0455b5ba7d75b9c3d743754828020ca751f29fe93447a106207553b89d7a39d"
POOLED = Path(
    "/home/openclaw/muse-work/zk0-study001-fl-20261121/study_assets/pooled_preprocessor"
)
POST_STATE = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"


def rewrite_device_steps_cpu(preprocessor: dict) -> dict:
    """Copy of a serialized preprocessor pipeline with every device
    step set to CPU (gap G-14). Pure function of the JSON structure;
    the artifact on disk is never rewritten by CPU verification."""
    import copy

    out = copy.deepcopy(preprocessor)
    steps = out.get("steps", out if isinstance(out, list) else [])
    for step in steps:
        if isinstance(step, dict) and step.get("registry_name") == "device_processor":
            cfg = step.get("config") or {}
            cfg["device"] = "cpu"
            step["config"] = cfg
    return out


def package(ckpt: Path) -> None:
    copied = []
    for src in sorted(POOLED.iterdir()):
        if src.name.startswith("policy_preprocessor"):
            shutil.copy2(src, ckpt / src.name)
            copied.append(src.name)
    print("copied pooled preprocessor files:", copied)
    assert "policy_preprocessor.json" in copied
    # The preprocessor references a tokenizer artifact directory; the
    # substrate's save_model_checkpoint does not copy it into round
    # checkpoints, so packaging must add it (else the packaged dir
    # fails to load: "Missing processor artifact 'tokenizer'").
    post_src = POOLED / POST_STATE
    assert post_src.is_file(), post_src
    shutil.copy2(post_src, ckpt / POST_STATE)
    print("copied postprocessor state:", POST_STATE)
    tok_src = POOLED / "tokenizer"
    if tok_src.is_dir():
        tok_dst = ckpt / "tokenizer"
        if tok_dst.exists():
            shutil.rmtree(tok_dst)
        shutil.copytree(tok_src, tok_dst)
        print("copied tokenizer artifact dir")


def _verify_dir(ckpt: Path) -> None:
    """The verification half, run against a (possibly temp) dir."""
    import fl_driver
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(ckpt))
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(ckpt)
    )
    h = fl_driver.preprocessor_stats_hash(pre)
    print("packaged preprocessor stats hash (F32 serialization):", h)
    # The packaged form is lerobot's F32 serialization - the same form
    # every arm's checkpoint carries - so bit-equality with the F64
    # D-9 hash is impossible BY CONSTRUCTION (measured: the F32
    # round-trip hash is a5557279...). The binding acceptance is
    # tensor-level: every packaged mean/std equals the pooled D-9
    # stats cast to F32, exactly.
    import numpy as np
    from safetensors.torch import load_file as _lf

    pooled = _lf(str(POOLED / "pooled_stats.safetensors"))
    norm = None
    for step in pre.steps:
        if type(step).__name__ == "NormalizerProcessorStep":
            norm = step
            break
    assert norm is not None
    for key in sorted(norm.stats):
        for sub in ("mean", "std"):
            if sub not in norm.stats[key]:
                continue
            v = norm.stats[key][sub]
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            got = np.asarray(v, dtype=np.float64)
            want64 = pooled[f"{key}|{sub}"].numpy().astype(np.float32).astype(np.float64)
            # The F32 serialization squeezes scalar stats ((1,) -> ());
            # values and order are the binding content.
            assert got.size == want64.size and np.array_equal(
                got.reshape(-1), want64.reshape(-1)
            ), f"packaged stats mismatch at {key}.{sub}"
    print("packaged preprocessor stats == pooled D-9 stats at F32: PASS")

    # The unnormalizer must carry the pooled action stats (F32 form).
    import torch
    from safetensors.torch import load_file

    want = load_file(str(POOLED / POST_STATE))
    found = None
    for step in post.steps:
        st = getattr(step, "stats", None)
        if isinstance(st, dict) and "action" in st:
            found = st
            break
    assert found is not None, "unnormalizer stats not found in loaded postprocessor"
    for sub in ("mean", "std"):
        got = torch.as_tensor(found["action"][sub]).detach().cpu().to(torch.float32)
        assert torch.equal(got, want[f"action.{sub}"]), f"post action.{sub} mismatch"
    print("packaged postprocessor action stats == pooled (F32) PASS")


def verify(ckpt: Path, cpu_verify: bool) -> None:
    if not cpu_verify:
        _verify_dir(ckpt)
        return
    # Gap G-14: verify on a temp copy whose preprocessor device step
    # is CPU. Everything else is symlinked; the artifact of record
    # keeps its cuda device step and is never modified.
    with tempfile.TemporaryDirectory(prefix="pkg_cpu_verify_") as td:
        tmp = Path(td) / "ckpt"
        tmp.mkdir()
        for entry in sorted(ckpt.iterdir()):
            os.symlink(entry, tmp / entry.name)
        pre_json = json.loads((ckpt / "policy_preprocessor.json").read_text())
        (tmp / "policy_preprocessor.json").unlink()
        (tmp / "policy_preprocessor.json").write_text(
            json.dumps(rewrite_device_steps_cpu(pre_json))
        )
        _verify_dir(tmp)
    print("CPU verification ran on a temp copy; artifact of record unchanged")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt", help="checkpoint directory")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip packaging; run only the verification half")
    ap.add_argument("--cpu-verify", action="store_true",
                    help="verify on a temp CPU copy (gap G-14); also "
                         "selected automatically when CUDA is unavailable")
    args = ap.parse_args(argv)

    ckpt = Path(args.ckpt)
    assert ckpt.is_dir(), ckpt
    assert (ckpt / "model.safetensors").is_file(), "not a checkpoint dir"

    if not args.verify_only:
        package(ckpt)

    cpu_verify = args.cpu_verify
    if not cpu_verify:
        import torch

        if not torch.cuda.is_available():
            print("CUDA unavailable - falling back to CPU verification (G-14)")
            cpu_verify = True
    verify(ckpt, cpu_verify)
    print("PACKAGING PASS:", ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
