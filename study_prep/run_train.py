"""Run one Study 001 full-FT arm segment (specialist / centralized /
sequential addition), or a P1/P3 probe. Mirrors run_ft.py's discipline:
dump the FULL effective config and assert the safety-critical fields
BEFORE training; then call lerobot's train() unchanged.

Usage:
  run_train.py --arm specialist --task block_stack --steps 3000 \
      --output-dir DIR [--save-freq 400] [--log-freq 20] \
      [--resume CKPT_STEP_DIR] [--check-only] [--no-save]
  run_train.py --arm centralized --steps 15000 --output-dir DIR ...
  run_train.py --arm sequential --task ring_insert --init-ckpt DIR \
      --steps 3000 --output-dir DIR ...
  run_train.py --arm specialist_exp --task block_stack --steps 3000 \
      --output-dir DIR ...   (fair expert-only specialist, from base)

Writes <output-dir>/../run_manifest_<job>.json at start and updates it
at exit (status, wall_s). All executable code under __main__ (DataLoader
spawn re-import discipline, 2x2 P1 finding).

Amendment D-10 (2026-11-20): the centralized arm is C-exp and the
sequential chain follows the amended centralized recipe — both build
expert-only (certified 155-tensor action-expert subset, backbone
frozen, flags true/true). Specialists remain full fine-tune.

Founder decision 2026-11-22 (specialists-scope resolution, study note
study001-specialists-scope-decision-20261122.md): the arm
"specialist_exp" adds the FAIR specialist baselines — single-task runs
from the pinned base, expert-only under the same D-10 scope mechanism
(155-tensor subset, flags true/true), 3,000 steps, same per-task P0
datasets as the full-FT specialists (which remain the ceiling
reference, untouched). Manifest amendment tag: SPEC-EXP-20261122.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-20261119/run/scripts")
import study_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["specialist", "centralized", "sequential",
                             "specialist_exp"])
    ap.add_argument("--task", default=None)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--init-ckpt", default=study_cfg.BASE)
    ap.add_argument("--save-freq", type=int, default=400)
    ap.add_argument("--log-freq", type=int, default=20)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    # Amendment D-10: scope is a function of the arm, nowhere else.
    # specialist_exp (founder decision 2026-11-22) shares the D-10
    # expert-only scope; the full-FT specialist arm is unchanged.
    expert_only = a.arm in ("centralized", "sequential", "specialist_exp")

    if a.arm == "centralized":
        episodes = study_cfg.all_episodes()
        job = "study001_centralized"
        subset_ref = {t: study_cfg.subset_sha(t) for t in study_cfg.TASKS}
    else:
        assert a.task in study_cfg.TASKS, a.task
        episodes = study_cfg.episodes_for(a.task)
        job = f"study001_{a.arm}_{a.task}"
        subset_ref = {a.task: study_cfg.subset_sha(a.task)}
    eps_sha = hashlib.sha256(
        json.dumps(sorted(episodes)).encode()).hexdigest()

    out = Path(a.output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "job": job, "arm": a.arm, "task": a.task, "steps": a.steps,
        "save_freq": a.save_freq, "output_dir": str(out),
        "init_ckpt": a.init_ckpt, "episodes_count": len(episodes),
        "episodes_list_sha256": eps_sha, "p0_subset_sha256": subset_ref,
        "train_expert_only": expert_only,
        "freeze_vision_encoder": expert_only,
        "amendment": ("SPEC-EXP-20261122" if a.arm == "specialist_exp"
                      else "D-10" if expert_only else None),
        "resume": a.resume, "status": "built",
        "argv": sys.argv[1:],
    }
    mpath = out.parent / f"run_manifest_{job}.json"
    mpath.write_text(json.dumps(manifest, indent=2))

    if a.resume:
        # LeRobot 0.7.0 resume resolution is CLI-shaped (RERUN spec §1.4).
        sys.argv.append(
            f"--config_path={a.resume}/pretrained_model/train_config.json")
        cfg = study_cfg.build_resume(
            episodes, a.output_dir, a.resume, a.steps, job,
            init_ckpt=a.init_ckpt, log_freq=a.log_freq,
            save_freq=a.save_freq, expert_only=expert_only)
    else:
        cfg = study_cfg.build(
            episodes, a.steps, a.output_dir, job, init_ckpt=a.init_ckpt,
            log_freq=a.log_freq, save_checkpoint=not a.no_save,
            save_freq=a.save_freq, expert_only=expert_only)

    # Assertions before any training (run_ft.py discipline).
    assert isinstance(cfg.output_dir, Path), type(cfg.output_dir)
    assert isinstance(cfg.dataset.root, Path), type(cfg.dataset.root)
    assert cfg.policy.train_expert_only is expert_only
    assert cfg.policy.freeze_vision_encoder is expert_only
    assert cfg.policy.push_to_hub is False
    assert list(cfg.dataset.episodes) == list(episodes)
    if not a.no_save and not a.resume:
        assert 0 < a.save_freq < a.steps, (a.save_freq, a.steps)
    eff = out.parent / f"effective_config_{job}.json"
    eff.write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    print(f"EFFECTIVE_CONFIG {eff}")
    print(f"RECIPE steps={cfg.steps} batch={cfg.batch_size} "
          f"lr={cfg.policy.optimizer_lr} betas={cfg.policy.optimizer_betas} "
          f"wd={cfg.policy.optimizer_weight_decay} "
          f"clip={cfg.policy.optimizer_grad_clip_norm} "
          f"warmup={cfg.policy.scheduler_warmup_steps} "
          f"decay_steps={cfg.policy.scheduler_decay_steps} "
          f"episodes={len(episodes)} eps_sha={eps_sha[:12]} "
          f"train_expert_only={cfg.policy.train_expert_only} "
          f"freeze_vision={cfg.policy.freeze_vision_encoder}")
    if a.check_only:
        manifest["status"] = "check-only-ok"
        mpath.write_text(json.dumps(manifest, indent=2))
        print("CHECK_ONLY_OK")
        return 0

    from lerobot.scripts.lerobot_train import train
    manifest["status"] = "training"
    manifest["train_start_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mpath.write_text(json.dumps(manifest, indent=2))
    t0 = time.time()
    try:
        train(cfg)
        manifest["status"] = "done"
        rc = 0
    except Exception as e:  # noqa: BLE001 - recorded, then re-raised
        manifest["status"] = f"failed: {type(e).__name__}: {e}"
        rc = 1
        raise
    finally:
        manifest["wall_s"] = round(time.time() - t0, 1)
        mpath.write_text(json.dumps(manifest, indent=2))
    print(f"TRAIN_END job={job} wall_s={manifest['wall_s']} status=done")
    return rc


if __name__ == "__main__":
    sys.exit(main())
