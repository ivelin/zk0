"""Scope proof for the fair expert-only specialists (founder decision
2026-11-22, study note study001-specialists-scope-decision-20261122.md).

Modeled on p1_cexp_check.py (the D-10 C-exp census): for EACH of the
five train tasks, build the exact specialist_exp run config via
study_cfg (single-task P0 episodes, 3,000 steps, expert_only=True),
construct the policy the way lerobot_train.train() does (make_policy),
and census the trainable tensors: they must be EXACTLY the certified
action-expert subset (zk0_node.fl.common.subset_keys — 155 tensors /
99,880,992 params), total params 450,046,176.
CPU only: the caller forces CUDA_VISIBLE_DEVICES="".
"""
import sys

sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-20261119/run/scripts")
sys.path.insert(0, "/home/openclaw/muse-work/zk0-node-build-20261117/repo/node/src")

import study_cfg  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies import make_policy  # noqa: E402
from zk0_node.fl.common import subset_keys  # noqa: E402

fails = []

ds_meta = LeRobotDatasetMetadata(study_cfg.DS_REPO, root=study_cfg.DS_ROOT)


def census(task):
    eps = study_cfg.episodes_for(task)
    cfg = study_cfg.build(eps, 3000, "/tmp/specexp_census_out",
                          f"census_spec_exp_{task}", expert_only=True)
    assert cfg.policy.train_expert_only is True
    assert cfg.policy.freeze_vision_encoder is True
    assert list(cfg.dataset.episodes) == list(eps)
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta,
                         rename_map=cfg.rename_map)
    sd = policy.state_dict()
    expected = set(subset_keys(sd))
    trainable = {k for k, p in policy.named_parameters() if p.requires_grad}
    total = sum(p.numel() for p in policy.parameters())
    train_n = sum(p.numel() for k, p in policy.named_parameters()
                  if p.requires_grad)
    ok = (trainable == expected and len(expected) == 155
          and train_n == 99_880_992 and total == 450_046_176)
    print(f"CENSUS spec_exp_{task}: episodes={len(eps)} "
          f"total_params={total} trainable_tensors={len(trainable)} "
          f"trainable_params={train_n} subset_tensors={len(expected)} "
          f"trainable_eq_subset={trainable == expected} "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        fails.append(task)
    del policy


for t in study_cfg.TASKS:
    census(t)

print("SPECEXP_CENSUS_RESULT",
      "FAIL " + ",".join(fails) if fails else "PASS")
sys.exit(1 if fails else 0)
