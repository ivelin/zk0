# Q-A — decision traces

Each entry: the decision, who made it, when, and the evidence it rests on.

## 1. Launch — APPROVED
Founder, 2026-11-25 ~14:22 CT (voice). The F50 results sprint plan approved; Q-A relaunch word given. Acceleration added ~14:28 CT: launch the moment the gates pass — no calendar wait ("Why wait until Friday?").

## 2. Data — certified set only
The run consumes only the Study 001 ArmnetBench training pool as certified in `register.csv` (frozen filter: primary verdict PASS + human certification on record). ep604 (the pool's only certified FAIL) is excluded. Preserved/community sets are excluded — all still PENDING, uncertified. Held-out episodes never train; they are the judging set.

## 3. Cable re-split — 40 train / 10 held-out
Founder decision 2026-11-23 (seed 20261123), per `cable-resplit-manifest-20261123.md`: the original 20 held-out cable episodes were 20/20 FAIL and are removed from the active eval set; 10 certified cable training episodes — 2164, 2169, 2170, 2183, 2190, 2193, 2194, 2195, 2200, 2208 (all CLEAN) — were drawn as the new certified held-out eval set. Cable trains on the remaining 40 (incl. ep2201, founder-adjudicated PASS 2026-11-24).

## 4. List correction — ep573 IN; run set 239
Reconciliation audit 2026-11-25, item C1: ep573 (block_stack) carried primary PASS + CLEAN, but its register `human_status` read "PENDING HUMAN APPROVAL" while the register's own notes field, the training verdicts_master, and the 2026-11-23 daily log all record the founder's adjudication — PASS, 2026-11-23 16:10 CT ("Gripper released green block and stack stood on its own"). The stale field had propagated into the readiness memo, which ruled ep573 out and built a 238 list on it. Register field corrected at the source 2026-11-25 (backup `register.csv.bak-pre-reconciliation-20261125`); ep573 adopted IN, following the ep2201 precedent (a recorded founder adjudication governs). Independent verification of the audit: CERTIFIED, counts recomputed from the register to the digit — certified PASS 249 (block 49; CLEAN 246 + RECOVERED 3), train 239 (236 CLEAN + 3 RECOVERED).

## 5. Preprocessor stats — keep the union-250 stats
Recorded choice for the launch report: the pooled preprocessor statistics stay the union-250 set fl_r30 itself used. Equivalence on the preprocessor axis outranks recomputing on 239; ep604's influence is aggregate normalization statistics only — it is not a training episode. Disclosed here and in the launch report, not hidden.

## 6. Claim discipline — two variables, denominator 90
The rerun differs from fl_r30 in two ways (AdamW carry + dataset 250 → 239). No Q-A claim may attribute the outcome to carry alone. The active eval set is 90 episodes; every reported score states its denominator.

## 7. Record home — this branch
Founder rule 2026-11-25 ~14:37 CT (voice): every experiment gets a dedicated GitHub branch; everything key — log, decisions, decision traces — lives in that branch. A GitHub issue may point to the branch but is not a second home. (SSOT append as Decision 114 handled from the parent lane.)

## 8. Mirror target — staging on-box, durability here
The job guard's telemetry mirror points at the DGX staging dir in the run home (proven 2026-11-25, three clean syncs). The Muse VM target is structurally unreachable from the DGX (platform-layer refusal, proven with a live listener 2026-11-25); the founder's desktop is paused (memtest). Off-box durability for the run's records is this branch, updated as the run progresses.

## 9. Run spec — exactly THREE arms (founder, binding)
Founder, 2026-11-25 ~14:41 CT (voice): "That's it. I don't want any other trainings done." On the certified 239-episode set:
1. **Specialists** — five fine-tuned expert models, one per task, each trained ONLY on its own task's episodes (block 49, eye 50, ring 50, tool 50, cable 40). Full fine-tune, as in Study 001 (census: 489 trainable tensors / 392,904,096 params, name-set sha256 91412384…).
2. **Centralized pooled** — one model trained on all tasks simultaneously, pooled data (all 239). Expert-only, as in Study 001 (155 / 99,880,992, name-set b36812bc…).
3. **Federated (Q-A)** — five clients, one task per node; each node sees ONLY its own task's data; the aggregator averages across all five; AdamW carry as registered. Expert-only (155 / 99,880,992).
Arms 2 and 3 are all-else-equal to each other (the FL-vs-centralized comparison isolates federation + AdamW carry as registered). Sequencing: FL first (headline, longest), then centralized, then the five specialists; one GPU, serial, no concurrency. No additional arms, variants, or side runs. Gate manifest for the relaunch: `study_prep/qa_gate_manifest_239_20261125.json` in the run tree (registers all three arms: cameras, preprocessor + stats hash, per-arm census, episode lists, optimizer/schedule, budget).

## 10. Waiver — raw-save checkpoint form (founder ruling)
Founder, 2026-11-25 14:58 CT (voice): "Apply the waiver." The overrides-off config-equivalence gate (2026-11-25 ~14:56 CT) passed 118 of 119 checks and failed exactly one: `saved_checkpoint.input_features` (fl-artifact, shape-exact rule) — the substrate's raw `save_model_checkpoint` writes the checkpoint config with pre-rename camera names at dataset-native shapes (after the checker's rename normalization: camera1 [3,576,1024], camera2 [3,576,1024], camera3 [3,720,1280], state [6]) vs the registered packaged form (camera1/2/3 [3,256,256] + state [6]).
**Reasoning:** the raw form is the substrate's real checkpoint behavior for every arm and every round — all checkpoints of the 2026-11-23 Q-A run and of Study 001 carry it; it is not a divergence introduced by this run. The packaged form (pooled preprocessor + the A5 metadata config correction, the Study 001 production packaging path) is what the eval-side gate shape-checks before any judging, and the smoke's packaging phase must prove the packaged miniature passes that shape-exact check before launch. The waiver is registered in the gate manifest as `W-rawsave-form` (manifest sha256 b9e21bec… -> 5bec3de8…) and cited by id at gate time (`--waive saved_checkpoint.input_features=W-rawsave-form`); the re-run report lists the check as DIFF-WAIVED. **Scope: this ONE form check only — every other gate check stands unwaived, and no configuration override (--checkpoint-config / --save-template-config / --server-*) is used.**
