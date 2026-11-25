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
