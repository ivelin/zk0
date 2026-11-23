# Twin-Suite v2 — Observation Probe: First-Chunk Agreement + Ghost Rollout

**Date:** 2026-11-23 · **Type:** the decisive experiment registered in
`gap-mechanism-diagnosis-v2.md` §3/§7, run under the founder's v2
priority directive (2026-11-23 07:14 CT). Diagnosis only — no
certification, standard, criterion, or threshold is changed by
anything here. Frozen scenes/criteria; the harness's camera synthesis
was **not** modified.
**Checkpoints (controls-manifest pins):** C-exp
(weights SHA-256 `8f6807ed…48379`) and FL round-30 A5-repaired
(`5b1cbf29…84b4`). **Scenes:** the 8 solvability-validated block_stack
scenes (dev 411/453/489/498 + envelope 499/527/546/549), nominal
variants. **Code + artifacts:** DGX
`zk0-study001-twin-v2-20261122/probe/` (scripts + `work/` outputs:
saved chunks, frames, states, ghost trajectories).

> Every gate output discussed here carries the §8 label: funnel
> verdict — sim bar unanchored to real-arm delta.

## 1. Design

**Step 1 — first-chunk action agreement.** For each scene, the twin is
initialised exactly as the rollout harness does (arm at the recorded
first action, 0.25 s held pre-settle). From that identical initial
state, the policy service is queried for chunk 0 (50 actions) under:

- **A** — the twin harness's rendered initial frames (substituted
  `frontfix`, scene `topcam`, synthesised `wristcam`), harness state;
- **A2** — an exact repeat of A (noise-floor control; the service seeds
  noise per (episode, chunk), held identical, so any A–B difference is
  attributable to the observations, not sampling);
- **B** — the **recorded demo's frame-0 images**, with the **same
  state input as A** (isolates the image contract);
- **B2** — recorded images + recorded frame-0 state (the true
  offline-eval chunk-0 input; sanity cross-check only, §3).

**Step 2 — ghost rollout** (run because Step 1 diverged materially):
C-exp closed-loop in the twin on episodes 411 and 489, actuated loop
identical to the harness, except the policy is fed the **recorded demo
frames at every chunk boundary**; the state channel remains the
twin's own state. Measured: per-step joint error vs the demo (the
records' `gt` action array — the same reference the diagnosis used)
and the v1.1 mechanical outcome.

## 2. Step 1 results — the observation effect at step 0

Noise floor (A vs A2): **exactly 0.000° in all 16 cells** — the repeat
is bit-identical, so every A–B difference below is observation-caused.

Mean-abs divergence A vs B over the 50-action chunk (degrees;
radians in §2.1), per scene:

| Episode | C-exp A–B mean | C-exp max | FL A–B mean | FL max |
|---|---|---|---|---|
| 411 | 28.66 | 82.98 | 24.30 | 74.77 |
| 453 | 8.61 | 23.73 | 5.03 | 12.57 |
| 489 | 6.17 | 29.86 | 7.47 | 23.40 |
| 498 | 12.94 | 42.74 | 10.57 | 30.64 |
| 499 | 13.11 | 41.15 | 12.68 | 45.95 |
| 527 | 11.49 | 41.60 | 7.62 | 24.56 |
| 546 | 14.01 | 52.21 | 14.82 | 51.19 |
| 549 | 10.46 | 35.66 | 6.45 | 20.01 |
| **Median** | **12.21** | — | **9.10** | — |

Per-step L2 divergence (mean over the chunk) reaches 79.7° (C-exp,
ep411). The divergence is broad across joints, not one channel —
worst cell per-joint mean-abs (C-exp ep411):
[6.3, 55.1, 30.8, 27.7, 34.4, 17.6]°. Full per-joint tables are in
`probe/work/step1_results.json`.

### 2.1 Effect size in radians

Median A–B mean-abs divergence: **0.213 rad (C-exp)**, **0.159 rad
(FL)**; largest single-action divergence 1.45 rad (C-exp ep411).
Against a noise floor of **0 rad**, the step-0 observation effect is
unbounded in relative terms and large in absolute terms: swapping
only the initial images — state, seed, service, checkpoint held
fixed — rewrites the first action chunk at the scale of the whole
closed-loop divergence the diagnosis measured by 10% of episode.

## 3. Sanity cross-check (B2) — loose, by construction

B2 vs the saved offline `pred[0:50]`: median mean-abs **3.25°**
(C-exp), **9.52°** (FL). The independent verifier established why
this cannot be an exact reproduction: the offline eval ran on CUDA
(CUDA RNG stream), the probe service is CPU-only (CPU RNG stream) —
same seed formula, different stream and device arithmetic. B2 is
therefore a directional sanity check only (the recorded-frame path
behaves offline-like, at single-digit degrees), **not** a
reproduction test, and no claim below rests on it.

## 4. Step 2 results — the ghost does not recover the demo

Per-step joint error vs the demo (mean / median, degrees):

| Episode | Offline (OL) | Closed-loop (L1) | Ghost (recorded frames) |
|---|---|---|---|
| 411 | 21.0 / 20.2 | 56.2 / 65.1 | **52.9 / 54.0** |
| 489 | 24.2 / 18.9 | 64.8 / 69.0 | **55.1 / 59.0** |

Mechanical outcome (v1.1 criteria): ghost **fails primary and
secondary on both episodes** — no success where the closed loop
failed. (L1 ep411 had passed the secondary near-miss criterion; the
ghost did not.) Final object positions are in the same neighbourhood
in ghost and L1 (on ep411 the dragged cube ends at the identical
position, (0.066, −0.023) m, in both).

Reading: restoring the recorded images recovers only ~3–10° of mean
trajectory error and no outcome. The ghost still diverges to
~53–55° — far above the open-loop range (21–24°). What the ghost did
**not** fix is everything else that differs: the state channel (the
twin's own joints fed back, including the gripper calibration mapping
and the initial-pose offset), and the fact that once the twin's arm
departs from the demo's configuration, recorded frames depict a
different arm than the one being controlled — the compounding loop
(candidate (a) of the diagnosis) keeps running through the state
channel and the dynamics/contact.

## 5. Verdict

**Observation mechanism: MEASURED.** Two effect sizes, both against
a zero noise floor:

1. **Step 0 (images alone):** median **12.2° (C-exp) / 9.1° (FL)**
   mean-abs per joint-step on the first action chunk — the policies
   are strongly observation-sensitive to the twin's rendered views
   before any dynamics enter. Candidate (e) of the diagnosis is
   promoted from "documented, unmeasured" to **measured**.
2. **Closed loop (images restored):** the ghost recovers only a
   small fraction of the closed-loop divergence and no successes.
   Per the diagnosis's own decision rule, the ghost branch is
   **negative**: the image contract is **incriminated as a real
   contributor and exonerated as the sole cause**. Dynamics/contact
   (candidate (c), already measured at the settle stage) and the
   state channel move up to co-leads.

**Qualifier (verifier finding, carried in full):** "image contract"
here bundles everything about the images as fed — scene appearance
**plus** resolution/aspect/resampling (twin renders are square
256×256; recorded frames are native 576×1024 / 720×1280). The
measurement should not be quoted as a pure scene-realism effect.

## 6. What this means for the §3.4 registered downgrade path

§3.4 step 3 demands a *measured* modality mechanism before the
ordering control can be re-examined. The probe supplies one — and
bounds it. The step-0 image effect (0.16–0.21 rad median) is a real
modality defect localised to the observation contract, of the kind
§3.4 contemplates. But the ghost result blocks the stronger claim
that the observation contract *explains* the ordering failure:
with recorded images the policies still diverge and still score
zero. A downgrade argument built on the image contract alone would
overstate what was measured. The remaining registered route runs
through the co-lead mechanisms: the decisive follow-on is a
**full-input ghost** (recorded images **and** recorded state fed at
every chunk, actions executed in the twin), which separates the
state-feedback loop from execution/dynamics divergence; together
with the diagnosis §5 oracle-pose probe it would complete the
attribution §3.4 needs. Neither follow-on is registered or run here.

## 7. Verification

- **Step 1: CERTIFY-WITH-FINDINGS** (independent verifier, separate
  context). All 16 cells recomputed from the saved chunks — zero
  discrepancies > 1e-6 on any field; A2 = A bit-for-bit in every
  cell; A/B state identity and frame-difference confirmed in code
  and data (99.3–99.8% of pixels differ per camera after resize).
  The two findings are the §3 and §5 qualifiers above.
- **Step 2: CERTIFY-WITH-FINDINGS** (independent verifier, separate
  context). All headline claims — both ghost error profiles, both
  mechanical outcomes, and the L1/offline comparisons — reproduce
  exactly under independent recomputation from the saved ghost
  trajectories (gripper via the harness's own `grip_rad_to_deg`);
  `objects_final` matches the trajectory object arrays, and the code
  audit confirms only recorded frames were fed, with the twin state
  channel and the harness actuation/scoring contract unchanged. The
  findings are the interpretation qualifications carried in §4–§5
  (the ghost restores images only — the state channel and dynamics
  remain the twin's — so a negative ghost bounds, and does not
  erase, the Step 1 step-0 measurement), plus two plumbing findings
  about the probe code itself, neither affecting any number:
  (i) `probe_ghost.py`'s built-in L1 comparison silently returned
  null — its glob (`…__ep<ep>_nominal*.npz`) and variant filter
  (`"nominal"`) do not match the battery's actual file/schema
  (`…__ep411_0_nominal.npz`, variant `"0"`); the §4 L1 numbers were
  computed separately from the L1 traj/rollouts files and the
  verifier reproduced them directly from those files, but they were
  not produced by the probe code as written (fix the glob/filter
  before any reuse); (ii) the ghost omits the battery runner's
  generalised solver-failure check, OR-ing only v1.1 `evaluate`'s
  flag with non-finite qpos — immaterial here (both episodes end
  with finite, in-workspace objects at z = 0.0149, so
  `solver_failure=False` stands under either version).

## 8. Execution record

Step 1 ran under the job guard (`runs/observation-probe-step1`,
logonly breaker; mirror FAIL rc=255 — the known infrastructure gap,
logged, telemetry on-box). The ghost launch was refused by the
guard preflight three times (load1 14.83 > 14; MemAvailable
57,998 MB < 60,000 MB — Phase 6 batch 3 and the Q-A smoke holding
the box); refusals honored, launch retried on a 5-minute loop and
accepted on attempt 2 of the loop (`runs/observation-probe-ghostR2`).
No scene, criterion, or harness camera code was modified; no
training; no HF writes.

*Artifacts: DGX `zk0-study001-twin-v2-20261122/probe/` —
`probe_step1.py`, `extract_demo_frames.py`, `probe_ghost.py`,
`work/step1_results.json`, `work/chunks_*.npz`, `work/twin_*.npy`,
`work/demo_*.npy`, `work/ghost_*.npy`, `work/ghost_result_*.json`,
`work/ghost_traj_*.npz`.*
