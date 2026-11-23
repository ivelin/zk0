# Twin-Suite v2 — Stage 1 Certification Synthesis: Independent Verification

**Date:** 2026-11-23 · **Verifier context:** independent, separate from the
synthesis builder (pre-reg §9). **Subject:** `stage1-certification-synthesis-v2.md`
+ `stage1-certification-summary-v2.json`, commit `e1fbeba` on
`study001/fl-substrate-20261121` (remote tip at verification time; the shared
DGX clone's local HEAD `334eda3` carries the identical synthesis commit
message/content lineage).
**Method:** every count below was recomputed by the verifier's own scripts
from the raw `rollouts.jsonl` records and `traj_*.npz` files on the DGX
(`/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/controls_battery/`),
not from the builder's `analyze.py` output. Wilson intervals recomputed
(z = 1.96). No rollouts were re-run. Battery outputs were read-only.

> Every gate output carries the §8 label: funnel verdict — sim bar
> unanchored to real-arm delta (no calibration pairs yet).

## Overall verdict: **CERTIFIED-WITH-FINDINGS**

All load-bearing claims reproduce exactly. One minor, non-verdict-bearing
diagnostic figure (§2.2 maximum displacements) could not be reproduced and
is corrected below (Finding F1). Two documentation notes (F2, F3) do not
change any number or ruling.

## Per-claim-group verdicts

### 1. Battery integrity — CERTIFIED
Recomputed: L1 300 records (100/checkpoint), L2 120 (40/checkpoint),
L3 15 (5/checkpoint); total **435/435**, matching `worklist.jsonl` (435
lines) and `completion-manifest.json` (435/435). Solver failures **0/435**;
records with `excluded` true **0/435**; invalid Drake records under the
`analyze.py` `drake_valid` rule **0/15** (chunk-0 unique colours 848–1,376
per view, verifier-recomputed; min non-clear fractions 0.55–1.00 — both
match the synthesis). 435 `traj_*.npz` files present.

### 2. L1 mechanical pass counts + Wilson + secondary ordering — CERTIFIED
Recomputed per task × checkpoint: block_stack 0/40, eye_drops 0/20,
tool_insert 0/5, cable_clip 0/35 for **every** checkpoint; pooled **0/100**
per checkpoint (0/300 total). Wilson 95% uppers recomputed: pooled 0.0370
(synthesis ≤0.037 ✓), block 0.0876, eye 0.1611, tool 0.4345, cable 0.0989 —
all match the summary JSON. Secondary passes recomputed: block_stack
base 3/40, FL 5/40, C-exp 7/40; all other tasks 0; pooled **3 < 5 < 7** ✓.

### 3. Ordering control (§3.4) FAIL + excuse procedure — CERTIFIED
Pre-reg §3.4 read directly: base-fails-all holds in every cell including
cable_clip under A1.4 (base 0/35) ✓; strict ordering base < FL < C-exp
against the A1.1 reference 0/26/70 does **not** reproduce (0 = 0 = 0 in
every cell) ✓. The excuse procedure (steps 1–5) as applied in the
synthesis matches the registered text: step 3 requires a specific
modality mechanism *documented with measurements*; no error-growth or
equivalent measurement exists in the battery records (verifier confirmed
no such field/quantity in any record), so no pair is excused, the step-4
downgrade is correctly not taken, and step 5's FAIL follows. **§3.4 FAIL
is the correct ruling.**

### 4. Envelope stability (§3.3) — CERTIFIED
Recomputed from L2 vs the L1 nominal instance (variant "0", corner
nominal): **0 successes in 120 L2 rollouts; 0 outcome flips in 120
corner-vs-nominal pairs**; every corner verdict FAIL = nominal FAIL.
No verdict is friction-contingent (FAIL at nominal, both friction
corners, and all other corners). The synthesis's floor caveat is
appropriate and is carried in its text.

### 5. Cross-engine (§3.2) + Drake watch item — CERTIFIED (with F2)
Verifier's own pairing (L3 ↔ L1 variant "0"/nominal): outcome agreement
**15/15** (block 12/12, eye 3/3; 5/5 per checkpoint); secondary agreement
**15/15** (recomputed independently — see F2). Resolution (a) is
supported by trajectory evidence the verifier spot-checked directly:
- ep527 C-exp cube_a net displacement: **Drake 0.1213 m (12.1 cm)** vs
  **MuJoCo nominal 0.0374 m (3.7 cm)** ✓ (synthesis: 12.1 / 3.7 cm).
- Cross-engine final-position deltas (from `objects_final`): base max
  0.00008 m, FL max 0.0551 m, C-exp max **0.1197 m** on ep527 cube_a ✓
  (synthesis: ~0.0001 / 0.055 / 0.120 m).
- ep499 C-exp: Drake cubes net 2.6 / 4.4 cm; MuJoCo nominal 5.0 / 1.1 cm ✓.
  ep546: Drake no motion, MuJoCo cube_a 2.3 cm ✓. ep549: Drake 2.4 cm,
  MuJoCo 0.2 cm ✓.
- eye ep1609: carton displacement ≈ 0 in both engines (≤1.5×10⁻⁶ m) ✓ —
  the no-contact failure is not Drake-specific.
- `finger_mesh_failed` (under `actuator_info`) lists the same 3 excluded
  finger-part meshes in the L3 records checked, yet contact demonstrably
  occurs in Drake — supporting "not (b)". No pair disagrees on outcome
  or secondary class — supporting "not (c)". **(a) stands.**

### 6. Per-task rulings + §8 label — CERTIFIED
Each ruling cites a pre-reg basis that says what the synthesis claims:
block_stack NOT CERTIFIED (§3: any failed standard denies certification;
§3.4 FAIL on the full 8-scene denominator) ✓; eye_drops NOT CERTIFIED
(§3.4 FAIL, 4 scenes) ✓; tool_insert NOT CERTIFIED / insufficient
denominator (1 validated scene per prep §2; no pre-reg exception permits
certifying on it; §§3.2/3.3 untested — correctly stated, not passed) ✓;
ring_insert NO GATE per A2.2(iii) verbatim ("a task with no validated
scenes has no gate and is reported not certified"; 0/8 validated) ✓;
cable_clip v1.1 Stage 0 certification unchanged (pre-reg preamble "Kept";
the v2 exercise is reported as a fact, no v2 verdict issued) ✓. The §8
"sim bar unanchored to real-arm delta" label appears on the synthesis
headline and bottom line ✓.

### 7. Open items honestly stated — CERTIFIED
§7 Bounds Amendment not frozen: confirmed — the summary JSON records
`bounds_frozen: false` and the synthesis reports measured agreement
without declaring a pass against any bound ✓. §4 judging layer never
executed in Part B: verifier confirmed **no judging fields exist in any
of the 435 records** ✓. 56/96 gate scenes never solvability-attempted:
prep §2 arithmetic confirmed (20/40 attempted validated register;
96 − 40 = 56) ✓.

## Findings

- **F1 (minor, diagnostic only — no verdict affected).** Synthesis §2.2's
  "maximum object displacement per rollout" figures (base 8.6 cm, FL
  9.0 cm, C-exp 9.8 cm, block variants) do not reproduce under the
  synthesis's own §9 definition (net displacement ‖a[−1] − a[0]‖ per
  `obj_*` array). Verifier-recomputed L1 block_stack maxima:
  **base 8.9 cm, FL 11.3 cm, C-exp 15.7 cm** (max excursion essentially
  identical). The corrected figures strengthen, not weaken, the §2.2
  point (policies are not inert), and no count, bound, or ruling depends
  on them. All other §2.2/§4.2 trajectory figures reproduce exactly
  (block z-rise maxima 1.3 / 1.7 / 5.5 cm; tool z-rise 3.8–4.8 cm; eye
  carton displacement 0 in all 60 L1 rollouts).
- **F2 (documentation note).** The registered `analyze.py` cross-engine
  check compares `sim_success` and `solver_failure` only; the secondary
  agreement 15/15 in synthesis §4.1 is therefore an extension beyond
  `analyze.py`, although §6's extension list does not name it explicitly.
  The figure itself is correct (verifier-recomputed 15/15).
- **F3 (documentation note).** `analyze.py`'s envelope `verdict_stable`
  compares per-corner verdicts computed on n = 4 (block) / n = 1 (eye)
  against the K=5-pooled nominal verdict; the synthesis additionally
  performed the nominal-instance flip check (§5), which the verifier
  reproduced (0/120). No discrepancy in outcome.

## Bottom line

The synthesis is an accurate recomputation of the battery: **Stage 1
certifies no precision task**, on a correctly applied §3.4 ordering-control
FAIL, with §3.2 outcome agreement 15/15, §3.3 envelope stability 0/120
flips, and all open items (§7 bounds unfrozen, §4 judging absent, 56/96
scenes unattempted) honestly stated. **CERTIFIED-WITH-FINDINGS** (F1–F3
above; none alters a verdict). No gate run on any non-control checkpoint
is authorized. All verdicts are funnel verdicts — sim bar unanchored to
real-arm delta (no calibration pairs yet) (§8).
