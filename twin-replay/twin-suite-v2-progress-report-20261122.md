# Twin-Suite v2 — Progress / Completion Report (2026-11-22/23)

**Founder approval:** 2026-11-22 21:34 CT ("Approve the flip — pre-register twin-suite v2"), plus founder additions 2026-11-22 21:44 CT (validator privacy tiers, solvability validation, perfect-checkpoint definition).
**Governing document:** `sim-gate-design-memo-20261122.md` (independently verified CERTIFY-WITH-FIXES, fixes applied).
**Headline outcome:** The v2 instrument was built and exercised end-to-end, and it **failed its own registered certification control**: in closed-loop twins, all three control checkpoints (base, FL round-30, C-exp) score **0 on the primary mechanical criterion in every task**, so the §3.4 known-ordering control does not reproduce and **no task is instrument-certified**. The standard was not tuned to pass. Candidate gating under v2 remains blocked. Every stage below carries an independent verifier verdict; the §3.4 FAIL was independently verified and stands as settled.

> Mandatory label (pre-reg §8): **funnel verdict — sim bar unanchored to real-arm delta (no calibration pairs yet).** Gate outputs support funnel verdicts and rankings only, never absolute real-world rates.

---

## 1. Phase 1 — Pre-registration (frozen)

- Pre-registration: `twin-suite-v2-preregistration-20261122.md`, written from the verified memo's skeleton — instrument-certification standard (§3.1 parameter audit, §3.2 MuJoCo↔Drake cross-engine agreement, §3.3 envelope stability, §3.4 known-ordering controls with the cross-modality assumption stated), gate protocol (§4: closed-loop, K=5 variants, mechanical criteria primary + frontier judging recorded), three-valued verdicts (§5: PASS = Wilson lower bound ≥60% AND envelope-stable; INDETERMINATE escalates, never rounds), promotion = PASS on every task, §6 control checkpoints, §7 bounds procedure, §8 claims discipline, §9 blinding + per-class verification, §11 v1.1 versioning rules, §12 compute discipline.
- **Frozen commit `bb11d20`** on `ivelin/zk0 @ study001/fl-substrate-20261121` (with the design memo). Decisions SSOT: **Decision 62** appended to the local mirror (rclone syncs to Drive).
- Independent verification: **CERTIFIED-WITH-FINDINGS**; findings closed by **Amendment 1, commit `a8a2729`** (FL source label for the 26/100 figure; subset RNG procedure pinned and reproduced; §1 hash coverage enumerated; base-fails-all scope).
- **Founder additions landed as Amendment 2, commit `b74c308`** (with a Decision 62 addendum). They did not land in the original freeze or in Amendment 1 for a simple sequencing reason: the pre-registration had already been committed when the additions arrived, and Amendment 1 was already taken by the verification fix pass — so the additions went through the amendment process as Amendment 2, never by silent edit. Contents: **A2.1** validator privacy tiers (A/B/C) + the spec-only scene path with a per-component dependence table; **A2.2** solvability validation — a scene counts only with ≥1 executed pass trajectory in the twin (frontier+human review is the realism screen, never the proof; gated/candidate/control checkpoints barred as executors; unvalidated scenes excluded from denominators); **A2.3** perfect-checkpoint definition as the aspirational limit (operational PASS unchanged). No conflict with the memo's skeleton required resolution; the additions extend §3/§4 without altering any frozen threshold.

## 2. Phase 2 — Stage 1 build, per component (builder → verifier verdict)

| Component | Result | Verifier verdict |
|---|---|---|
| §3.1 Parameter audit | 40 parameter rows (17 [MEASURED-EST] / 3 [PROP] / 20 [DEFAULT]); discrepancies D1–D6 confirmed (headline: frozen scenes run timestep **0.0005 s**, not the documented 0.002 s); 10 envelope corners/task frozen | CERTIFIED-WITH-FINDINGS → fix pass → delta **CERTIFIED** |
| MuJoCo rollout harness | Two-process closed-loop harness mirroring the Study 001 eval contract; actuated arm with STS3215-honest limits; two camera defects found and fixed at the v2 layer only (blank `frontcam`, missing wrist camera) | Mechanics **CONFIRMED** |
| Grasp blocker | Vendored gripper geometry cannot pinch-lift the 30 mm cube (independently re-run and confirmed) | **CONFIRMED** |
| Finger-pad fix | Collision geoms ported verbatim from the MuJoCo Menagerie edition of the same SO-ARM100 model into a v2 arm copy; pinch-lift + 2 s hold passes at two of three descent heights; pinch-centre sits ~24.5 mm off the `gripperframe` site (now the targeting standard) | CERTIFIED-WITH-FINDINGS → tip-sphere lineage delta **CERTIFIED-WITH-FINDINGS** (trial outputs not preserved — record gap D1; arithmetic erratum D2 since appended) |
| Drake port part 1 (kinematic) | Same engine-independent inputs; builder's MuJoCo baseline bit-exact vs v1.1 on 16/16 dev episodes; outcome agreement 15/16 (point) / 16/16 (hydroelastic); weld-pin defect class found in all Drake solver flags | CERTIFIED-WITH-FINDINGS |
| Drake actuated build | Actuated URDF at the MuJoCo actuator numbers; pads transcribed to ~0.002 mm; grasp approach-path divergence vs MuJoCo confirmed digit-for-digit (effort-limited PD stalls ~20 mm short on the scripted approach; direct-descent diagnostic passes in both engines) | CERTIFIED-WITH-FINDINGS |
| Drake renderer fix | All Drake renders were **blank** — camera math used the OpenGL −z convention against Drake's z-forward frame (plus .stl visuals ignored by RenderEngineVtk). Fixed; frames content-checked (~850–1,300 unique colours/view vs 1 before); rollout records now carry a `frame_content` statistic | Delta **CERTIFIED** |
| Solvability (A2.2), frozen controllers | 20 dev-subset scenes attempted, one frozen controller set: **8/20 validated** (block 2/4, eye 3/4, ring 0/4, tool 1/4, cable 2/4); cable weld device justified by measurement (pinch-centre face gap 6.15 mm at the joint limit vs cable Ø5 mm — a centred pinch is geometrically impossible) | CERTIFIED-WITH-FINDINGS |
| Controller revision round | One global revision (G1–G9, no per-scene patches, no relaxed limits/criteria): block 411/498 and cable 2130 gained; ring re-grasp hypothesis **falsified with a measured mechanism** (two-point equator grip leaves the ring free to pendulum, test-lift tilts 87–96°); one honest regression (tool 683) | CERTIFIED-WITH-FINDINGS; **register ruling: UNION permitted under A2.2** |
| Solvability errata (§12/§13 + finger-pad D2) | Four narrative corrections re-derived from raw records, append-only with integrity proofs | Delta **CERTIFIED** |
| Spec-only generator (A2.1) | 100 scenes (5 tasks × 4 × K=5), seed 20261126; decisive leakage check passed (generator reads only the spec file + padded arm include); regenerates byte-identical; settle 100/100 clean; spec gaps all declared (incl. G-E5 platform sub-geometry, G-C4 cable seg_0 damping 0.05 vs ball-joint 0.2, value left unchanged) | CERTIFIED-WITH-FINDINGS → hygiene delta **CERTIFIED** |
| Controls prep (Part A) | Envelope-subset solvability under both controller sets; controls manifest pinned **before any control run** (base `5e8d12a6…` / weights `7cd549ac…`; FL A5-repaired weights `5b1cbf29…` verified against the pre-reg pin; C-exp 15,000-step artifact weights `8f6807ed…`, lineage unambiguous) | **CERTIFIED** |
| Controls battery (Part B) | 435/435 rollouts (L1 300 / L2 120 / L3 15), both engines, solver failures 0 — **all controls 0 on the primary criterion; §3.4 FAIL** | CERTIFIED-WITH-FINDINGS; **§3.4 FAIL stands as settled** |

## 3. Solvability register (A2.2, union per verifier ruling)

Validated denominators — dev subset: block 4/4, eye 3/4, ring 0/4, tool 1/4, cable 3/4 (**11/20**). Envelope subset: block 4/4, eye 1/4, ring 0/4, tool 0/4, cable 4/4 (**9/20**). **Combined: block 8/8, eye 4/8, ring 0/8, tool 1/8, cable 7/8 = 20/40 attempted.** The remaining 56 of the 96 gate scenes were never solvability-attempted; extending validation is a separate step. Spec-only suite validated denominator: **0/4 per task** (the generator deliberately does not self-validate; solvability there is future reference-controller work). **ring_insert has no validated scenes under any controller version and therefore no gate.**

## 4. The gate's first control run (Part B readout)

L1 nominal, primary mechanical criterion, solver failures 0 everywhere:

| Checkpoint | block (n=40) | eye (n=20) | tool (n=5) | cable (n=35) | pooled (n=100) |
|---|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 | **0** |
| FL round-30 (A5-repaired) | 0 | 0 | 0 | 0 | **0** |
| C-exp (15,000 steps) | 0 | 0 | 0 | 0 | **0** |

- §5 control verdicts: **FAIL for every checkpoint × task** with a denominator (Wilson uppers 8.8–43.4 pp, all < 60%).
- **§3.4 known-ordering: FAILS as registered.** Base-fails-all holds; FL and C-exp are indistinguishable from base on the primary criterion, so base < FL < C-exp does not reproduce. The verifier proved the zeros are behaviour, not a dead harness: per-checkpoint action trajectories on identical scenes differ by 0.81–2.02 rad max-abs, objects are left unmoved — the policies act, but never form the pinch/placement the criterion requires in closed loop. The registered downgrade path was **not** invoked: it requires a modality mechanism *documented with measurements*, and none was measured in this battery (trajectory error-growth was not computed). Per the pre-reg: "if no mechanism is evidenced, the control FAILS… There is no third outcome and no silent pass."
- Secondary criterion (not gated): block_stack base 3/40 < FL 5/40 < C-exp 7/40 — monotone in the expected direction, far below the bar, and within ±run noise per the verifier's re-runs (one secondary flip in 6); zero on all other tasks.
- §3.2 cross-engine: **15/15 agreement** — at the floor (all FAIL in both engines): consistency evidence, not discrimination evidence. All Drake records non-blank (`frame_content` verified).
- §3.3 envelope stability: **0 outcome flips in 120 corner rollouts** (block FAIL at nominal and all corners; eye pooled-across-corners FAIL matching nominal; per-corner n=1 reads are INDETERMINATE by construction).
- **§7 Bounds Amendment: drafted and arithmetic-verified, NOT registered.** Draft values: cross-engine disagreement bound ≤20% (Wilson upper of 0/15); ordering noise bands block 8.8 / eye 16.1 / cable 9.9 / tool 43.4 / pooled 3.7 pp; envelope criterion stands (0/120 flip dispersion); mechanical/judging disagreement bound honestly left unset (judging layer not run). Registration is deferred because certification failed — with no task certified, no candidate may be gated, and freezing bounds for a non-operating gate would misrepresent its status. The draft lives in `controls-battery-v2.md`.

## 5. Instrument certification status per task

| Task | §3.1 audit | §3.2 cross-engine | §3.3 envelope | §3.4 controls | **Certification** |
|---|---|---|---|---|---|
| block_stack | ✓ | agreement at floor | stable | **FAIL** | **NOT CERTIFIED** |
| eye_drops_to_basket | ✓ | agreement at floor | stable | **FAIL** | **NOT CERTIFIED** |
| ring_insert | ✓ | — | — | no validated scenes | **NO GATE** |
| tool_insert | ✓ | agreement at floor | stable | **FAIL** (n=1 scene — uninformative denominator) | **NOT CERTIFIED** |
| cable_clip | (v1-lineage) | — | — | **FAIL** under v2 | **NOT CERTIFIED under v2** — the v1.1 Stage-0 *scene* certification stands separately and is untouched by this result |

Suite identity: **param_hash_v2 `311adc26fe06c3b5`** (`suite_manifest_v2.json`, inputs enumerated per Amendment 1 A1.3). Record committed to `ivelin/zk0 @ study001/fl-substrate-20261121`, commit **`d277750`** (211 files, ls-remote verified), labeled a non-certified research record. **Publication: NONE** — the certified-only publication rule bars any HF release of the v2 suite; the v1.1 cable_clip HF dataset (pinned `579ad96b…`) is unaffected.

## 6. Open items

1. **OI-1 / OI-2 (founder, physical):** mat dimensions and cube material/mass remain unmeasured in any record — the physical setup belongs to the dataset creators; the audit carries labeled estimates (mat 550×400×3 mm ±20%; cube 20 g, envelope 5–35 g). Resolvable only via the ArmnetBench creators or hands on the original setup.
2. **DGX job-guard off-box mirror is broken** (mirror target connection-refused; this VM has no sshd/tailnet listener). Every v2 batch set the target, logged MIRROR FAIL rc=255, and used **no override**; telemetry exists on-box only. Per pre-reg §12 this is **blocking for Stage 2** and needs an infrastructure fix (a reachable mirror target) before any candidate gating — independent of the certification outcome.
3. **The §3.4 route decision (founder):** the only registered path that could reopen certification is §3.4 step 3 — *measure* the modality mechanism (trajectory error-growth / compounding-error analysis on the retained battery rollouts) and, if a mechanism is evidenced, apply the registered downgrade/exclusion procedure. Alternatives are instrument changes (a v2.1 suite version under the §11 versioning rules — e.g. criterion or scene-distribution changes), which would need their own pre-registration amendment. No route was chosen or started in this build.
4. **ring_insert:** no validated scenes (rim-held ring tilts; re-grasp-flat falsified by the pendulum measurement). A different solvability executor class (e.g. human-in-sim) is the remaining registered option.
5. **tool_insert:** 1 validated scene — no meaningful verdict is possible on that denominator regardless of certification.
6. **56 of 96 gate scenes never solvability-attempted** (only dev + envelope subsets were in Stage 1 scope).
7. **Spec-only path:** generator certified, validated denominator 0/4 per task; spec-only solvability runs are future work. Gap G-C4 (cable seg_0 damping 0.05 vs 0.2) awaits a future suite-version decision.
8. **Drake approach-path divergence:** scripted-battery approach stalls in Drake where MuJoCo passes (approach-path contact behaviour); §3.2 was decided on policy-outcome agreement per the pre-reg, but the divergence is on the record for any future Drake-side grasp work.
9. **Record gaps (minor, disclosed):** tip-sphere trial outputs not preserved on the DGX (decision verified, trial table stands on the fix note); battery records carry `checkpoint_id` rather than a weights hash (identity proven indirectly by the verifier); rollouts are not bit-deterministic (~1 cm final-position jitter in re-runs; primary outcomes reproduced 6/6).

## 7. What was NOT done (per instructions)

- No gate runs on any checkpoint beyond the three registered controls.
- No Phase 6 cable_clip checkpoint evals (separate scheduling; the v1.1 authorization is unaffected).
- No HF publication of anything v2 (certified-only rule).
- No changes to any frozen standard, criterion, or threshold at any point — every failure above is reported against the standard as registered.

*Evidence base: `twin-suite-v2-docs/` in this folder (23 documents: pre-registration verifications, per-component reports and verifications, solvability report + errata, controls prep/battery + verifications, suite manifest) and the DGX v2 workdir `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/`. Study-branch record: commit `d277750`.*
