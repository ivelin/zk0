# Study 001 Twin-Replay — Independent Code Verification, twin-suite v1.1 (V-Code)

Verifier: V-Code, fresh independent context (did not write any code under audit).
Date: 2026-11-22 (~19:10 CST, DGX clock). Scope: CODE class only.
Ground truth: docs/twin-replay-preregistration-20261122.md,
docs/twin-replay-amendment1-20261122.md. Builders' claims
(docs/builder1-v11-harness.md, changelog v1.1 entries) were treated as claims
to check, never as evidence. All line citations re-derived from the raw files
on the DGX at audit time. CPU-only, nice -n 10, DGX training undisturbed.
No full re-validation was run (none exists for v1.1 yet); this certifies code,
not results.

## VERDICT: NOT-CERTIFIED (code class) — one blocking defect, narrowly scoped

Four of five tasks' v1.1 code is correct and every other check passes,
including F-02, F-05, the peg-jitter removal, the clipped sampler, the global
timestep fix, and the absence of any per-episode/per-variant special-casing.
Certification is withheld for ONE defect class: **cable_clip variant scoring
is internally inconsistent as coded** — the scene builder jitters the clip
block per variant while evaluate() scores the channel against the nominal
pose-file position, so cable variants k>=1 are judged against a phantom clip
(Check 4/5 finding V11-01). On an all-FAILURE recorded task this biases
variant agreement upward, the same direction of error as v1's F-12. The
remediation is small and specified at the end; everything else in this report
is written so a re-audit after that fix can be short.

---

## Check 1 — F-01 closure (primary criteria vs preregistration)

PASS for block_stack, eye_drops_to_basket, ring_insert, tool_insert.
PARTIAL for cable_clip (see V11-01/V11-02).

- contacts_with() is now live code, not dead: defined replay.py:361-370,
  called at replay.py:391 (block), :407 (eye), :454 (tool), :485 (cable).
- block_stack (replay.py:382-401): success = centre within lower top
  footprint inset 25% (:389-390, (e/2)*0.75 per axis) AND contact with lower
  cube (:392) AND upper contact set subset of {lower} (:393), which with
  :392 means exactly {lower} — the registered "no other support contact".
  The v1 silent proxies are gone from the decision: face gap dz is computed
  at :388 and recorded only as diagnostic "face_gap_m" (:395); the v1
  6 mm / e*1.2 / 0.004 m gates no longer exist. Minor residuals (non-blocking):
  contact is with the lower cube body, not literally its top face — mitigated
  by upper:=higher-z (:384) plus the 11.25 mm inset, which a side-adjacent
  cube (~30 mm centre offset) cannot satisfy; "rests stably" is read at one
  instant at settle end, no velocity test (same as any settle-read design).
- eye_drops_to_basket (replay.py:402-419): centre inside basket interior
  footprint (outer/2 - wall thickness, :405-406) AND support test — basket
  contact present (:408), no platform contact (:409), no world (mat/floor)
  contact (:410); success :411. Matches the registered support definition.
- ring_insert (replay.py:420-442): engaged = planar ring-to-peg <=
  RING_R_IN - PEG_R = 0.003 m physical clearance (:432-433) — v1's silent
  +5 mm +2 mm are gone. low_ok = 0 <= ring_bottom - board_top <= 0.010
  (:435) — v1's 5 mm interpenetration credit is gone. Peg read from the sim
  body (:427), correct now that the peg is unjittered. Residual (minor):
  ring_bottom assumes a flat ring (centre z - THICK/2); a ring resting on
  edge would be mismeasured — it would also fail engagement in practice.
- tool_insert (replay.py:443-460): centre-in-slot now in the slot frame
  (yaw rotation :450-453; v1's world-axis bug fixed), axis <= 20 deg (:448,
  abs-dot, so 180-deg flip counts as aligned — consistent with an undirected
  slot axis), silent z-gate replaced by tray contact (:454). Registered
  "resting in the tray" is now a contact test. Carried residual: SLOT_SIZE
  itself remains a LOCAL-CHOICE (twin_params.py:95), never sourced from the
  recordings — pre-existing, unchanged by v1.1, noted in v1 as F-04.
- Position values as diagnostics only: confirmed for block (above). The
  remaining position terms in primaries are the registered ones themselves
  (footprint containment, ring clearance boolean, slot containment, cable
  channel volume). No extra-mm allowance survives in block/eye/ring/tool.
- Cross-task residual (minor, all tasks): evaluate() never tests whether the
  manipulated object is still weld-attached at settle; an object suspended
  by the grasp weld while touching the target body could in principle pass a
  contact test without resting. still_attached_at_end is recorded in the
  result (replay.py:298) but not consulted by evaluate(). No evidence it
  fires in practice; flagged for the re-validation verifier to spot-check.

### V11-02 (MINOR, cable) — channel tolerances retained from v1, still unregistered
replay.py:483-484: in_channel allows |local x| <= 0.012, |local y| <=
CLIP_CHANNEL_W/2 + 0.002, |z - chan_z| <= 0.012 — the same tolerances v1's
F-01 flagged as silent. Builder 1's changelog #8 calls the new element
(retention = in-channel AND contact with clip_block body, replay.py:485) a
documented modelling device for a non-articulating sprung clip; that device
is reasonable and is documented in code (:342-352) and changelog #8. But the
channel VOLUME tolerances themselves were never registered or amended.
Measured against the built jaw geometry (build_scenes.py clip_jaw_a/b:
centres y = +/-(cw/2+0.004), half-extents (0.012, 0.004, 0.006)): the x bound
equals the jaw half-length exactly and the y bound (+2 mm) reaches exactly
the jaw inner faces' outer edge, while the z bound (+/-12 mm) is twice the
jaw half-height (+/-6 mm). Small, mostly geometry-tracking, but formally
unregistered — needs an amendment line, not silence.

## Check 2 — F-02 closure (secondary criterion)

CLOSED. replay.py:499-509 thresholds per variant: 0.030 m for
block/eye/tool (:500-501); ring 0.020 m when not engaged, secondary_pass=True
when engaged (boolean governs, :502-505); cable exempt, secondary_pass=None
(:506-508). secondary_threshold_m emitted per variant (:509).
run_validation.py propagates nominal_secondary_pass + secondary_threshold_m
per episode (:110-111) and nominal_secondary_passes per task (:119, :134);
full per-variant values ride in variants_detail. Nominal is only taken from
a valid variant (defect/solver-failure variants `continue` before the
nominal assignment, run_validation.py:88-100) — correct.
Consistent with the preregistration bar (certification names outcome
agreement only) + Amendment 1 (secondary assessed on nominal): the secondary
still does not gate "reproduced" — but it can now FAIL VISIBLY, which is
what F-02 demanded. Minor: evaluate() hardcodes 0.030/0.020 literals;
twin_params.py:150-152 SECONDARY_THRESH_M defines the same values and is
referenced nowhere — a duplicated source of truth that can silently diverge.

## Check 3 — F-05 closure (replay fps)

CLOSED, independently verified at the source. I read
/home/openclaw/muse-work/zk0-onarm-20261117/data/armnetbench_v01_lerobot_so101/meta/info.json
myself: top-level "fps": 20, and video.fps = 20 for front/top/wrist.
twin_params.py:24-28 reads FPS from exactly that file at import
(fallback 20.0 = the verified value). The code path uses it:
replay.py:203 steps_per_frame = round((1/P.FPS)/P.TIMESTEP); a live import
under nice -n 10 printed FPS 20.0. Minor: the bare `except` fallback would
silently mask a missing/corrupt metadata file — acceptable only because the
fallback equals the verified value; a loud failure would be cleaner.

## Check 4 — Cable solver-failure detection, timestep uniformity, special-casing

Detection/exclusion/reporting: PASS. replay.py:471-488 — solver_failure if
any segment non-finite (:474-475), planar radial > CABLE_MAX_RADIAL_M
(:478), or z outside [CABLE_Z_MIN_M, CABLE_Z_MAX_M] (:478); bounds from
twin_params.py:155-157 (1.0 m / -0.05 / +0.50). success requires seated AND
not exploded (:488), so an exploded variant can no longer agree by
FAILURE==FAILURE. run_validation.py:92-95 excludes solver-failure variants
from valid/agreeing denominators and logs them as defects; per-episode
n_solver_failures (:112) and per-task solver_failure_variants (:133) are
reported. Note: the summary reports the COUNT; an explicit rate fraction is
not computed (derivable from episodes_out defects + valid totals).
Timestep: PASS, uniform. twin_params.py:29 TIMESTEP = 0.0005; consumed
globally at replay.py:57 (model.opt.timestep override) and build_scenes.py
XML <option timestep> (a current scene XML reads timestep="0.0005"). No
per-task/per-episode timestep branch exists.
Special-casing scan: PASS (the automatic NOT-CERTIFIED trigger does NOT
fire). Grep + full read of replay.py / build_scenes.py / run_validation.py /
twin_params.py: no episode-id conditionals, no per-episode or per-variant
thresholds, no episode-keyed tables in the changed code. Episode numbers
appear only in docstring proof narratives (replay.py:34-41, :190). Branching
is by task and by object class only (e.g. centre_bound by object type,
replay.py:198-202 — pre-existing v1 grasp device, unchanged by v1.1).
Cable build-defect skip for cable_seg bodies (replay.py:114) is per-TASK,
pre-existing, disclosed in code.

### V11-01 (MAJOR, BLOCKING) — cable clip reference: scene jittered, evaluator nominal
make_draws jitters the CLIP BLOCK for cable_clip variants:
build_scenes.py:124-126 bodies list for cable_clip is ["clip_block",
"cable"], with draws at :129-146; build() applies the draw to the scene's
clip_block via jpose (build_scenes.py:545-548). But evaluate() takes the
clip position from the POSE FILE (nominal, unjittered) —
replay.py:462-466 reads (poses)["objects"]["clip_block"], never the sim
body — unlike ring_insert, where the peg IS read from the sim body
(replay.py:427). Verified numerically at audit time (nice -n 10, live
make_draws): cable_clip ep1939 draws displace clip_block by up to the full
clip bound in every variant (v1: (+0.018, +0.040) m; v2: (+0.035, -0.040) m;
v3: (+0.002, +0.011) m; v4: (-0.010, +0.002) m), while the channel test
tolerances are 6 mm in y and 12 mm in x/z (replay.py:483-484). A segment
genuinely seated in the JITTERED clip therefore cannot satisfy in_channel
against the NOMINAL clip position except by coincidence: cable variants
k>=1 (80 of 100 cable variants) are scored against a clip that is not where
the scene put it. All 20 recorded cable outcomes are FAILURE, so the bias
runs toward sim FAILURE = agreement. This also contradicts changelog #12 /
builder1 report Fix 5 ("outcome reference stays at nominal", "jitter applies
to the manipulated object + sampled parameters only") — for cable_clip the
reference fixture IS jittered in the scene and NOT tracked by the evaluator.
Fix (either): read the clip_block position/yaw from the sim body in
evaluate() exactly as ring reads the peg, or remove clip_block from the
jittered bodies so scene and evaluator agree at nominal. Then re-run the
cable validation set. Until then cable variant agreement statistics from
this code are not meaningful, and the code class cannot be certified.

## Check 5 — Peg+board jitter removal

CLOSED for ring_insert. grep fixture_jitter across code/ = 0 hits;
make_draws (build_scenes.py:153-156) explicitly emits none; build() places
board/peg at fixture nominal with the comment at build_scenes.py:471
("no fixture jitter -- peg+board at nominal, always"); live-regenerated
draws for ring ep7 contain no fixture_jitter key. Evaluator reads the peg
from the sim body (replay.py:427), so scene and criterion agree.
Caveat: the same defect class survives for cable's clip_block — see V11-01.

## Check 6 — Sampler clipping (Amendment 1, +/-1 sigma)

CLOSED. build_scenes.py:141-146 _clip_gauss clips rng.normal(0, sigma) to
[-sigma, +sigma] for xy and yaw. Live check (nice -n 10): make_draws for
block ep411, ring ep7, cable ep1939, tool ep622, eye ep1609, all k — 28
body-draws checked against each pose's sigma_xy_cm / sigma_yaw_deg (cable
layout 8 cm), 0 out of bounds; several draws sit exactly AT the bound
(e.g. cable +/-0.08), confirming clipping is active, not just tight sampling.
Carried residuals from v1 F-13 (unchanged, minor): poses lacking sigma fall
back to a silent 3.5 cm (build_scenes.py:136); twin_params.py:142 JITTER
tier table is still dead code with a stale comment (:137-141) describing
uniform tier draws the sampler does not use.

## Check 7 — param_table_hash coverage

SUBSTANTIALLY CLOSED, with enumerated misses. build_scenes.py:32-41
HASH_FILES (fixed order, name+contents): code/twin_params.py,
code/build_scenes.py, code/replay.py, code/run_validation.py,
code/calibration.py, code/finalize.py, docs/physics-parameters.md. Live
value at audit: 4d8d350b6907427c — identical to builder1's quoted hash,
confirming no code drift since the builder report. v1's failure mode
(identical hash across runs 1-3 despite code changes) is fixed for the
run path.
Still NOT covered (finding V11-03, MODERATE):
(a) The pose-pipeline code that produces the scene JSONs the hash purports
    to stamp: code/patch_finalize.py and code/patch2.py (v1 F-07 hand-pick
    tables) and the v1.1 code/builder2_stage*.py repair pipeline — finalize.py
    is hashed, its patch siblings are not.
(b) ALL scene/calibration DATA: scenes/calibration_fk_affine.json,
    scenes/calibration_v2.json, scenes/detections_px.json, fixtures and the
    per-episode pose JSONs themselves. A pose or calibration edit changes
    every scene and score while param_hash stays constant.
(c) The dataset metadata file twin_params reads FPS from (low risk; value
    verified = 20).
Also changelog #13's claim "every code file the pipeline executes" is
inaccurate in both directions: calibration.py is a standalone one-off script
nothing imports, while the files in (a) do execute in suite generation.

## Check 8 — Changelog (v1.1 entries, COORD-1, completeness)

Builder 1 entries #8-#16: contemporaneous-plausible — PASS, a real repair of
v1's F-09. File mtimes corroborate: replay.py 18:55:20 (entries #8/#9/#10 at
18:55), run_validation.py 18:56:12 (#13 at 18:56), twin_params.py 18:59:23
(#14 timestep at 18:59), build_scenes.py 19:01:14 (#15 re-application at
19:01), builder1 report 19:01:59. Every code change I found in the v1.1 diff
surface has an entry: criteria #8, secondary #9, cable detection #10, FPS
#11, peg/sampler #12, hash+reporting #13, timestep #14, clobber #15, smoke
#16; Builder 3's colours/tray/docs have their own entries (duplicate #8/#9/#10)
and Builder 2's pose work its B2-* entries.
COORD-1: ACCURATE. #8/#9/#10 do each appear twice with different content
(Builder 1 harness vs Builder 3 scenes), B2-* is a separate series, and the
note correctly states numbering was left as-written. Disambiguation by
timestamp+author works.
Finding V11-04 (MINOR): two timestamps are NOT contemporaneous. The
changelog file's own mtime is 19:08:34 and the DGX clock read 19:10 at
audit, yet entry B2-5 is timestamped 19:12 and COORD-1 19:14 — both in the
future relative to the file that contains them; docs/builder2-v11-poses.md
(mtime 19:07:47) likewise predates the 19:12 completion its changelog entry
announces. Small (minutes, not v1's ~2 h), but "written contemporaneously"
(builder1 report) is not true of these two entries; they read as
pre-written expected times.
No orphan code changes found: no v1.1 code change lacks a changelog entry.
Bookkeeping re-checks from v1: F-06 stale closedness doc — FIXED
(twin_params.py:40-54 now documents the inverted direction, matching
replay.py:48). F-08 (changelog #3a misdescribes the block final-pose rule) —
NOT corrected; v1 entry #3 stands as written. F-07 (per-episode hand picks
in finalize/patch files) — the files are unchanged from v1; v1.1's B2 audit
discloses and bounds the practice for tool poses, but the changelog itself
still never mentions the finalize/patch hand-pick tables. Both remain open
as bookkeeping items (poses class owns the substance).
Stale text (minor): replay.py:5 header still says "(500 Hz)"; replay.py:10-12
still claims criteria "EXACTLY as written" — now true for 4 tasks, not cable
(V11-01).

## Check 9 — Determinism (code level)

PASS, no new nondeterminism introduced. make_draws seeds per variant via
SeedSequence([VARIANT_SEED, crc32(task__ep), k]) (build_scenes.py:122-123),
independent of run order/worker count; the _clip_gauss closure consumes the
single per-variant RNG in fixed bodies-list order (:124-149).
run_validation.py uses ProcessPoolExecutor.map (:74-75), which preserves
job order; episode output is sorted by (task, episode) (:82) and per-task
aggregation iterates that sorted order. contacts_with returns a set, but
every use is membership/subset or sorted() for output (replay.py:393,
:399, :416). Grasp loop iterates weld_slots in model (XML) order;
dict/set iteration never feeds a numeric result unsorted. Builder 1's
byte-identical double-regeneration claim for block ep411 + ring ep7 is
consistent with this code path and with the unchanged param_hash; I did not
independently re-run full replays (CPU budget / no re-validation exists yet).

---

## Remediation to certification (complete list)

BLOCKING:
1. V11-01 — make cable scene and evaluator use the SAME clip reference
   (read clip_block from the sim body in evaluate(), as ring does for the
   peg; or stop jittering clip_block). Re-run cable set afterwards.
NON-BLOCKING (record as findings under CERTIFIED-WITH-FINDINGS once 1 lands):
2. V11-02 — register or amend the cable channel volume tolerances.
3. V11-03 — extend param_table_hash to patch_finalize.py, patch2.py,
   builder2 pose pipeline, and the calibration/pose data JSONs (or rename
   the claim to match the coverage).
4. V11-04 — correct B2-5 / COORD-1 timestamps to actual write times.
5. Evaluate() to consume P.SECONDARY_THRESH_M instead of literals; remove or
   use the dead JITTER table; fix replay.py:5 "(500 Hz)".
6. Correct v1 changelog entry #3 per F-08; disclose F-07 hand-pick tables in
   the changelog.

## Per-check summary

1. F-01: CLOSED for block/eye/ring/tool; cable PARTIAL (V11-01, V11-02).
2. F-02: CLOSED.
3. F-05: CLOSED (fps = 20 verified in dataset metadata, code reads it).
4. Cable detection/timestep/no-special-casing: PASS, except V11-01.
5. Peg+board jitter: CLOSED (cable analogue = V11-01).
6. Sampler clipping: CLOSED (live draws verified, 0/28 out of bounds).
7. param_hash: SUBSTANTIALLY CLOSED; misses enumerated (V11-03).
8. Changelog: Builder 1 entries contemporaneous & complete; COORD-1
   accurate; B2-5/COORD-1 timestamps impossible (V11-04); F-08/F-07
   bookkeeping still open.
9. Determinism: PASS at code level.
