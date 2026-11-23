# Twin-Replay Workstream — Progress Checkpoint (2026-11-22)

Coordinator log. Updated at each phase boundary so an interruption loses
minutes, not the workstream.

- 17:15 CT — Founder directive received. Workstream = Phases 1–5 only.
- Recon done: eval records at DGX
  `/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122/`
  (records/*.npz hold gt/pred T×6 degrees; results/ holds FINAL judgments;
  sample/ holds first/final frames for 25 episodes only — frames for all 100
  must be extracted from dataset videos). Dataset:
  `/home/openclaw/muse-work/zk0-onarm-20261117/data/armnetbench_v01_lerobot_so101`
  (2499 eps, 8 tasks — 3 are SEALED BENCH-002 tasks, out of scope).
  Sim model: `/home/openclaw/muse-work/zk0-sim-v01-20261115/sim/model/`
  (so101_new_calib.xml + scene_tabletop.xml prior art; table top z=0, arm
  base at origin in that scene). Sim venv:
  `/home/openclaw/muse-work/zk0-sim-v01-20261115/.venv` (MuJoCo 4.0.1).
- Scouted scenes from top-view first frames (coordinator, direct view):
  common workspace = WOWROBO dark mat inside a tan cardboard enclosure,
  SO-101 arm entering from the near edge. block_stack: orange + purple cubes
  on the mat. eye_drops: blue/white Eye Drops carton on a white raised
  platform; clear plastic basket at left. ring_insert: cream circular board
  with 6 recesses + central wooden peg; ring starts near the peg. tool_insert:
  black/red foam tray of chrome socket extension bars (labeled slots) + one
  loose tool on the mat in front of the tray.
- DGX workdir created: `/home/openclaw/muse-work/zk0-study001-twin-20261122/`
  {code,scenes,validation,frames,logs,docs}.
- Pre-registration written BEFORE any validation run:
  `twin-replay-preregistration-20261122.md` (outcome-agreement primary,
  final-position secondary 3.0 cm / ring 2.0 cm, certification bar ≥80%
  outcome agreement per task, honesty rules, grasp device declared).
- DGX training is live (chain/specialists, ~87% CPU ×4 procs, load ~5.8/20
  cores) — twin work CPU-only, niced.

## Phase status

- [ ] Phase 1 — scene inventory (Builder A)
- [ ] Phase 2 — physics parameter table (Builder A)
- [ ] Phase 3 — per-episode initial states (Builder B)
- [ ] Phase 4 — twin build + demo-replay validation (Builder C)
- [ ] Phase 5 — independent verification (separate context)
- [ ] Docs mirrored + committed to study001/fl-substrate-20261121
- [ ] Completion report to parent (incl. costed Phase 6/7 plan)
