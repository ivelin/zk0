# Study 001 Twin-Replay — Asset Inventory (Amendment 1 §A1, Builder C)

Surveyed 2026-11-22 ~17:55 CT. Rule: tested assets are the default building
blocks; anything built new is labelled NEW + why no tested asset fit.

| Building block | Asset used | Provenance | Testing it survived | Reuse verdict |
|---|---|---|---|---|
| Arm model | `zk0-sim-v01-20261115/sim/model/so101_new_calib.xml` + 13 mesh STLs in `sim/model/assets/` | The Robot Studio SO-101 (Apache-2.0), vendored in the zk0 sim project; new-calibration zero convention | The project's own sim v0 pick-and-place demo (`zk0_sim_demo.py`, incl. FK-solved grasp poses + kinematic-carry grasp); the Study 001 FK-replay work (joint mapping degrees→radians into identically named joints proven there); the Nov 15 demo + audit logs (per the twin-replay completion report's V-Objects note: the underlying testing is real -- Nov 15 demo + audit logs; the earlier 'SIM-1/SIM-2 era sim validation' label was uncorroborated and is dropped) | **REUSED as-is** (included, unmodified). This is the single most-tested arm asset in the collection. |
| Arm model (alt copies) | `zk0-sim-v0-20261115/sim/model/so101_new_calib.xml` (identical role, earlier copy); `zk0-sim-eval2-20261115/assets/SO101/` (new + old calib + camera variant + joints_properties.xml); menagerie copies in `zk0-sim-v01-20261115/menagerie/robotstudio_so101/` and `zk0-sim-eval-20261115/mj/menagerie_so101/` (upstream MuJoCo Menagerie SO-101, incl. scene_box.xml / scene_grasp.xml) | Upstream menagerie / project copies | Menagerie scenes are upstream demo scenes (box grasp), not validated against our recordings; eval2 copies used in eval-harness experiments | Not used: the v01 vendored copy is the one our demo + FK replay actually exercised. Others inventoried as fallbacks if the v01 model misbehaves. |
| Tabletop scene pattern | `zk0-sim-v01-20261115/sim/model/scene_tabletop.xml` | zk0 sim v0 | Ran in the v0 demo (table top z=0, arm base at origin, 40 mm cube pick-and-place completed in sim) | **REUSED as the scene pattern**: same world convention (tabletop z=0, arm at origin), same include mechanism. Twin scenes extend it with mat/enclosure/fixtures. |
| Cube | scene_tabletop's 40 mm cube (single box geom, free joint) | zk0 sim v0 | Pick-and-place demo | **PATTERN REUSED, size NEW**: twin cubes are 30 mm per Builder A's measured estimate from the recordings (the demo's 40 mm was a demo prop, not the Study 001 prop). Geometry mechanism (box + free joint) identical. |
| Kinematic-carry grasp pattern | `zk0_sim_demo.py` Demo class (pin object to gripper while holding) | zk0 sim v0 | v0 demo | Pattern informed the weld design; the twin harness uses MuJoCo weld equalities instead (declared device, Builder A §7) because welds interact with contacts during carry. The demo's GRIPPER_OPEN=1.45 / CLOSED=0.10 rad convention is reused for the visual gripper joint mapping. |
| Grasp/BAM-era scenes | `zk0-simready-shelf-20261115/` (Genesis checks, mug downloads `dl_mugs*.py`, `diag_grasp.py`); `sim-experience-audit-20261115/sim_attempt.py`; `sim-stack-check-20261115/load_test*.py` | Nov 15 sim-readiness work | Load/attempt diagnostics only — no reusable task-scene XMLs for the five Study 001 tasks exist in these dirs (checked: no .xml scene files) | Nothing reusable for these tasks; noted so the absence is on record, not assumed. |
| Repo clone sim folders | `zk0-study001-fl-20261121/` (the ivelin/zk0 working clone) | zk0 repo | `train-fl-simulation.sh` is Flower *federation* simulation, not physics sim; no MJCF task scenes found under the clone (maxdepth-3 scan) | Nothing to reuse for twin scenes. |
| Mat / enclosure / fixtures (platform, basket, board+peg, tray+slot, clip block) | — | — | — | **NEW** (built in `code/build_scenes.py` from Builder A's inventory + parameter table). Why: no tested asset exists for the WOWROBO mat, cardboard enclosure, or any of the five task fixtures — the recordings' props were never modelled in the project's sim collection. Dimensions/friction are Builder A's sourced estimates, labelled as such in the parameter table. |
| Task objects (carton, ring, tool, cable chain, USB) | — | — | — | **NEW** (same reason: no existing assets; built to Builder A's parameter table — ring as 12-capsule composite, cable as 24-segment ball-joint capsule chain per A's declared approximation). |
| Cameras / rendering | MuJoCo offscreen Renderer (EGL) in the sim venv | MuJoCo 4.0.1 | Same venv/render path as the sim project | REUSED (library, not an asset file). |

## Summary

Reused: the SO-101 arm model + meshes (verbatim), the tabletop scene
convention, the cube mechanism, the gripper open/closed convention.
New (no tested asset existed): mat, enclosure, all fixtures, all task
objects. Every NEW item's parameters trace to Builder A's sourced table —
estimation stayed labelled, per Amendment 1.
