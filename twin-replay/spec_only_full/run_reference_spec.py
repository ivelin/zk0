#!/usr/bin/env python3
"""Spec-only solvability driver - twin-suite v2, Amendment 3 (2026-11-23).

Runs the FROZEN reference controllers (frozen set: refcore.py +
run_reference.py; rev2.1 set: refcore2.py + run_reference3.py) on
spec-only (Tier C) scenes from the FULL gate suite. GENERATION:
ORACLE-GUIDED (Amendment 2 s.A2.2 route b).

The frozen modules are NOT modified. This driver patches, in this
process only:
  - refcore{,2}.ensure_scene -> the staged spec scene XML for this run
    (a copy of the canonical scene file with ONLY the arm-include path
    rewritten to this machine's arm copy; geometry/physics identical,
    canonical sha256 recorded in the output record);
  - build_scenes.load_episode_poses -> an empty record (Tier C has no
    episodes; the frozen evaluator uses it only for the episode-only
    secondary criterion and for fallbacks that read sim bodies first);
  - build_scenes.load_fixtures -> twin_params.DEFAULT_FIXTURES (the
    same defaults the spec generator placed fixtures from) overridden
    by this scene's manifest target_specification where the frozen
    evaluator reads a fixture (tool slot; basket xy).
Controllers run with gt=None: no demo exists at Tier C, so the
controllers' demo-anchored IK seed branches are absent BY
CONSTRUCTION (recorded here and in the solvability register; the
controller code is otherwise executed exactly as frozen).

Post-processing of the record (this driver's own fields only; the
frozen record fields are untouched except as noted):
  - secondary_pass := None (the secondary criterion is episode-suite-
    only per s.A2.1: it compares against a recorded final position a
    Tier C validator does not have);
  - scene identity: scene_id / scene_index / suite / tier / controller
    version / canonical scene sha256 are added; "episode" is set to
    the scene_id string.

Usage: run_reference_spec.py <frozen|rev21> <task> <scene_index> <out.json>
Env:   SPEC_MANIFEST  path to the full suite's suite_manifest.json
       SPEC_STAGED    dir holding the staged scene XMLs
"""
import copy, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MANIFEST = Path(os.environ["SPEC_MANIFEST"])
STAGED = Path(os.environ["SPEC_STAGED"])


def main():
    ver, task, sidx, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    if ver == "frozen":
        import refcore as RC
        import run_reference as DRV
        ctrl_label = "frozen (refcore.py + run_reference.py)"
    elif ver == "rev21":
        import refcore2 as RC
        import run_reference3 as DRV
        ctrl_label = "rev2.1 (refcore2.py + run_reference3.py)"
    elif ver == "rev3":
        # Ring executor revision (rotation-constraining seat), from the
        # executor-revision workstream; ring_insert only. Uses the
        # frozen refcore, exactly as that workstream's own spec adapter
        # (run_reference4_spec.py) does.
        assert task == "ring_insert", "rev3 exists only for ring_insert"
        import refcore as RC
        import run_reference4 as R4
        ctrl_label = "rev3 (refcore.py + run_reference4.py, rotation-constraining seat)"
    else:
        raise SystemExit(f"unknown controller version {ver}")
    import build_scenes as B
    import twin_params as P

    man = json.loads(MANIFEST.read_text())
    entry = man["tasks"][task]["scenes"][sidx]
    want_k = int(os.environ.get("SPEC_VARIANT", "0"))
    nominal = [v for v in entry["variants"] if v["k"] == want_k][0]
    staged_xml = STAGED / nominal["scene"]
    assert staged_xml.exists(), f"staged scene missing: {staged_xml}"

    # ---- in-process patches (frozen code untouched on disk) ----
    RC.ensure_scene = lambda t, e: staged_xml

    def _poses(t, e):
        return {"final": {}, "objects": {}, "fixtures_b": {}}
    B.load_episode_poses = _poses

    def _fixtures(t, poses=None):
        fx = copy.deepcopy(P.DEFAULT_FIXTURES.get(t, {}))
        tgt = entry["target_specification"]
        gt = entry["ground_truth_initial_state"]
        if t == "tool_insert":
            fx["slot"] = {"pos": [tgt["slot_centre_xy"][0], tgt["slot_centre_xy"][1],
                                  tgt["slot_floor_top_z"]],
                          "yaw_deg": tgt.get("slot_yaw_deg", 0.0)}
        if t == "eye_drops_to_basket" and "basket" in fx and "basket_pos_xy" in gt:
            z = fx["basket"]["pos"][2] if "pos" in fx["basket"] else 0.03
            fx["basket"] = {**fx["basket"],
                            "pos": [gt["basket_pos_xy"][0], gt["basket_pos_xy"][1], z]}
        return fx
    B.load_fixtures = _fixtures

    if ver == "rev3":
        fn, weld = R4.do_ring_rev3, False
    else:
        fn, weld = DRV.TASKS[task]
    r = RC.Ref(task, sidx, use_weld_device=weld)
    try:
        verdict, notes = fn(r)
    except Exception as e:
        import traceback
        verdict, notes = "ERROR", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    rec = r.record(verdict, notes)
    rec["executor"] = "v1.1-weld-device" if weld else "padded-arm scripted (pinch-centre)"
    rec["secondary_pass"] = None
    rec["secondary_note"] = "episode-suite-only criterion (A2.1); not applicable to spec-only scenes"
    rec["episode"] = entry["scene_id"]
    rec["scene_id"] = entry["scene_id"]
    rec["scene_index"] = sidx
    rec["suite"] = man["suite"]
    rec["tier"] = "C (spec only)"
    rec["controller"] = ctrl_label
    rec["scene_sha256"] = nominal["sha256"]
    rec["scene_variant"] = f"k={nominal['k']}" + (" (nominal)" if nominal["k"] == 0 else "")
    rec["variant_kind"] = nominal.get("kind")
    rec["staged_include_note"] = ("staged copy: ONLY the arm-include path differs "
                                  "from the canonical scene file (machine-local arm copy)")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(json.dumps({"task": task, "scene_id": entry["scene_id"], "controller": ver,
                      "verdict": verdict, "sim_success": rec["sim_success"],
                      "notes": notes}, default=str))


if __name__ == "__main__":
    main()
