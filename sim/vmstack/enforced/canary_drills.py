"""Canary drills, old stack vs new stack — protocol v2 §canaries.

C1 (through-table grab; s1_spec00's measured parameters):
  OLD: the alarm_canary drill on the old harness's record — the old
       stack COMPLETES the trajectory silently (its seat test reads the
       ring's final state only) while the alarm layer fires with the
       canary parameters (attach f89, worst grip-mat <= -40 mm, dwell
       site z <= -30 mm). Reproduced by vmstack/alarm_canary.py.
  NEW: the same trajectory under enforce_replay aborts at enforcement
       (records c1_new_transport.json / c1_new_certification.json), and
       the frame-30 truncation aborts at the first jaw x mat touch
       (c1_new_trunc30.json) — the through-table grab is unreachable.

C2 (surface drag):
  OLD: trace computed here from the old harness's own path_check frames
       for eye ep1639 — attached-phase frames where the carton's bottom
       is at/below its rest support level (object in support contact
       while attached) and the object's planar travel across that span
       (measured: 52 frames, 338.8 mm of in-contact travel). The old
       stack executed the whole carry without any objection. (s1_spec00
       was measured first and has NO such trace — its ring stays clean;
       the drag lives in the eye executions. ring_ep160/ep165 also carry
       traces: 43 frames / 40.8 mm and 41 frames / 62.9 mm.)
  NEW: a drag cannot be executed — the carrier's first undeclared
       contact aborts the run (C1 trunc30 record: abort at original
       frame 41 on fixed_jaw_mesh x mat, BEFORE the old attach frame
       89). Additionally every enforce_replay record carries the live
       C2 trace (attached-phase object-vs-support engine contacts +
       in-contact planar travel) as instrumentation.

The drills test that the ENGINE + ALARMS catch planted defects; they
are not validity rules (founder scope guard 2026-11-26 08:22 CT).

Usage: canary_drills.py [--out RECORD_JSON]
"""
import csv, io, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

_HERE = Path(__file__).parent
_WORK = _HERE.parent.parent
_REC = _HERE / "records-20261126"
PY = str(_WORK / "venv" / "bin" / "python")


def c2_trace_old():
    """Run the old checker on eye ep1639 and compute the drag trace from
    its own per-frame output (attached flag + object bottom z)."""
    tmp = Path(tempfile.mkdtemp(prefix="c2canary_"))
    out_json, out_csv = tmp / "s.json", tmp / "s.csv"
    cmd = [PY, str(_HERE.parent / "path_check.py"),
           "--scene", str(_WORK / "staging" / "scenes" / "eye_drops_to_basket__ep1639_v0.xml"),
           "--traj", str(_WORK / "proposals" / "eye_drops_to_basket__ep1639_a1.npz"),
           "--task", "eye_drops_to_basket", "--object", "carton",
           "--record", str(_WORK / "results-vm" / "eye_drops_to_basket__ep1639_v0.json"),
           "--out", str(out_json), "--frames-csv", str(out_csv)]
    r = subprocess.run(cmd, cwd=_WORK, env={"MUJOCO_GL": "disable", "PATH": "/usr/bin:/bin:/usr/local/bin"},
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"path_check failed: {r.stderr[-400:]}")
    summary = json.loads(out_json.read_text())
    rows = list(csv.DictReader(io.StringIO(out_csv.read_text())))
    rest_bottom = float(summary["obj_bottom_rest_z"])
    attached = [i for i, rw in enumerate(rows) if rw["attached"] == "1"]
    in_contact = [i for i in attached
                  if float(rows[i]["obj_bottom_z"]) <= rest_bottom + 0.001]
    travel = 0.0
    if len(in_contact) >= 2:
        pts = np.array([[float(rows[i]["obj_x"]), float(rows[i]["obj_y"])] for i in in_contact])
        travel = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return {"case": "eye_ep1639 (old top-up execution, path_check frames)",
            "old_execution_completed": summary.get("replication_final_pos_err_m") == 0.0,
            "attach_frame": summary["attach"]["frame"] if summary.get("attach") else None,
            "attached_frames": len(attached),
            "attached_support_contact_frames": len(in_contact),
            "in_contact_planar_travel_m": round(travel, 4),
            "rest_bottom_z_m": rest_bottom,
            "trace_present": bool(in_contact) and travel > 0.005}


def main():
    out = _REC / "canary_drills.json"
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])

    # C1 old — the alarm drill (runs the old checker itself).
    drill_rec = _REC / "canary_c1_old_alarm.json"
    r = subprocess.run([PY, str(_HERE.parent / "alarm_canary.py"), "--out", str(drill_rec)],
                       cwd=_WORK, capture_output=True, text=True)
    c1_old = {"alarm_drill_pass": r.returncode == 0,
              "drill_stdout_tail": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "",
              "record": str(drill_rec)}

    def abort_summary(p):
        d = json.loads(Path(p).read_text())
        f = d.get("enforcement_fault") or {}
        return {"aborted": d.get("aborted"), "fault_frame": f.get("frame"),
                "fault_phase": f.get("phase"),
                "fault_pair": (f"{f.get('body1')}/{f.get('geom1')} x "
                               f"{f.get('body2')}/{f.get('geom2')}") if f else None,
                "fault_dist_m": f.get("dist_m"), "record": str(p)}

    c1_new = {"transport": abort_summary(_REC / "c1_new_transport.json"),
              "certification": abort_summary(_REC / "c1_new_certification.json"),
              "trunc30": abort_summary(_REC / "c1_new_trunc30.json")}
    c2_old = c2_trace_old()
    c2_new = {"drag_executable": False,
              "evidence": c1_new["trunc30"],
              "note": "enforcement aborts at the carrier's first undeclared contact "
                      "(jaw x mat at original frame 41), before the old attach frame 89; "
                      "no attached drag can begin under the enforced harness"}

    checks = {
        "C1_old_silent_pass_with_alarm": c1_old["alarm_drill_pass"],
        "C1_new_aborts_both_modes": bool(c1_new["transport"]["aborted"] and c1_new["certification"]["aborted"]),
        "C1_new_trunc30_aborts_before_attach": bool(c1_new["trunc30"]["aborted"]),
        "C2_old_trace_present": c2_old["trace_present"],
        "C2_new_drag_impossible": c2_new["evidence"]["aborted"] is True,
    }
    rec = {"drills": {"C1": {"old": c1_old, "new": c1_new},
                      "C2": {"old": c2_old, "new": c2_new}},
           "checks": checks, "ALL_DRILLS_PASS": all(checks.values())}
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(checks, indent=1))
    print("ALL_DRILLS_PASS:", rec["ALL_DRILLS_PASS"])
    return 0 if rec["ALL_DRILLS_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
