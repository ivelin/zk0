"""The arm bridge: the node's one path to the SO-101 (spec stage 6
machinery; stage 1-2 checks 4-8 probe it).

Responsibilities, in the order a command flows:
  connect -> scan + firmware audit (a mixed-firmware bus is refused:
  LeRobot's own connect asserts the same; the bridge refuses to run
  actions on a bus it cannot positively identify)
  read    -> contract channels ONLY (registers.FEEDBACK_CHANNELS),
             decoded with the contract's Status bit names; a servo
             reporting a Status fault blocks actions until it clears
  act     -> every action passes the hard safety layer
             (safety_layer.SafetyLayer) BEFORE the adapter writes
             it; a watchdog refusal makes the bridge first command
             a hold at the last read position
  record  -> the safety layer's JSONL log + the bridge's cycle
             record are the evidence trail

The bridge is adapter-agnostic: the real LeRobot follower and the
MOCK adapter exercise identical code paths. Which one a record came
from is carried in every record (`adapter`, `mock` fields) — MOCK
evidence is never hardware evidence.
"""
import math
import time
from pathlib import Path

from ..calibration import JOINT_NAMES, PROBE_POSES_DEG
from ..safety import DEFAULT_LIMITS
from .registers import decode_status
from .safety_layer import SafetyLayer, SafetyRefusal

GRIPPER_MJCF_RAD = (-0.174533, 1.745330)   # staged model gripper range


class BridgeError(Exception):
    pass


def model_ranges_and_fk(work_dir):
    """Joint ranges in bridge units + an FK closure, both derived
    from the staged arm model (the same file the sim harness runs):
    the safety envelope and the sim share one source of geometry."""
    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(
        str(Path(work_dir) / "so101_realistic.xml"))
    ranges, qadr = {}, {}
    for j in JOINT_NAMES:
        jid = m.joint(j).id
        qadr[j] = int(m.jnt_qposadr[jid])
        lo, hi = (float(v) for v in m.jnt_range[jid])
        if j == "gripper":
            ranges[j] = (0.0, 100.0)
        else:
            ranges[j] = (math.degrees(lo), math.degrees(hi))
    site_id = m.site("gripperframe").id
    d = mujoco.MjData(m)
    glo, ghi = GRIPPER_MJCF_RAD

    def fk_tcp(positions):
        for j in JOINT_NAMES:
            v = float(positions[j])
            if j == "gripper":
                q = glo + (v / 100.0) * (ghi - glo)
            else:
                q = math.radians(v)
            d.qpos[qadr[j]] = q
        mujoco.mj_forward(m, d)
        return tuple(float(x) for x in d.site_xpos[site_id])

    return ranges, fk_tcp


class ArmBridge:
    def __init__(self, adapter, work_dir=None, profile="normal",
                 limits=None, clock=time.monotonic, safety_log=None):
        self.adapter = adapter
        self.profile = profile
        self.clock = clock
        if work_dir is not None:
            ranges, fk = model_ranges_and_fk(work_dir)
        else:
            ranges, fk = ({j: (-180.0, 180.0) for j in JOINT_NAMES},
                          None)
            ranges["gripper"] = (0.0, 100.0)
        self.safety = SafetyLayer(ranges, fk_tcp=fk, limits=limits,
                                  profile=profile, clock=clock,
                                  log_path=safety_log)
        self.audit_info = None
        self.last_positions = None
        self.cycles = []

    # ---------------------------------------------------------- connect
    def connect(self):
        self.adapter.connect()
        scan = self.adapter.scan()
        expected = {i + 1 for i in range(len(JOINT_NAMES))}
        missing = expected - set(scan)
        if missing:
            raise BridgeError(
                f"bus scan found ids {sorted(scan)}; expected the "
                f"SO-101's six servos at ids 1-6 — missing "
                f"{sorted(missing)} (check power and the daisy chain)")
        self.audit_info = self.adapter.audit()
        fws = {v["firmware"] for v in self.audit_info.values()}
        if len(fws) > 1:
            raise BridgeError(
                f"mixed servo firmware on the bus: "
                f"{ {m: v['firmware'] for m, v in self.audit_info.items()} } "
                f"— LeRobot's connect asserts uniform firmware; the "
                f"bridge refuses to run actions on a mixed bus")
        return {"scan": {str(k): v for k, v in scan.items()},
                "audit": self.audit_info}

    def disconnect(self):
        self.adapter.disconnect()

    def calibration(self):
        return self.adapter.read_calibration()

    # ------------------------------------------------------------- read
    def read_contract_state(self):
        raw = self.adapter.read_state()
        positions = {m: float(v) for m, v in
                     raw["Present_Position"].items()}
        self.last_positions = positions
        faults = {}
        for m, s in raw["Status"].items():
            bits = decode_status(int(s))
            if bits:
                faults[m] = bits
        return {
            "positions": positions,
            "gripper_load_permille": float(
                raw["Present_Load"]["gripper"]),
            "voltage_v": {m: round(int(v) * 0.1, 1) for m, v in
                          raw["Present_Voltage"].items()},
            "temperature_c": {m: int(v) for m, v in
                              raw["Present_Temperature"].items()},
            "moving": {m: bool(int(v)) for m, v in
                       raw["Moving"].items()},
            "status_faults": faults,
            "channels": sorted(raw.keys()),
            "raw": raw,
        }

    # -------------------------------------------------------------- act
    def send_action(self, action: dict) -> dict:
        state = self.read_contract_state()
        if state["status_faults"]:
            self.safety._event(
                "refusal", check="servo_status",
                reason=f"servo status faults: {state['status_faults']}")
            raise SafetyRefusal(
                "servo_status",
                f"servo status faults {state['status_faults']} — "
                f"actions blocked until the fault clears")
        try:
            applied = self.safety.check_action(state["positions"],
                                               action)
        except SafetyRefusal as r:
            if r.check == "watchdog" and self.safety.held:
                # Hold first: command the last read position (in
                # range by construction), then surface the refusal.
                self.adapter.write_goal(dict(state["positions"]))
                self.safety.note_hold_issued()
            raise
        self.adapter.write_goal(applied)
        post = self.adapter.read("Present_Load", ["gripper"])
        self.safety.feed_gripper_load(float(post["gripper"]))
        self.cycles.append({"requested": dict(action),
                            "applied": dict(applied)})
        return applied

    def stop(self, reason="operator stop"):
        self.safety.stop(reason)
        if self.last_positions:
            self.adapter.write_goal(dict(self.last_positions))

    def resume(self):
        self.safety.resume()


# ------------------------------------------------------------------
# The MOCK demonstration (evidence for the bridge + safety layer;
# hardware proof is the acceptance test on the founder's arm)
# ------------------------------------------------------------------
class _FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _demo_bridge(home, work_dir, scenario):
    from .mock import MockBusAdapter
    clock = _FakeClock()
    log = Path(home) / "records" / "arm" / f"safety-log-{scenario}.jsonl"
    bridge = ArmBridge(MockBusAdapter(scenario), work_dir=work_dir,
                       clock=clock, safety_log=log)
    return bridge, clock


def run_mock_demo(home, work_dir) -> dict:
    """Scripted MOCK run: connect + audit, the contract read set, a
    legal probe-pose walk, a clamped out-of-range goal (counted), a
    refused wild goal, a watchdog hold + resume, and — on a second
    mock bus — the gripper-stall saturation abort and the
    voltage-fault action block. Returns the demo record (the caller
    stores it, signed, scope 'mock')."""
    steps = []

    def step(name, fn):
        try:
            out = fn()
            steps.append({"step": name, "outcome": "ok", **(out or {})})
            return out
        except (SafetyRefusal, Exception) as e:  # noqa: BLE001
            steps.append({"step": name, "outcome": "raised",
                          "error_type": type(e).__name__,
                          "error": str(e),
                          "check": getattr(e, "check", None)})
            return None

    bridge, clock = _demo_bridge(home, work_dir, "normal")
    conn = step("connect + scan + firmware audit", bridge.connect)
    state = step("contract read set", bridge.read_contract_state)
    if state:
        steps[-1]["channels"] = state["channels"]
        steps[-1]["positions"] = state["positions"]

    def walk():
        # Legs longer than the per-command cap are subdivided, the
        # way a trajectory generator feeds setpoints — the cap is a
        # per-command rule, not a per-pose one. Each sub-command is
        # computed from a fresh contract read, so it is within the
        # cap of the arm's actual position when it is sent.
        commands = 0
        for pose_list in PROBE_POSES_DEG:
            pose = dict(zip(JOINT_NAMES, pose_list))
            for _ in range(16):
                cur = bridge.read_contract_state()["positions"]
                delta = {j: pose[j] - cur[j] for j in JOINT_NAMES}
                if max(abs(v) for v in delta.values()) <= 40.0:
                    bridge.send_action(dict(pose))
                    commands += 1
                    break
                bridge.send_action(
                    {j: cur[j] + max(-40.0, min(40.0, delta[j]))
                     for j in JOINT_NAMES})
                commands += 1
            else:
                raise RuntimeError(f"probe pose {pose} not reached")
        return {"poses_applied": len(PROBE_POSES_DEG),
                "commands_sent": commands}
    step("legal action sequence (calibration probe poses)", walk)

    def clamp_probe():
        # The S6b signature itself: a gripper goal past the end of
        # its range (theirs was -0.65 rad past the floor; here, past
        # the 100 top of the LeRobot 0-100 gripper range).
        cur = dict(bridge.last_positions)
        target = dict(cur)
        target["gripper"] = 130.0
        before = bridge.safety.clamp_count
        applied = bridge.send_action(target)
        return {"requested_gripper": 130.0,
                "applied_gripper": applied["gripper"],
                "clamps_counted": bridge.safety.clamp_count - before}
    step("out-of-range goal clamped + counted (S6b rule)", clamp_probe)

    def wild_goal():
        cur = dict(bridge.last_positions)
        target = dict(cur)
        target["shoulder_pan"] = -170.0    # out of range AND far
        return bridge.send_action(target)
    step("wild out-of-range goal refused by the safety layer",
         wild_goal)

    def watchdog():
        clock.advance(5.0)                  # command stream stalls
        cur = dict(bridge.last_positions)
        try:
            bridge.send_action(dict(cur))
            return {"watchdog_fired": False}
        except SafetyRefusal as r:
            held = bridge.safety.held
            bridge.resume()
            bridge.send_action(dict(cur))
            return {"watchdog_fired": True, "check": r.check,
                    "hold_latched": held, "resumed": True}
    step("command watchdog: stall -> hold -> resume", watchdog)
    steps.append({"step": "safety events (normal bus)",
                  "outcome": "ok",
                  "events": bridge.safety.events,
                  "clamp_count_total": bridge.safety.clamp_count})
    bridge.disconnect()

    stall, _ = _demo_bridge(home, work_dir, "gripper_stall")
    step("stall bus: connect", stall.connect)

    def stall_run():
        # Close in cap-sized steps onto the virtual object; once the
        # gripper pins at the stall, every further close command
        # feeds a saturated load sample until the guard aborts.
        stall.send_action({"gripper": 60.0})
        for _ in range(40):
            cur = stall.read_contract_state()["positions"]["gripper"]
            stall.send_action(
                {"gripper": max(20.0, cur - 40.0)})
        return {"aborted": False}
    step("gripper stall: sustained saturation aborts the run",
         stall_run)
    steps.append({"step": "safety events (stall bus)", "outcome": "ok",
                  "events": stall.safety.events,
                  "stopped_latched": stall.safety.stopped})
    stall.disconnect()

    vf, _ = _demo_bridge(home, work_dir, "voltage_fault")
    step("voltage-fault bus: connect", vf.connect)
    step("voltage-fault bus: action blocked on servo status",
         lambda: vf.send_action({"shoulder_pan": 5.0}))
    vf.disconnect()

    mixed, _ = _demo_bridge(home, work_dir, "mixed_firmware")
    step("mixed-firmware bus: connect refused", mixed.connect)

    return {
        "kind": "arm-mock-demo",
        "scope": "mock",
        "mock": True,
        "adapter": "mock-sts3215-bus (MOCK)",
        "hardware_evidence": False,
        "note": "MOCK demonstration against a simulated bus with the "
                "LeRobot follower interface. Hardware proof is the "
                "acceptance test on the founder's arm (week plan, "
                "Sunday) — nothing here is a claim about real "
                "hardware.",
        "connect": conn,
        "steps": steps,
        "limits": {"workspace_box_m": DEFAULT_LIMITS["workspace_box_m"],
                   "damage_bound_N_provisional":
                       DEFAULT_LIMITS["damage_bound_N"]},
    }
