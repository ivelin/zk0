"""The hard safety layer, arm-side (spec section 4.1).

Slice 1 shipped this layer's configuration and command clamp
(`safety.py`); this module is the enforcement the arm bridge runs
every command through BEFORE anything reaches the bus. It sits below
the learning: no weight file, config flag, or operator setting can
disable it or loosen it past the certified envelope — an operator
may only tighten it (spec section 4.1), which is why the limits are
constructed from `safety.DEFAULT_LIMITS` + the staged arm model's
own joint ranges, never from a policy's output.

Enforcement, in order, for every commanded action:
  0. stop latch — after `stop()` (or a saturation abort) every
     command is refused until an explicit `resume()`.
  1. watchdog — in an active session, a command arriving after a
     gap longer than the watchdog window is refused and the layer
     latches `held`; the bridge answers a held latch by commanding
     the last read position (a hold, by construction in-range) and
     the operator/agent must `resume()`. Loss of command freshness
     must end in hold-then-stop, never in carrying on (spec 4.1).
  2. joint-range clamp — every joint's goal is clamped into the arm
     model's own range and the clamp is COUNTED (the S6b lesson,
     SSOT decision 27: an out-of-range close command held at stall
     produced ~47x object-weight clamp force; out-of-range commands
     are clamped and counted, and sustained saturation aborts).
  3. wrist_roll seam guard (backlog C4) — the wrist_roll motor is a
     full-turn motor in a wrapped frame: a single command that
     would jump it more than half a turn, or drive it from one
     range end to the other, is refused as a seam crossing. Checked
     before the speed cap so a wrap is named as what it is.
  4. speed cap — a goal farther than the per-command delta cap from
     the present position is REFUSED, not clamped: silently
     rewriting a far goal into a near one would hide a policy (or
     addressing) error. Caps tighten in the canary profile (4.2).
  5. workspace box — the commanded pose's tool-centre point,
     computed by forward kinematics on the staged arm model from
     contract-channel positions only, must stay inside the declared
     box (spec 4.1: commands that would leave it are refused).
  6. gripper saturation guard — Present_Load samples (the contract's
     duty channel, section 4 of the contract: a contact detector,
     not a force regulator) are windowed during action execution;
     if the saturated fraction of the window exceeds the certified
     abort fraction, the run ABORTS (latched stop + SafetyAbort).

Every clamp, refusal, hold, and abort is appended to the layer's
event list and, when a log path is configured, to a JSONL safety
log in the node's record store. The log is the evidence the demo
and the acceptance test are judged from.
"""
import json
import time
from pathlib import Path

from ..calibration import JOINT_NAMES
from ..safety import DEFAULT_LIMITS

#: STS3215 Present_Load magnitude at/above which the servo is at its
#: torque limit: 990 of the 0-1000 duty counts (99%). The sim gate's
#: saturation definition is the same shape (|force| >= 0.99 x limit).
SATURATED_PERMILLE = 990
SATURATION_WINDOW = 20

PROFILES = {
    # Per-command joint-delta caps (degrees; gripper in its 0-100
    # units) and the watchdog window. Canary (spec 4.2) tightens
    # both and shrinks the workspace box.
    "normal": {"max_delta": 45.0, "watchdog_s": 2.0,
               "workspace_margin_m": 0.0},
    "canary": {"max_delta": 10.0, "watchdog_s": 1.0,
               "workspace_margin_m": 0.03},
}


class SafetyRefusal(Exception):
    """A command was refused by the safety layer (named check)."""

    def __init__(self, check, reason):
        self.check = check
        self.reason = reason
        super().__init__(f"{check}: {reason}")


class SafetyAbort(Exception):
    """The run is aborted by the safety layer (latched stop)."""


class SafetyLayer:
    def __init__(self, joint_ranges: dict, fk_tcp=None, limits=None,
                 profile: str = "normal", clock=time.monotonic,
                 log_path=None):
        """joint_ranges: {motor: (lo, hi)} in bridge units (body
        joints in degrees, gripper in LeRobot 0-100 units).
        fk_tcp: callable(positions_bridge_units) -> (x, y, z) metres,
        or None (workspace check then cannot run and says so)."""
        if profile not in PROFILES:
            raise ValueError(f"unknown safety profile {profile!r}")
        self.ranges = {m: (float(lo), float(hi))
                       for m, (lo, hi) in joint_ranges.items()}
        self.fk_tcp = fk_tcp
        self.limits = limits or DEFAULT_LIMITS
        self.profile = profile
        self.p = PROFILES[profile]
        self.clock = clock
        self.log_path = Path(log_path) if log_path else None
        self.events = []
        self.clamp_count = 0
        self.stopped = False
        self.held = False
        self.session_active = False
        self.last_command_t = None
        self._load_window = []

    # ------------------------------------------------------------ log
    def _event(self, kind, **kw):
        ev = {"t_wall": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime()),
              "kind": kind, **kw}
        self.events.append(ev)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(ev) + "\n")
        return ev

    # -------------------------------------------------------- control
    def stop(self, reason="operator stop"):
        self.stopped = True
        self._event("stop", reason=reason)

    def resume(self):
        self.stopped = False
        self.held = False
        self.session_active = False
        self.last_command_t = None
        self._event("resume")

    def note_hold_issued(self):
        self._event("watchdog_hold_issued")

    # ------------------------------------------------------ saturation
    def feed_gripper_load(self, load_permille):
        """Window a Present_Load sample (duty counts, signed). Called
        by the bridge on every action cycle while the gripper is
        commanded closed/holding. Raises SafetyAbort (and latches
        the stop) when the saturated fraction of the window exceeds
        the certified abort fraction."""
        if load_permille is None:
            return
        self._load_window.append(abs(float(load_permille)))
        self._load_window = self._load_window[-SATURATION_WINDOW:]
        if len(self._load_window) < SATURATION_WINDOW:
            return
        frac = sum(1 for v in self._load_window
                   if v >= SATURATED_PERMILLE) / len(self._load_window)
        if frac > self.limits["gripper_saturation_abort_frac"]:
            self.stopped = True
            self._event("abort", check="gripper_saturation",
                        reason=f"saturated fraction {frac:.2f} of the "
                               f"last {SATURATION_WINDOW} load samples "
                               f"exceeds the abort fraction "
                               f"{self.limits['gripper_saturation_abort_frac']}",
                        saturated_frac=round(frac, 3))
            raise SafetyAbort(
                f"gripper saturation: {frac:.0%} of load samples at "
                f"the torque limit (abort fraction "
                f"{self.limits['gripper_saturation_abort_frac']:.0%}) — "
                f"the S6b stall-clamp signature; run stopped")

    # ------------------------------------------------------------ main
    def check_action(self, present: dict, action: dict) -> dict:
        """Run one commanded action through the layer.

        present: {motor: position} in bridge units (the last
        contract read). action: {motor: goal} (partial allowed).
        Returns the applied action (after clamping). Raises
        SafetyRefusal (named check) or SafetyAbort.
        """
        now = self.clock()
        if self.stopped:
            self._event("refusal", check="stopped",
                        reason="safety layer is stopped; resume() "
                               "is an explicit act")
            raise SafetyRefusal("stopped", "safety layer is stopped")
        if self.held:
            self._event("refusal", check="watchdog",
                        reason="watchdog hold is latched; resume() "
                               "before further commands")
            raise SafetyRefusal("watchdog", "watchdog hold latched")
        if (self.session_active and self.last_command_t is not None
                and now - self.last_command_t > self.p["watchdog_s"]):
            self.held = True
            self._event("watchdog_hold",
                        gap_s=round(now - self.last_command_t, 3),
                        reason="command gap exceeded the watchdog "
                               "window; hold latched")
            raise SafetyRefusal(
                "watchdog",
                f"command gap {now - self.last_command_t:.2f}s > "
                f"watchdog window {self.p['watchdog_s']}s — hold "
                f"latched; the bridge holds the last read position "
                f"and resume() is required")

        applied = {}
        for motor, goal in action.items():
            if motor not in self.ranges:
                self._event("refusal", check="unknown_joint",
                            reason=f"no range for joint {motor!r}")
                raise SafetyRefusal("unknown_joint",
                                    f"no range for joint {motor!r}")
            lo, hi = self.ranges[motor]
            g = float(goal)
            clamped = min(max(g, lo), hi)
            if clamped != g:
                self.clamp_count += 1
                self._event("clamp", joint=motor, requested=g,
                            applied=clamped,
                            reason="goal outside the joint's own "
                                   "range (counted, S6b lesson)")
            applied[motor] = clamped

        if "wrist_roll" in applied and "wrist_roll" in present:
            lo, hi = self.ranges["wrist_roll"]
            cur, goal = float(present["wrist_roll"]), applied["wrist_roll"]
            if abs(goal - cur) > 180.0:
                self._event("refusal", check="seam_guard",
                            reason="single command jumps wrist_roll "
                                   "more than half a turn (wrap)")
                raise SafetyRefusal(
                    "seam_guard",
                    "wrist_roll: single command jumps more than "
                    "half a turn — the wrapped-frame seam (backlog "
                    "C4); approach the seam in steps")
            near = 2.0
            if ((cur <= lo + near and goal >= hi - near)
                    or (cur >= hi - near and goal <= lo + near)):
                self._event("refusal", check="seam_guard",
                            reason="command drives wrist_roll from "
                                   "one range end to the other")
                raise SafetyRefusal(
                    "seam_guard",
                    "wrist_roll: command crosses from one range end "
                    "to the other (backlog C4 seam)")

        for motor, g in applied.items():
            if motor not in present:
                continue
            delta = abs(g - float(present[motor]))
            if delta > self.p["max_delta"]:
                self._event("refusal", check="speed_cap",
                            joint=motor, delta=round(delta, 2),
                            reason=f"goal {delta:.1f} units from "
                                   f"present exceeds the "
                                   f"{self.profile} per-command cap "
                                   f"{self.p['max_delta']}")
                raise SafetyRefusal(
                    "speed_cap",
                    f"{motor}: goal is {delta:.1f} from present; "
                    f"per-command cap is {self.p['max_delta']} "
                    f"({self.profile} profile) — refused, not clamped")

        if self.fk_tcp is not None:
            pose = {m: float(present.get(m, 0.0)) for m in JOINT_NAMES}
            pose.update(applied)
            tcp = self.fk_tcp(pose)
            box = self.limits["workspace_box_m"]
            mgn = self.p["workspace_margin_m"]
            inside = all(box[ax][0] + mgn <= tcp[i] <= box[ax][1] - mgn
                         for i, ax in enumerate(("x", "y", "z")))
            if not inside:
                self._event("refusal", check="workspace",
                            tcp=[round(float(v), 4) for v in tcp],
                            reason="commanded pose leaves the "
                                   "declared workspace box")
                raise SafetyRefusal(
                    "workspace",
                    f"commanded TCP {[round(float(v), 3) for v in tcp]} "
                    f"leaves the declared workspace box "
                    f"(profile {self.profile})")

        self.session_active = True
        self.last_command_t = now
        return applied
