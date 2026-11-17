"""MOCK bus adapter — simulated STS3215 registers behind the same
BusAdapter interface as the real LeRobot follower.

*** THIS IS A MOCK. *** It exists so the bridge, the safety layer,
and doctor checks 4-8 can be exercised end to end before the
founder's arm is attached. Every record it produces is labeled
MOCK, and nothing about it is hardware evidence: hardware proof is
the acceptance test on the founder's SO-101. The mock models the
*interface and register semantics* of the contract (positions in
bridge units, load as duty counts 0-1000 with direction bit
semantics folded into the sign, voltage 0.1 V counts, temperature
degC, moving/status), with a first-order position servo per joint.
It does not model real servo physics — that is what the sim harness
and the real arm are for.

Scenarios:
  normal          all six motors answer; loads stay low.
  gripper_stall   the gripper meets a virtual object partway closed
                  and its load saturates (the S6b stall signature) —
                  drives the saturation-guard evidence.
  mixed_firmware  wrist_roll reports a different firmware version —
                  the doctor audit must FAIL this bus.
  voltage_fault   the Status register carries the voltage bit and
                  Present_Voltage sags — the bridge must refuse to
                  run actions.
"""
import math

from ..calibration import JOINT_NAMES
from .bus import MOTOR_IDS, BusAdapter
from .registers import CONTRACT_CHANNELS

HOME = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
        "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 100.0}

#: The virtual object in the gripper_stall scenario: the gripper
#: cannot close past this (0-100 units) and load saturates there.
STALL_AT = 55.0


def _model_number():
    try:
        import rustypot
        return int(rustypot.find_model("sts3215").model_number)
    except Exception:
        return None


class MockBusAdapter(BusAdapter):
    name = "mock-sts3215-bus (MOCK)"
    is_mock = True

    def __init__(self, scenario: str = "normal"):
        if scenario not in ("normal", "gripper_stall",
                            "mixed_firmware", "voltage_fault"):
            raise ValueError(f"unknown mock scenario {scenario!r}")
        self.scenario = scenario
        self.connected = False
        self.present = dict(HOME)
        self.goal = dict(HOME)
        self.tick_count = 0
        self.commands = []          # every write_goal, for the record

    # ------------------------------------------------------------ bus
    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def _require(self):
        if not self.connected:
            raise RuntimeError("mock bus is not connected")

    def scan(self) -> dict:
        self._require()
        nb = _model_number()
        return {MOTOR_IDS[m]: nb for m in JOINT_NAMES}

    def audit(self) -> dict:
        self._require()
        out = {}
        for m in JOINT_NAMES:
            fw = "3.9"
            if self.scenario == "mixed_firmware" and m == "wrist_roll":
                fw = "3.7"
            out[m] = {"id": MOTOR_IDS[m], "model": "sts3215",
                      "firmware": fw, "simulated": True}
        return out

    def read_calibration(self) -> dict:
        self._require()
        # Simulated stored calibration: full-turn wrist_roll, the
        # rest near-full-range — the shape of a real SO-101 file.
        out = {}
        for m in JOINT_NAMES:
            span = 4095 if m == "wrist_roll" else 3600
            out[m] = {"id": MOTOR_IDS[m], "range_min": 200,
                      "range_max": 200 + span, "homing_offset": 0,
                      "simulated": True}
        return out

    def read_servo_config(self) -> dict:
        self._require()
        # The simulated bus carries LeRobot's configure() values —
        # what a follower reports after a LeRobot connect.
        out = {m: {"Position_P_Coefficient": 16,
                   "Position_I_Coefficient": 0,
                   "Position_D_Coefficient": 32,
                   "simulated": True} for m in JOINT_NAMES}
        out["gripper"].update({"Max_Torque_Limit": 500,
                               "Protection_Current": 250,
                               "Overload_Torque": 25})
        return out

    # -------------------------------------------------------- dynamics
    def _step_dynamics(self):
        """One control tick of the simulated servos: first-order
        approach to goal; the stall scenario pins the gripper."""
        self.tick_count += 1
        for m in JOINT_NAMES:
            goal = self.goal[m]
            if (self.scenario == "gripper_stall" and m == "gripper"
                    and goal < STALL_AT):
                goal = STALL_AT
            cur = self.present[m]
            self.present[m] = cur + (goal - cur) * 0.35

    def _load(self, motor):
        goal = self.goal[motor]
        if motor == "gripper":
            if (self.scenario == "gripper_stall"
                    and goal < STALL_AT - 1.0
                    and abs(self.present[motor] - STALL_AT) < 2.0):
                return -1000.0      # closing into the object, at limit
            err = abs(goal - self.present[motor])
            return -min(140.0, 20.0 + err * 2.0)
        err = abs(goal - self.present[motor])
        return min(220.0, 15.0 + err * 3.0)

    # ------------------------------------------------------------ io
    def read(self, register: str, motors=None) -> dict:
        self._require()
        if register not in CONTRACT_CHANNELS:
            raise ValueError(
                f"{register!r} is not a telemetry-contract channel — "
                f"the bridge reads contract channels only")
        motors = list(motors or JOINT_NAMES)
        out = {}
        for m in motors:
            if register == "Present_Position":
                out[m] = round(self.present[m], 3)
            elif register == "Goal_Position":
                out[m] = round(self.goal[m], 3)
            elif register == "Present_Velocity":
                out[m] = int((self.goal[m] - self.present[m]) * 4)
            elif register == "Present_Load":
                out[m] = self._load(m)
            elif register == "Present_Voltage":
                out[m] = 61 if self.scenario == "voltage_fault" else 74
            elif register == "Present_Temperature":
                out[m] = 31
            elif register == "Present_Current":
                out[m] = int(abs(self._load(m)) * 0.4)
            elif register == "Moving":
                out[m] = int(abs(self.goal[m] - self.present[m]) > 0.05)
            elif register == "Status":
                out[m] = 1 if self.scenario == "voltage_fault" else 0
        return out

    def write_goal(self, positions: dict):
        self._require()
        for m, v in positions.items():
            self.goal[m] = float(v)
        self.commands.append(dict(positions))
        self._step_dynamics()
