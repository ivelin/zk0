"""Bus adapters: the seam between the bridge and a motor bus.

The bridge talks to exactly one interface (`BusAdapter`). Two
implementations exist:

- `RealBusAdapter` — LeRobot 0.7.0's `SOFollower` (feetech bus).
  Telemetry reads go through the bus's sync_read of contract
  registers ONLY (registers.py); identity audit (model numbers,
  firmware) uses LeRobot's own scan/ping machinery — the same calls
  LeRobot's connect path itself makes (`_scan`, `_assert_same_
  firmware`'s register reads) — never a wider telemetry surface.
- `MockBusAdapter` (arm/mock.py) — a simulated bus with the same
  interface, for the MOCK test evidence. It is labeled MOCK
  everywhere it appears; it is not hardware evidence.

Units at this seam are BRIDGE units: body joints in degrees in the
calibrated frame (LeRobot's use_degrees=True follower frame),
gripper in LeRobot's 0-100 range. Both adapters speak them; the
conversion to/from raw counts is the adapter's (on the real bus it
is LeRobot's own normalization against the arm's calibration).
"""
import glob
import os
from pathlib import Path

from ..calibration import JOINT_NAMES
from .registers import FEEDBACK_CHANNELS

#: The SO-101 follower's motors, in LeRobot's table order
#: (calibration.JOINT_NAMES): bus ids 1-6, all Feetech STS3215.
MOTOR_IDS = {name: i + 1 for i, name in enumerate(JOINT_NAMES)}
EXPECTED_MODEL = "sts3215"


def find_arm_port(pinned: str = None, by_id_dir="/dev/serial/by-id",
                  acm_pattern="/dev/ttyACM*"):
    """Resolve the arm's serial port: the pinned /dev/serial/by-id
    path first (stable across reboots), else the single /dev/ttyACM*
    device. Returns (path, how) or (None, why) — a why starting
    with "ambiguous" means devices exist but cannot be told apart
    (a failure, not an absence)."""
    if pinned:
        if os.path.exists(pinned):
            return pinned, "pinned /dev/serial/by-id path"
        return None, f"pinned port {pinned} does not exist"
    by_id = sorted(glob.glob(str(Path(by_id_dir) / "*")))
    feetech = [p for p in by_id if "Feetech" in p or "1A86" in p
               or "WCH" in p]
    if len(feetech) == 1:
        return feetech[0], "the single Feetech-class by-id device"
    acm = sorted(glob.glob(acm_pattern))
    if len(acm) == 1:
        return acm[0], "the single /dev/ttyACM device (unpinned)"
    if not acm and not by_id:
        return None, "no /dev/ttyACM* device and no /dev/serial/by-id"
    return None, (f"ambiguous: {len(acm)} ttyACM devices, "
                  f"{len(by_id)} by-id entries — pin the port in "
                  f"config (arm.port)")


class BusAdapter:
    """The interface the bridge consumes. See module docstring."""

    name = "abstract"
    is_mock = False

    def connect(self):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def scan(self) -> dict:
        """{bus_id: model_number} for every id answering on the bus
        (LeRobot's scan semantics)."""
        raise NotImplementedError

    def audit(self) -> dict:
        """Per-motor identity: {motor: {id, model, firmware}} plus
        bus-level flags the doctor check judges."""
        raise NotImplementedError

    def read(self, register: str, motors=None) -> dict:
        """{motor: value} for one contract register."""
        raise NotImplementedError

    def read_state(self) -> dict:
        """One control tick of feedback: {register: {motor: value}}
        for every contract feedback channel."""
        return {reg: self.read(reg) for reg in FEEDBACK_CHANNELS}

    def read_positions(self) -> dict:
        """{motor: position} in bridge units."""
        return self.read("Present_Position")

    def write_goal(self, positions: dict):
        """Command goal positions in bridge units (already through
        the safety layer — adapters never second-guess it)."""
        raise NotImplementedError

    def read_calibration(self) -> dict:
        """The bus's stored calibration, plain dict per motor:
        {id, range_min, range_max, homing_offset}."""
        raise NotImplementedError

    def read_servo_config(self) -> dict:
        """The protective servo configuration (doctor check 8):
        {motor: {Position_P_Coefficient, Position_I_Coefficient,
        Position_D_Coefficient}} plus the gripper's torque
        protection registers. These are CONFIGURATION registers
        LeRobot's configure() writes at connect — read here for
        audit only; they are not telemetry and never feed control."""
        raise NotImplementedError


class RealBusAdapter(BusAdapter):
    name = "lerobot-so-follower"
    is_mock = False

    def __init__(self, port: str, calibration_dir=None):
        self.port = port
        self.calibration_dir = calibration_dir
        self._follower = None

    def _f(self):
        if self._follower is None:
            from lerobot.robots.so_follower import (
                SOFollower, SOFollowerConfig)
            cfg = SOFollowerConfig(
                port=self.port, id="zk0_arm",
                calibration_dir=(str(self.calibration_dir)
                                 if self.calibration_dir else None),
                use_degrees=True)
            self._follower = SOFollower(cfg)
        return self._follower

    def connect(self):
        # calibrate=False: connecting never drives the arm through
        # a calibration routine; calibration is its own explicit,
        # confirmed act (spec stage 2 / slice 2).
        self._f().connect(calibrate=False)

    def disconnect(self):
        if self._follower is not None and self._follower.is_connected:
            self._follower.disconnect()

    def scan(self) -> dict:
        # LeRobot's own scan path (the calls its setup flow makes):
        # the per-motor servo definition + broadcast scan. Identity
        # audit only — telemetry reads stay on contract registers.
        bus = self._f().bus
        definition = bus._definition("shoulder_pan")
        return {int(k): int(v) for k, v in
                bus._scan(definition).items()}

    def audit(self) -> dict:
        bus = self._f().bus
        motors = list(JOINT_NAMES)
        majors = bus.sync_read("Firmware_Major_Version", motors)
        minors = bus.sync_read("Firmware_Minor_Version", motors)
        out = {}
        for m in motors:
            out[m] = {"id": MOTOR_IDS[m], "model": EXPECTED_MODEL,
                      "firmware": f"{majors[m]}.{minors[m]}"}
        return out

    def read(self, register: str, motors=None) -> dict:
        return self._f().bus.sync_read(register,
                                       list(motors or JOINT_NAMES))

    def write_goal(self, positions: dict):
        self._f().bus.sync_write(
            "Goal_Position", {m: float(v) for m, v in positions.items()})

    def read_calibration(self) -> dict:
        cal = self._f().bus.read_calibration()
        return {m: {"id": int(c.id), "range_min": int(c.range_min),
                    "range_max": int(c.range_max),
                    "homing_offset": int(c.homing_offset)}
                for m, c in cal.items()}

    def read_servo_config(self) -> dict:
        bus = self._f().bus
        out = {}
        for reg in ("Position_P_Coefficient", "Position_I_Coefficient",
                    "Position_D_Coefficient"):
            vals = bus.sync_read(reg, list(JOINT_NAMES))
            for m, v in vals.items():
                out.setdefault(m, {})[reg] = int(v)
        for reg in ("Max_Torque_Limit", "Protection_Current",
                    "Overload_Torque"):
            out.setdefault("gripper", {})[reg] = int(
                bus.read(reg, "gripper"))
        return out
