"""`zk0 doctor` — the 12 self-checks of spec §1.4.

Doctor runs at install, on demand, and before any contribution. Each
check returns a status of pass / fail / skip / not_present with the
reason in plain language. Hardware checks on a machine with no arm
degrade honestly to not_present — never to a fake pass. The full
output is written as one diagnostic bundle (the support-escalation
format, spec §1.4).

Status semantics: `fail` means something the manifest or the mode
requires is wrong; `skip` means the check does not apply yet and its
detail says exactly what is missing and which build slice or stage
supplies it; `not_present` is the hardware-absent state.
"""
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import load_config, zk0_home
from .records import manifest_sha256, runtime_manifest_path, utcnow_iso
from .safety import DEFAULT_LIMITS, clamp_command, workspace_contains
from .sim import assets

PASS, FAIL, SKIP, NOT_PRESENT = "pass", "fail", "skip", "not_present"


def _check(cid, name, status, detail, fix=None):
    out = {"id": cid, "name": name, "status": status, "detail": detail}
    if fix:
        out["fix"] = fix
    return out


def _manifest():
    return json.loads(runtime_manifest_path().read_text())


def _pin(manifest, name):
    for p in manifest["pins"]:
        if p["name"] == name:
            return p
    return None


# ------------------------------------------------------------------ 1
def check_runtime_integrity(cfg):
    m = _manifest()
    problems = []
    if m["version"] != __version__:
        problems.append(f"manifest version {m['version']} != package {__version__}")
    harness_dir = Path(__file__).parent / "sim" / "harness"
    for fname, want in m["components"]["sim_harness"]["files_md5"].items():
        f = harness_dir / fname
        if not f.exists():
            problems.append(f"vendored harness file missing: {fname}")
        elif hashlib.md5(f.read_bytes()).hexdigest() != want:
            problems.append(f"vendored harness file hash mismatch: {fname}")
    # Build slice 3: the vendored BAM servo profile (punkfab servo
    # law + the fitted m6 parameters) and the installed bam
    # package's own parameter file must both hash to the certified
    # profile's bytes — the runtime servo model IS the certified
    # one, or doctor says it is not.
    bam_dir = Path(__file__).parent / "sim" / "bam"
    bam_files = m["components"].get("bam_profile", {}).get("files_md5", {})
    for fname, want in bam_files.items():
        f = bam_dir / fname
        if not f.exists():
            problems.append(f"BAM profile file missing: {fname}")
        elif hashlib.md5(f.read_bytes()).hexdigest() != want:
            problems.append(f"BAM profile file hash mismatch: {fname}")
    if bam_files:
        try:
            import bam as _bam_pkg
            live = (Path(_bam_pkg.__file__).parent / "params" /
                    "feetech_sts3215_7_4V" / "m6.json")
            want = bam_files.get("params/feetech_sts3215_7_4V/m6.json")
            if hashlib.md5(live.read_bytes()).hexdigest() != want:
                problems.append("installed bam package's m6.json "
                                "differs from the certified profile "
                                "parameters")
        except Exception as e:  # noqa: BLE001
            problems.append(f"bam package parameter check failed: {e}")
    info = assets.arm_model_info("realistic")
    asset = m["assets"][0]
    if not info["present"]:
        problems.append("sim assets not staged (model-realistic missing)")
    elif info.get("canonical_md5") != asset["md5"]:
        problems.append(f"staged arm model md5 {info.get('canonical_md5')} "
                        f"!= manifest pin {asset['md5']}")
    if problems:
        return _check(1, "Runtime integrity", FAIL, "; ".join(problems),
                      fix="re-stage assets via dev/bootstrap.sh; do not edit "
                          "vendored harness files (see PROVENANCE.md)")
    return _check(1, "Runtime integrity", PASS,
                  f"package {__version__} == manifest {m['version']}; vendored "
                  f"harness hashes match; staged so101-realistic md5 matches "
                  f"the pinned {asset['md5']}; BAM servo profile hashes "
                  f"match the certified profile (slice 3)")


# ------------------------------------------------------------------ 2
def check_dependency_consistency(cfg):
    m = _manifest()
    subs, hard_fail = [], False
    v = sys.version_info
    if (v.major, v.minor) == (3, 12):
        subs.append(f"python {v.major}.{v.minor}.{v.micro} == pin 3.12.x")
    else:
        hard_fail = True
        subs.append(f"python {v.major}.{v.minor}.{v.micro} != pin 3.12.x")
    venv = os.environ.get("ZK0_RUNTIME_VENV")
    if venv and sys.executable.startswith(venv):
        subs.append("interpreter is the runtime venv (no system-Python leakage)")
    else:
        hard_fail = True
        subs.append("interpreter is NOT the runtime venv — run via bin/zk0 "
                    "(system-Python leakage, spec §1.4 #2)")
    for mod, pin_name in (("mujoco", "mujoco"), ("numpy", "numpy")):
        pin = _pin(m, pin_name)
        try:
            got = __import__(mod).__version__
            if got == pin["version"]:
                subs.append(f"{mod} {got} == pin {pin['version']}")
            else:
                hard_fail = True
                subs.append(f"{mod} {got} != pin {pin['version']}")
        except ImportError:
            hard_fail = True
            subs.append(f"{mod} NOT INSTALLED (pin {pin['version']})")
    for mod, pin_name in (("lerobot", "lerobot"), ("av", "pyav"),
                          ("datasets", "datasets"), ("flwr", "flwr"),
                          ("newton", "newton"), ("torch", "torch")):
        pin = _pin(m, pin_name)
        try:
            got = __import__(mod).__version__
            if got == pin["version"]:
                subs.append(f"{mod} {got} == pin {pin['version']}")
            else:
                hard_fail = True
                subs.append(f"{mod} {got} != pin {pin['version']}")
        except ImportError:
            hard_fail = True
            subs.append(f"{mod} NOT INSTALLED (pin {pin['version']}) — "
                        f"the train pack is required for Stages 3/5/8; "
                        f"run dev/bootstrap.sh")
    # Build slice 3 packs: the serial bus (LeRobot's [serial-motors]
    # extra: rustypot + pyserial) for the arm bridge, and the BAM
    # fitted-servo library (better-actuator-models, pinned commit).
    import importlib.metadata as _md
    for mod, dist, pin_name in (("rustypot", "rustypot", "rustypot"),
                                ("serial", "pyserial", "pyserial"),
                                ("bam", "better-actuator-models", "bam"),
                                ("colorama", "colorama", "colorama")):
        pin = _pin(m, pin_name)
        if pin is None:
            continue
        try:
            __import__(mod)
            got = _md.version(dist)
            if got == pin["version"]:
                subs.append(f"{dist} {got} == pin {pin['version']}")
            else:
                hard_fail = True
                subs.append(f"{dist} {got} != pin {pin['version']}")
        except ImportError:
            hard_fail = True
            subs.append(f"{dist} NOT INSTALLED (pin {pin['version']}) — "
                        f"required by the arm bridge / BAM servo "
                        f"profile (build slice 3); run dev/bootstrap.sh")
    subs.append("known resolver override: flwr 1.23.0 installed "
                "--no-deps (see the flwr pin's note in the manifest) "
                "— pip check reports THREE known conflicts, all "
                "flwr's: click<8.2 (carried: 8.5.1; huggingface-hub "
                "requires >=8.4.2 — jointly unsatisfiable), "
                "iterators<0.0.3,>=0.0.2 (carried: 0.2.0), and "
                "pathspec<0.13.0,>=0.12.1 (carried: 1.1.1). The "
                "runtime does not claim a clean resolve, and flwr "
                "is exercised by the local-round proof instead")
    return _check(2, "Dependency consistency", FAIL if hard_fail else PASS,
                  subs)


# ------------------------------------------------------------------ 3
def check_compute_profile(cfg):
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "model\t", "hardware")):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if cpu_model == "unknown" and shutil.which("lscpu"):
        try:
            out = subprocess.run(["lscpu"], capture_output=True, text=True,
                                 timeout=5)
            for line in out.stdout.splitlines():
                if line.startswith("Model name:"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        except (subprocess.SubprocessError, OSError):
            pass
    gpu = "not detected"
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                gpu = out.stdout.strip().splitlines()[0]
        except (subprocess.SubprocessError, OSError):
            pass
    compute = {"platform": platform.platform(), "machine": platform.machine(),
               "cpu_count": os.cpu_count(), "cpu_model": cpu_model, "gpu": gpu}
    node_manifest = {
        "node_id": cfg["node_id"], "embodiment": "so101",
        "identity_kind": cfg["identity_kind"], "mode": cfg["mode"],
        "compute": compute,
        "runtime": {"package": "zk0 Node", "package_version": __version__,
                    "runtime_manifest_sha256": manifest_sha256(runtime_manifest_path())},
        "updated_utc": utcnow_iso(),
    }
    home = zk0_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "node-manifest.json").write_text(json.dumps(node_manifest, indent=1))
    return _check(3, "Compute profile", PASS,
                  f"{compute['machine']}, {compute['cpu_count']} CPUs "
                  f"({cpu_model}), GPU: {gpu}; recorded into node-manifest.json")


# --------------------------------------------------------------- 4–8
# Build slice 3: checks 4-6 and 8 are REAL implementations probing
# the arm bridge (arm/). On a machine with no arm they still report
# not_present — the bridge's absence is detected, never papered
# over. Each check accepts an injected adapter (the MOCK bus) so the
# pass/fail paths are exercised in tests; production runs build the
# real LeRobot adapter from the resolved port.
def _serial_ports():
    return sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


def _real_adapter(cfg):
    """(adapter, None) or (None, (status, detail)) for checks 4/8."""
    from .arm.bus import RealBusAdapter, find_arm_port
    pinned = (cfg.get("arm") or {}).get("port")
    port, how = find_arm_port(pinned)
    if port is None:
        if how.startswith("ambiguous"):
            return None, (FAIL, f"arm port cannot be resolved: {how}",
                          "pin arm.port in config to the arm's "
                          "/dev/serial/by-id path")
        return None, (NOT_PRESENT,
                      f"no arm bus present ({how}) — expected on a "
                      f"machine with no arm attached", None)
    return RealBusAdapter(port), None


def _mock_note(adapter):
    return (" [MOCK adapter — interface evidence only, not hardware]"
            if getattr(adapter, "is_mock", False) else "")


def check_arm_bus(cfg, adapter=None):
    from .arm.bridge import ArmBridge, BridgeError
    from .arm.registers import decode_status
    if adapter is None:
        adapter, absent = _real_adapter(cfg)
        if adapter is None:
            status, detail, fix = absent
            return _check(4, "Arm bus scan", status,
                          detail + "; the scan, firmware-uniformity "
                          "audit (backlog S2) and voltage/fault "
                          "naming (S3) run against the bridge when "
                          "an arm is present", fix=fix)
    bridge = ArmBridge(adapter)
    try:
        conn = bridge.connect()
    except BridgeError as e:
        return _check(4, "Arm bus scan", FAIL, f"bus audit failed: {e}",
                      fix="uniform servo firmware is required — "
                          "LeRobot's connect asserts the same; update "
                          "the odd servo or replace it")
    except Exception as e:  # noqa: BLE001 — doctor reports, never crashes
        return _check(4, "Arm bus scan", FAIL,
                      f"bus scan failed: {type(e).__name__}: {e}")
    try:
        state = bridge.read_contract_state()
    except Exception as e:  # noqa: BLE001
        return _check(4, "Arm bus scan", FAIL,
                      f"contract read failed: {type(e).__name__}: {e}")
    finally:
        bridge.disconnect()
    problems = []
    volts = state["voltage_v"]
    for m, v in volts.items():
        if not (6.0 <= v <= 8.4):
            problems.append(f"{m} bus voltage {v} V outside the "
                            f"STS3215 7.4 V system's sane band "
                            f"(6.0-8.4 V) — check the supply before "
                            f"running the arm (backlog S3)")
    for m, bits in state["status_faults"].items():
        problems.append(f"{m} reports servo status fault(s): "
                        f"{', '.join(bits)} (backlog S3)")
    ids = sorted(int(k) for k in conn["scan"])
    if problems:
        return _check(4, "Arm bus scan", FAIL, "; ".join(problems))
    fw = {v["firmware"] for v in conn["audit"].values()}
    return _check(4, "Arm bus scan", PASS,
                  f"six servos answer on the bus at ids {ids}; "
                  f"firmware uniform ({', '.join(sorted(fw))}, "
                  f"backlog S2); bus voltage "
                  f"{min(volts.values())}-{max(volts.values())} V; "
                  f"no servo status faults (S3); telemetry read "
                  f"over contract channels only"
                  + _mock_note(adapter))


def check_port_identity(cfg, by_id_dir="/dev/serial/by-id",
                        acm_pattern="/dev/ttyACM*"):
    from .arm.bus import find_arm_port
    pinned = (cfg.get("arm") or {}).get("port")
    if pinned:
        if os.path.exists(pinned):
            return _check(5, "Port identity", PASS,
                          f"arm port pinned to the stable path "
                          f"{pinned} (never enumeration order, "
                          f"backlog S4)")
        return _check(5, "Port identity", NOT_PRESENT,
                      f"pinned arm port {pinned} does not exist — "
                      f"the arm is not attached (or the pin is "
                      f"stale; it names the path so a human can "
                      f"tell)")
    port, how = find_arm_port(None, by_id_dir, acm_pattern)
    if port is None:
        if how.startswith("ambiguous"):
            return _check(5, "Port identity", FAIL,
                          f"arm port cannot be told apart: {how}",
                          fix="pin arm.port in config to the arm's "
                              "/dev/serial/by-id path")
        return _check(5, "Port identity", NOT_PRESENT,
                      f"no arm ports to pin ({how}); ports are "
                      f"pinned by stable device path, never by "
                      f"enumeration order (backlog S4)")
    if port.startswith("/dev/tty"):
        return _check(5, "Port identity", FAIL,
                      f"arm found at {port} but NOT pinned — an "
                      f"enumeration-order path can name a different "
                      f"device after a reboot (backlog S4)",
                      fix="pin arm.port in config to the arm's "
                          "/dev/serial/by-id path")
    return _check(5, "Port identity", FAIL,
                  f"arm resolves to {port} ({how}) but the pin is "
                  f"not recorded in config — a resolution that "
                  f"lives only in this run is not an identity",
                  fix="pin arm.port in config to that by-id path")


def _capture_frame(path):
    """One frame via the runtime's camera stack; returns
    (width, height, mean, std) or raises."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"no frame from {path}")
        gray = frame.mean(axis=2) if frame.ndim == 3 else frame
        return (int(frame.shape[1]), int(frame.shape[0]),
                float(gray.mean()), float(gray.std()))
    finally:
        cap.release()


def check_camera_identity(cfg, devices=None, capture=None):
    declared = ((cfg.get("arm") or {}).get("cameras")
                or {"front": {"path": None}, "wrist": {"path": None}})
    if devices is None:
        devices = []
        for p in sorted(glob.glob("/dev/video*")):
            name = ""
            sysfs = Path("/sys/class/video4linux") / Path(p).name / "name"
            try:
                name = sysfs.read_text().strip()
            except OSError:
                pass
            devices.append((p, name))
    if not devices:
        return _check(6, "Camera identity & viewpoint", NOT_PRESENT,
                      "no camera devices found (/dev/video*) — the "
                      "declared camera set (front + wrist, the "
                      "vision clause's declared poses) and the "
                      "one-frame viewpoint verification (backlog "
                      "S5) apply when cameras are attached")
    capture = capture or _capture_frame
    verified, problems = [], []
    for role, spec in sorted(declared.items()):
        pin = (spec or {}).get("path")
        if not pin:
            problems.append(
                f"{role} camera is not pinned to a device path — "
                f"identity by enumeration order is not identity "
                f"(backlog S5)")
            continue
        if not os.path.exists(pin):
            problems.append(f"{role} camera's pinned path {pin} "
                            f"does not exist")
            continue
        try:
            w, h, mean, std = capture(pin)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{role} camera at {pin}: frame "
                            f"capture failed ({e})")
            continue
        if std < 1.0:
            problems.append(f"{role} camera at {pin}: frame is "
                            f"flat (std {std:.2f}) — lens covered "
                            f"or the device is not a camera view")
            continue
        verified.append(f"{role}: {w}x{h} frame at {pin}, "
                        f"mean {mean:.0f}, std {std:.1f}")
    if problems:
        return _check(6, "Camera identity & viewpoint", FAIL,
                      "; ".join(problems + verified),
                      fix="pin arm.cameras.<role>.path in config to "
                          "each camera's /dev/v4l/by-id path and "
                          "re-run doctor")
    return _check(6, "Camera identity & viewpoint", PASS,
                  "declared camera set matches the attached "
                  "devices; one frame captured and checked per "
                  "camera (S5): " + "; ".join(verified))


def check_calibration_state(cfg):
    # Slice 2: calibration lives in the node config (Stage 2 state),
    # fingerprinted against the staged arm model; the M0 gate
    # (Stage 3) is what verifies it — doctor reports the state.
    from . import calibration as C
    from .sim import assets
    calib = C.ensure_calibration(cfg)
    md5 = assets.arm_model_info("realistic").get("canonical_md5")
    if calib["generation"] < 1:
        return _check(7, "Calibration state", SKIP,
                      "calibration generation 0 — this node has never "
                      "been calibrated (Stage 2); the M0 join gate "
                      "will refuse it until it is")
    expect = C.calibration_fingerprint(calib, md5) if md5 else None
    if calib.get("fingerprint") != expect:
        return _check(7, "Calibration state", FAIL,
                      "stored calibration fingerprint does not match "
                      "the calibration in config — the calibration "
                      "changed since it was fingerprinted; the M0 "
                      "gate is re-opened until it is refreshed and "
                      "the gate re-run",
                      fix="refresh the fingerprint over the verified "
                          "calibration (Stage 2/3), then `zk0 gate run`")
    return _check(7, "Calibration state", PASS,
                  f"generation {calib['generation']}, action_norm "
                  f"{calib['action_norm'].get('type')}, fingerprint "
                  f"current ({calib['fingerprint'][:16]}…)")


# LeRobot's protective configuration (its configure() writes these
# at connect; the contract's section 2 table records them): the
# gripper's torque ceiling is what stands between a bad grasp and a
# stalled clamp at full torque (backlog O1).
EXPECTED_SERVO_CONFIG = {
    "gripper": {"Max_Torque_Limit": 500, "Protection_Current": 250,
                "Overload_Torque": 25},
    "pid": {"Position_P_Coefficient": 16, "Position_I_Coefficient": 0,
            "Position_D_Coefficient": 32},
}


def check_servo_configuration(cfg, adapter=None):
    if adapter is None:
        adapter, absent = _real_adapter(cfg)
        if adapter is None:
            status, detail, fix = absent
            return _check(8, "Servo configuration", status,
                          detail + "; the protective servo "
                          "configuration (gripper Max_Torque_Limit, "
                          "protection current, overload torque; "
                          "backlog O1) is verified against the arm "
                          "when one is attached", fix=fix)
    try:
        adapter.connect()
        conf = adapter.read_servo_config()
    except Exception as e:  # noqa: BLE001 — doctor reports, never crashes
        return _check(8, "Servo configuration", FAIL,
                      f"servo configuration read failed: "
                      f"{type(e).__name__}: {e}")
    finally:
        try:
            adapter.disconnect()
        except Exception:  # noqa: BLE001
            pass
    problems = []
    for reg, want in EXPECTED_SERVO_CONFIG["gripper"].items():
        got = conf.get("gripper", {}).get(reg)
        if got != want:
            problems.append(f"gripper {reg} = {got}, expected "
                            f"{want} (LeRobot's protective "
                            f"configuration, backlog O1)")
    from .calibration import JOINT_NAMES
    for m in JOINT_NAMES:
        for reg, want in EXPECTED_SERVO_CONFIG["pid"].items():
            got = conf.get(m, {}).get(reg)
            if got != want:
                problems.append(f"{m} {reg} = {got}, expected {want}")
    if problems:
        return _check(8, "Servo configuration", FAIL,
                      "; ".join(sorted(set(problems))),
                      fix="reconnect through the bridge (LeRobot's "
                          "configure() writes the protective values "
                          "at connect) and re-run doctor")
    return _check(8, "Servo configuration", PASS,
                  "protective configuration verified on all six "
                  "servos: position PID 16/0/32; gripper "
                  "Max_Torque_Limit 500 (50%), Protection_Current "
                  "250, Overload_Torque 25 (backlog O1)"
                  + _mock_note(adapter))


# ------------------------------------------------------------------ 9
def check_sim_stack(cfg):
    try:
        import mujoco
    except ImportError:
        return _check(9, "Sim stack", FAIL, "mujoco not importable in the runtime")
    info = assets.arm_model_info("realistic")
    staged = assets.model_dir("realistic") / "so101.xml"
    if not staged.exists():
        return _check(9, "Sim stack", FAIL,
                      "staged sim model missing (model-realistic/so101.xml)",
                      fix="run dev/bootstrap.sh to stage the pinned assets")
    try:
        model = mujoco.MjModel.from_xml_path(str(staged))
    except Exception as e:  # noqa: BLE001 — doctor reports, never crashes
        return _check(9, "Sim stack", FAIL, f"MuJoCo failed to load the staged "
                                            f"arm model: {e}")
    return _check(9, "Sim stack", PASS,
                  f"MuJoCo {mujoco.__version__} loads so101-realistic "
                  f"(nq={model.nq}, nu={model.nu}); staged asset md5 "
                  f"{info.get('live_include_md5')} matches the pinned "
                  f"canonical file: "
                  f"{info.get('live_include_md5') == info.get('canonical_md5')}. "
                  f"Servo profiles: variant (MJCF position actuators) "
                  f"and BAM fitted-servo (ZK0_SERVO=bam, build "
                  f"slice 3). Telemetry-contract firewall: the "
                  f"benchmark controller is the scripted oracle "
                  f"pilot (rung 0, diagnostic); the checkpoint "
                  f"evaluator (Stage 5) consumes declared camera "
                  f"frames + contract state only")


# ----------------------------------------------------------------- 10
def check_safety_layer(cfg):
    staged = assets.model_dir("realistic") / "so101.xml"
    if not staged.exists():
        return _check(10, "Safety layer", FAIL,
                      "cannot verify limits without the staged arm model",
                      fix="run dev/bootstrap.sh to stage the pinned assets")
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(staged))
    grip_range = model.jnt_range[model.joint("gripper").id]
    applied, clamped = clamp_command(-0.65, grip_range)
    problems = []
    if not clamped or applied != float(grip_range[0]):
        problems.append(f"clamp probe failed: -0.65 rad -> {applied} "
                        f"(expected the joint floor {float(grip_range[0])})")
    for label, pt in (("SIM-1 cube start", (0.26, 0.02, 0.0205)),
                      ("SIM-1 box centre", (0.27, -0.14, 0.035))):
        if not workspace_contains(DEFAULT_LIMITS, pt):
            problems.append(f"workspace box does not contain {label} {pt}")
    if problems:
        return _check(10, "Safety layer", FAIL, "; ".join(problems))
    return _check(10, "Safety layer", PASS,
                  f"limits config present (saturation abort at "
                  f"{DEFAULT_LIMITS['gripper_saturation_abort_frac']:.0%} of "
                  f"carry, damage bound {DEFAULT_LIMITS['damage_bound_N']} N "
                  f"PROVISIONAL); clamp probe: the S6b out-of-range command "
                  f"-0.65 rad is clamped to the gripper floor "
                  f"{float(grip_range[0])} rad and counted; workspace box "
                  f"contains the SIM-1 scene. Arm-side enforcement is "
                  f"LIVE in the bridge since build slice 3 "
                  f"(arm/safety_layer.py): every bridge action passes "
                  f"range clamp + count, per-command speed cap, "
                  f"wrist_roll seam guard, workspace box, command "
                  f"watchdog, and the gripper saturation abort before "
                  f"it reaches the bus; the MOCK demonstration "
                  f"(`zk0 arm demo-mock`) exercises each refusal")


# ----------------------------------------------------------------- 11
def check_network(cfg, probe=None):
    if cfg.get("mode") == "sim":
        return _check(11, "Network", SKIP,
                      "sim mode: no network required (spec §2); the "
                      "local-network scope, when joined, is checked "
                      "at enrollment time by the aggregator")
    if cfg.get("mode") == "staging":
        ep = (cfg.get("endpoints") or {}).get("staging") or ""
        url = None
        if "(enroll " in ep and ep.rstrip().endswith(")"):
            url = ep.split("(enroll ", 1)[1].rstrip(")")
        if not url:
            return _check(11, "Network", SKIP,
                          "staging mode; no staging aggregator "
                          "endpoint recorded yet — `zk0 fl "
                          "join-staging` records one after a round; "
                          "the staging doctor subset is checks 3 "
                          "and 11 (spec §1.4)")
        if probe is None:
            def probe(u):
                import urllib.request
                with urllib.request.urlopen(u.rstrip("/") + "/status",
                                            timeout=3) as r:
                    return json.loads(r.read())
        try:
            st = probe(url)
        except Exception as e:  # noqa: BLE001
            return _check(11, "Network", FAIL,
                          f"staging aggregator at {url} is not "
                          f"reachable ({type(e).__name__}: {e}) — a "
                          f"staging node that cannot reach its "
                          f"aggregator cannot contribute",
                          fix="check the staging service is running "
                              "(zk0 fl start-staging under the "
                              "staging service home)")
        if st.get("scope") != "staging":
            return _check(11, "Network", FAIL,
                          f"endpoint {url} answers but is not a "
                          f"staging aggregator (scope="
                          f"{st.get('scope')!r}) — scopes are not "
                          f"blurred (spec §2)")
        return _check(11, "Network", PASS,
                      f"staging aggregator reachable at {url}: "
                      f"network {st.get('network_id')}, scope "
                      f"staging, enrolled nodes "
                      f"{st.get('enrolled')}")
    return _check(11, "Network", SKIP,
                  f"mode is {cfg.get('mode')}; prod connector "
                  f"reachability is moot while prod enrollment is "
                  f"hard-blocked (spec §2.1, Decision 56)")


# ----------------------------------------------------------------- 12
def check_storage(cfg):
    home = zk0_home()
    home.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(home).free / 1e9
    need_gb = 5.0
    status = PASS if free_gb >= need_gb else FAIL
    return _check(12, "Storage", status,
                  f"{free_gb:.1f} GB free at {home} (datasets, checkpoints "
                  f"and the retained previous shared checkpoint, §1.5; "
                  f"slice-1 floor {need_gb:.0f} GB)")


CHECKS = [check_runtime_integrity, check_dependency_consistency,
          check_compute_profile, check_arm_bus, check_port_identity,
          check_camera_identity, check_calibration_state,
          check_servo_configuration, check_sim_stack, check_safety_layer,
          check_network, check_storage]


def run_doctor(cfg=None) -> dict:
    cfg = cfg or load_config()
    results = [fn(cfg) for fn in CHECKS]
    bundle = {
        "bundle": "zk0-doctor-diagnostic/0.1",
        "created_utc": utcnow_iso(),
        "node_id": cfg["node_id"], "mode": cfg["mode"],
        "package_version": __version__,
        "runtime_manifest_sha256": manifest_sha256(runtime_manifest_path()),
        "checks": results,
        "summary": {s: sum(1 for r in results if r["status"] == s)
                    for s in (PASS, FAIL, SKIP, NOT_PRESENT)},
    }
    out = zk0_home() / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    stamp = bundle["created_utc"].replace(":", "").replace("-", "")
    (out / f"doctor-{stamp}.json").write_text(json.dumps(bundle, indent=1))
    bundle["bundle_path"] = str(out / f"doctor-{stamp}.json")
    return bundle
