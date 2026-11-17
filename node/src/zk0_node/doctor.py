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
                  f"the pinned {asset['md5']}")


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
def _serial_ports():
    return sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


def check_arm_bus(cfg):
    ports = _serial_ports()
    if not ports:
        return _check(4, "Arm bus scan", NOT_PRESENT,
                      "no serial arm devices found (/dev/ttyACM*, /dev/ttyUSB*) "
                      "— expected on a machine with no arm attached; the bus "
                      "scan, firmware-uniformity audit (backlog S2) and "
                      "voltage/fault naming (S3) run when an arm is present")
    return _check(4, "Arm bus scan", SKIP,
                  f"serial devices present {ports}, but the arm bridge "
                  f"(bus scan / firmware audit) lands in build slice 2")


def check_port_identity(cfg):
    ports = _serial_ports()
    if not ports:
        return _check(5, "Port identity", NOT_PRESENT,
                      "no arm ports to pin; ports are pinned by stable "
                      "device path, never by enumeration order (backlog S4)")
    return _check(5, "Port identity", SKIP,
                  f"ports {ports} present; stable-path pinning lands with "
                  f"the arm bridge (build slice 2)")


def check_camera_identity(cfg):
    cams = sorted(glob.glob("/dev/video*"))
    if not cams:
        return _check(6, "Camera identity & viewpoint", NOT_PRESENT,
                      "no camera devices found (/dev/video*) — the declared "
                      "camera set (front + wrist) and the one-frame viewpoint "
                      "verification (backlog S5) apply when cameras are attached")
    return _check(6, "Camera identity & viewpoint", SKIP,
                  f"cameras {cams} present; viewpoint verification lands "
                  f"with the arm bridge (build slice 2)")


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


def check_servo_configuration(cfg):
    if not _serial_ports():
        return _check(8, "Servo configuration", NOT_PRESENT,
                      "no arm bus — the protective servo configuration "
                      "(gripper Max_Torque_Limit, protection current, "
                      "overload torque; backlog O1) is verified against "
                      "the real arm at bring-up (Stage 2)")
    return _check(8, "Servo configuration", SKIP,
                  "arm bus present; servo configuration audit lands with "
                  "the arm bridge (build slice 2)")


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
                  f"Telemetry-contract firewall: the slice-1 benchmark "
                  f"controller is the scripted oracle pilot (rung 0, "
                  f"diagnostic); contract-signal controllers land with "
                  f"the train pack (build slice 2)")


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
                  f"contains the SIM-1 scene. Arm-side enforcement lands "
                  f"with the arm bridge (build slice 2)")


# ----------------------------------------------------------------- 11
def check_network(cfg):
    if cfg.get("mode") == "sim":
        return _check(11, "Network", SKIP,
                      "sim mode: no network required (spec §2)")
    return _check(11, "Network", SKIP,
                  f"mode is {cfg.get('mode')}; connector reachability and "
                  f"round-enrollment probes land with the staging wiring "
                  f"(build slice 2)")


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
