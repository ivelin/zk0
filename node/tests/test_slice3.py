"""Slice-3 unit tests: the arm bridge + safety layer (against the
MOCK bus — interface evidence only), doctor checks 4-8/11 as real
implementations, the BAM profile's byte identity, and the staging
enrollment mode rule.

Stdlib unittest only, like the slice-1/2 tests. Tests that need the
staged sim assets or the sim stack skip when they are absent.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zk0_node.arm import registers
from zk0_node.arm.bus import find_arm_port
from zk0_node.arm.mock import MockBusAdapter
from zk0_node.arm.safety_layer import (SafetyAbort, SafetyLayer,
                                       SafetyRefusal)


def _ranges():
    r = {j: (-110.0, 110.0) for j in
         ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll")}
    r["gripper"] = (0.0, 100.0)
    return r


def _present(**over):
    p = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
         "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 100.0}
    p.update(over)
    return p


class TestContractTable(unittest.TestCase):
    def test_register_addresses_match_the_contract(self):
        # Contract section 2 (STS3215 control table): the load-
        # bearing addresses, checked against the signed-off table.
        self.assertEqual(registers.CONTRACT_CHANNELS[
            "Present_Position"][:2], (56, 2))
        self.assertEqual(registers.CONTRACT_CHANNELS[
            "Present_Load"][:2], (60, 2))
        self.assertEqual(registers.CONTRACT_CHANNELS[
            "Present_Voltage"][:2], (62, 1))
        self.assertEqual(registers.CONTRACT_CHANNELS[
            "Present_Current"][:2], (69, 2))
        self.assertEqual(registers.CONTRACT_CHANNELS[
            "Goal_Position"][:2], (42, 2))

    def test_mock_refuses_non_contract_registers(self):
        bus = MockBusAdapter()
        bus.connect()
        with self.assertRaises(ValueError):
            bus.read("Torque_Enable")
        state = bus.read_state()
        self.assertEqual(sorted(state.keys()),
                         sorted(registers.FEEDBACK_CHANNELS))


class TestSafetyLayer(unittest.TestCase):
    def test_clamp_is_applied_and_counted(self):
        layer = SafetyLayer(_ranges())
        applied = layer.check_action(_present(), {"gripper": 130.0})
        self.assertEqual(applied["gripper"], 100.0)
        self.assertEqual(layer.clamp_count, 1)

    def test_speed_cap_refuses_not_clamps(self):
        layer = SafetyLayer(_ranges())
        with self.assertRaises(SafetyRefusal) as ctx:
            layer.check_action(_present(), {"shoulder_pan": 100.0})
        self.assertEqual(ctx.exception.check, "speed_cap")

    def test_seam_guard_refuses_wrist_roll_wrap(self):
        layer = SafetyLayer(_ranges())
        with self.assertRaises(SafetyRefusal) as ctx:
            layer.check_action(_present(wrist_roll=-100.0),
                               {"wrist_roll": 100.0})
        self.assertEqual(ctx.exception.check, "seam_guard")

    def test_workspace_refusal_with_stub_fk(self):
        layer = SafetyLayer(_ranges(),
                            fk_tcp=lambda pose: (0.9, 0.0, 0.2))
        with self.assertRaises(SafetyRefusal) as ctx:
            layer.check_action(_present(), {"shoulder_pan": 5.0})
        self.assertEqual(ctx.exception.check, "workspace")

    def test_saturation_abort_latches_stop(self):
        layer = SafetyLayer(_ranges())
        with self.assertRaises(SafetyAbort):
            for _ in range(20):
                layer.feed_gripper_load(-1000.0)
        self.assertTrue(layer.stopped)
        with self.assertRaises(SafetyRefusal):
            layer.check_action(_present(), {"gripper": 50.0})

    def test_watchdog_hold_then_resume(self):
        now = [0.0]
        layer = SafetyLayer(_ranges(), clock=lambda: now[0])
        layer.check_action(_present(), {"shoulder_pan": 5.0})
        now[0] += 5.0
        with self.assertRaises(SafetyRefusal) as ctx:
            layer.check_action(_present(), {"shoulder_pan": 6.0})
        self.assertEqual(ctx.exception.check, "watchdog")
        self.assertTrue(layer.held)
        layer.resume()
        layer.check_action(_present(), {"shoulder_pan": 6.0})


class TestFindArmPort(unittest.TestCase):
    def test_pinned_missing_is_absent_not_invented(self):
        port, why = find_arm_port("/dev/serial/by-id/does-not-exist")
        self.assertIsNone(port)
        self.assertIn("does not exist", why)

    def test_single_by_id_device_resolves(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "usb-FTDI_FT232R_USB_UART_A1").touch()
            (Path(td) / "usb-Other_Device").touch()
            port, how = find_arm_port(None, by_id_dir=td,
                                      acm_pattern=str(Path(td) / "none-*"))
            self.assertIsNone(port)  # FTDI name is not Feetech-class

    def test_ambiguous_acm_is_named_ambiguous(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "ttyACM0").touch()
            (Path(td) / "ttyACM1").touch()
            port, why = find_arm_port(
                None, by_id_dir=str(Path(td) / "empty"),
                acm_pattern=str(Path(td) / "ttyACM*"))
            self.assertIsNone(port)
            self.assertTrue(why.startswith("ambiguous"))


class TestDoctorSlice3(unittest.TestCase):
    def test_check4_pass_with_mock_and_fail_with_mixed_firmware(self):
        from zk0_node import doctor
        ok = doctor.check_arm_bus({}, adapter=MockBusAdapter("normal"))
        self.assertEqual(ok["status"], "pass", ok)
        self.assertIn("MOCK", ok["detail"])
        bad = doctor.check_arm_bus(
            {}, adapter=MockBusAdapter("mixed_firmware"))
        self.assertEqual(bad["status"], "fail", bad)

    def test_check8_pass_with_mock_fail_with_bad_config(self):
        from zk0_node import doctor

        class BadCfg(MockBusAdapter):
            def read_servo_config(self):
                conf = super().read_servo_config()
                conf["gripper"]["Max_Torque_Limit"] = 1000
                return conf

        ok = doctor.check_servo_configuration(
            {}, adapter=MockBusAdapter("normal"))
        self.assertEqual(ok["status"], "pass", ok)
        bad = doctor.check_servo_configuration({}, adapter=BadCfg())
        self.assertEqual(bad["status"], "fail", bad)
        self.assertIn("Max_Torque_Limit", bad["detail"])

    def test_check5_pinned_paths(self):
        from zk0_node import doctor
        with tempfile.NamedTemporaryFile() as f:
            res = doctor.check_port_identity(
                {"arm": {"port": f.name, "cameras": {}}})
            self.assertEqual(res["status"], "pass", res)
        res = doctor.check_port_identity(
            {"arm": {"port": "/dev/serial/by-id/nope", "cameras": {}}})
        self.assertEqual(res["status"], "not_present", res)

    def test_check6_with_injected_devices(self):
        from zk0_node import doctor
        with tempfile.TemporaryDirectory() as td:
            front = Path(td) / "front-cam"
            wrist = Path(td) / "wrist-cam"
            front.touch(); wrist.touch()
            cfg = {"arm": {"port": None, "cameras": {
                "front": {"path": str(front)},
                "wrist": {"path": str(wrist)}}}}
            cap = lambda p: (640, 480, 110.0, 42.0)  # noqa: E731
            res = doctor.check_camera_identity(
                cfg, devices=[(str(front), "a"), (str(wrist), "b")],
                capture=cap)
            self.assertEqual(res["status"], "pass", res)
            cfg["arm"]["cameras"]["wrist"] = {"path": None}
            res = doctor.check_camera_identity(
                cfg, devices=[(str(front), "a")], capture=cap)
            self.assertEqual(res["status"], "fail", res)
        res = doctor.check_camera_identity({}, devices=[])
        self.assertEqual(res["status"], "not_present", res)

    def test_check11_staging_probe(self):
        from zk0_node import doctor
        cfg = {"mode": "staging",
               "endpoints": {"staging":
                             "127.0.0.1:8897 (enroll http://127.0.0.1:8898)"}}
        ok = doctor.check_network(
            cfg, probe=lambda u: {"scope": "staging",
                                  "network_id": "abc",
                                  "enrolled": []})
        self.assertEqual(ok["status"], "pass", ok)
        wrong = doctor.check_network(
            cfg, probe=lambda u: {"scope": "local"})
        self.assertEqual(wrong["status"], "fail", wrong)

        def boom(u):
            raise ConnectionError("refused")
        down = doctor.check_network(cfg, probe=boom)
        self.assertEqual(down["status"], "fail", down)


class TestStagingEnrollmentModeRule(unittest.TestCase):
    def test_aggregator_refuses_non_staging_mode(self):
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer

        from zk0_node.fl.network import Registry, _State, _enrollment_server
        from zk0_node.identity import NodeKey
        from zk0_node.records import content_sha256, sign_record

        import os as _os
        node_key = NodeKey(_os.urandom(32))
        server_key = NodeKey(_os.urandom(32))
        att = {"record_type": "join_gate_attestation",
               "node": {"node_id": "node-x"}, "pass": True,
               "expires_utc": "2999-01-01T00:00:00Z"}
        att["content_sha256"] = content_sha256(att)
        att = sign_record(att, node_key)
        registry, state = Registry(), _State()
        srv = _enrollment_server(0, "tok123", "net-x", server_key,
                                 registry, state, scope="staging",
                                 require_mode="staging")
        port = srv.socket.getsockname()[1]

        def post(mode):
            body = json.dumps({
                "node_id": "node-x", "token": "tok123", "mode": mode,
                "public_key_hex": node_key.public_key_hex,
                "attestation": att}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/enroll", data=body,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code

        try:
            self.assertEqual(post("sim"), 403)
            self.assertEqual(post("staging"), 200)
            self.assertEqual(registry.ids(), ["node-x"])
        finally:
            srv.shutdown()
            threading.Thread(target=srv.server_close, daemon=True).start()

    def test_client_refuses_staging_scope_in_sim_mode(self):
        from zk0_node.fl.client import run_client
        with self.assertRaises(ValueError):
            run_client("127.0.0.1:1", "http://127.0.0.1:1", "t",
                       "/nonexistent", cfg={"mode": "sim",
                                            "node_id": "n"},
                       scope="staging")


class TestBamProfileIdentity(unittest.TestCase):
    def test_vendored_profile_hashes_match_manifest(self):
        import hashlib
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "runtime-manifest.json").read_text())
        files = manifest["components"]["bam_profile"]["files_md5"]
        for rel, want in files.items():
            got = hashlib.md5(
                (root / "src" / "zk0_node" / "sim" / "bam" / rel
                 ).read_bytes()).hexdigest()
            self.assertEqual(got, want, rel)

    def test_installed_bam_params_are_the_certified_bytes(self):
        import hashlib
        try:
            import bam
        except ImportError:
            self.skipTest("bam not installed in this environment")
        live = (Path(bam.__file__).parent / "params" /
                "feetech_sts3215_7_4V" / "m6.json")
        self.assertEqual(hashlib.md5(live.read_bytes()).hexdigest(),
                         "3209fcb05bfbfddf68ae779fb87df02d")


class TestMockDemo(unittest.TestCase):
    def test_demo_record(self):
        from zk0_node.arm.bridge import run_mock_demo
        from zk0_node.sim import assets
        work = assets.model_dir("realistic")
        if not (work / "so101_realistic.xml").exists():
            self.skipTest("staged sim assets not present")
        with tempfile.TemporaryDirectory() as td:
            rec = run_mock_demo(Path(td), work)
        self.assertTrue(rec["mock"])
        self.assertEqual(rec["scope"], "mock")
        self.assertFalse(rec["hardware_evidence"])
        by_name = {s["step"]: s for s in rec["steps"]}
        walk = by_name["legal action sequence (calibration probe poses)"]
        self.assertEqual(walk["outcome"], "ok", walk)
        clamp = by_name["out-of-range goal clamped + counted (S6b rule)"]
        self.assertEqual(clamp["outcome"], "ok", clamp)
        self.assertEqual(clamp["clamps_counted"], 1)
        self.assertEqual(clamp["applied_gripper"], 100.0)
        wild = by_name["wild out-of-range goal refused by the safety layer"]
        self.assertEqual(wild["outcome"], "raised", wild)
        self.assertEqual(wild["error_type"], "SafetyRefusal")
        wd = by_name["command watchdog: stall -> hold -> resume"]
        self.assertEqual(wd["outcome"], "ok", wd)
        self.assertTrue(wd["watchdog_fired"])
        stall = by_name["gripper stall: sustained saturation aborts the run"]
        self.assertEqual(stall["outcome"], "raised", stall)
        self.assertEqual(stall["error_type"], "SafetyAbort")
        vf = by_name["voltage-fault bus: action blocked on servo status"]
        self.assertEqual(vf["outcome"], "raised", vf)
        self.assertEqual(vf["check"], "servo_status")
        mixed = by_name["mixed-firmware bus: connect refused"]
        self.assertEqual(mixed["outcome"], "raised", mixed)
        self.assertEqual(mixed["error_type"], "BridgeError")
        read = by_name["contract read set"]
        self.assertEqual(sorted(read["channels"]),
                         sorted(registers.FEEDBACK_CHANNELS))


if __name__ == "__main__":
    unittest.main()
