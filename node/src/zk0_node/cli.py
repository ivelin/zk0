"""The `zk0` CLI — a thin skin over zk0-core (spec §6.2).

Slice 1 commands: doctor, status, mode, enroll, sim run. Slice 2:
key, verify, gate, trainpack, eval, fl (local network). Slice 3:
arm (bridge scan + the MOCK demonstration), sim run --servo (the
BAM fitted-servo profile), fl start-staging / join-staging (the
staging connector path). Human-facing output leads with the
verdict in plain words; records and hashes are one flag away
(--json), not the default view.
"""
import argparse
import json
import sys

from . import __version__
from .config import (ProdEnrollmentBlocked, config_path, enroll_prod,
                     load_config, save_config, set_local_scope, set_mode)
from .doctor import run_doctor
from .records import jsonable


def _print_checks(bundle):
    glyph = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP",
             "not_present": "ABSENT"}
    for c in bundle["checks"]:
        detail = c["detail"]
        if isinstance(detail, list):
            detail = "\n      - " + "\n      - ".join(detail)
        print(f"[{glyph[c['status']]:6}] {c['id']:2}. {c['name']}: {detail}")
        if "fix" in c:
            print(f"        fix: {c['fix']}")
    s = bundle["summary"]
    print(f"\nsummary: {s['pass']} pass · {s['fail']} fail · "
          f"{s['skip']} skip · {s['not_present']} not present")
    print(f"diagnostic bundle: {bundle['bundle_path']}")


def _gate_or_refuse(cfg, founder_override) -> bool:
    """The M0 gate pre-check shared by fl join / join-staging:
    current attestation, else run the gate; a refusal is final
    unless a founder override is recorded. Prints its own verdicts;
    returns True when the caller may proceed."""
    from . import joingate
    if joingate.current_attestation(cfg) is not None:
        return True
    rec = joingate.run_gate(cfg)
    if rec["pass"]:
        return True
    if founder_override:
        try:
            ov = joingate.record_founder_override(
                cfg, rec, founder_override)
        except joingate.GateRefused as e:
            print(f"REFUSED — {e}")
            return False
        print(f"FOUNDER OVERRIDE recorded: {ov['path']} "
              f"(bypassed: {', '.join(ov['bypassed_checks'])})")
        return True
    print("JOIN REFUSED by the M0 gate:")
    for c in rec["checks"]:
        if not c["pass"]:
            print(f"  - {c['check']}: {c['reason']}")
    print(f"refusal record: {rec['path']}")
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="zk0", description="zk0 Node operator CLI")
    ap.add_argument("--version", action="version",
                    version=f"zk0 Node {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_doc = sub.add_parser("doctor", help="run the §1.4 self-checks")
    p_doc.add_argument("--json", action="store_true")

    sub.add_parser("status", help="node state: mode, identity, scopes")

    p_mode = sub.add_parser("mode", help="show or switch mode (spec §2.2)")
    p_mode.add_argument("target", nargs="?", choices=["sim", "staging", "prod"])
    p_mode.add_argument("--local-scope", choices=["join", "leave"],
                        help="join/leave the local-network scope (§2.3)")

    sub.add_parser("enroll", help="attempt prod enrollment (Stage 1 gates)")

    p_sim = sub.add_parser("sim", help="sim-mode commands")
    sim_sub = p_sim.add_subparsers(dest="sim_cmd", required=True)
    p_run = sim_sub.add_parser("run", help="run a benchmark task in sim")
    p_run.add_argument("--task", default="sim1", choices=["sim1"])
    p_run.add_argument("--arm-model", default="realistic",
                       choices=["realistic", "stock"],
                       help="realistic = so101-realistic (package default); "
                            "stock = the certified T1 control configuration")
    p_run.add_argument("--servo", default=None,
                       choices=["variant", "bam"],
                       help="servo profile: variant (default, the MJCF's "
                            "position actuators) or bam (the certified BAM "
                            "fitted-servo profile; also ZK0_SERVO)")
    p_run.add_argument("--out", default=None, help="output directory")

    p_arm = sub.add_parser("arm", help="the SO-101 arm bridge (slice 3)")
    arm_sub = p_arm.add_subparsers(dest="arm_cmd", required=True)
    arm_sub.add_parser("scan", help="scan the bus through the bridge: "
                                    "servos, firmware audit, contract state")
    arm_sub.add_parser("demo-mock", help="run the MOCK bridge demonstration "
                                         "(simulated bus — not hardware "
                                         "evidence)")

    p_key = sub.add_parser("key", help="node signing key (slice 2)")
    key_sub = p_key.add_subparsers(dest="key_cmd", required=True)
    key_sub.add_parser("show", help="show (creating if needed) the node key")

    p_ver = sub.add_parser("verify", help="verify a signed record")
    p_ver.add_argument("record", help="path to a record JSON")

    p_gate = sub.add_parser("gate", help="the M0 join gate (Stage 3)")
    gate_sub = p_gate.add_subparsers(dest="gate_cmd", required=True)
    gate_sub.add_parser("run", help="run the gate's four checks")
    gate_sub.add_parser("status", help="show the current attestation")

    p_tp = sub.add_parser("trainpack", help="train-pack proofs")
    tp_sub = p_tp.add_subparsers(dest="tp_cmd", required=True)
    tp_sub.add_parser("prove", help="LeRobot dataset write/read round-trip")

    p_ev = sub.add_parser("eval", help="checkpoint policy evaluation (Stage 5)")
    ev_sub = p_ev.add_subparsers(dest="ev_cmd", required=True)
    p_evc = ev_sub.add_parser("checkpoint", help="run a benchmark suite with the shared checkpoint as policy")
    p_evc.add_argument("--task", default="sim1", choices=["sim1", "sim2"])
    p_evc.add_argument("--trials", type=int, default=20)
    p_evc.add_argument("--ckpt-dir", default=None,
                       help="evaluate this checkpoint dir instead of "
                            "the staged shared checkpoint (Stages "
                            "8/10: fine-tuned / post-contribution "
                            "models); the record names the evaluated "
                            "dir")

    p_fl = sub.add_parser("fl", help="federated rounds (spec §2.3/§5)")
    fl_sub = p_fl.add_subparsers(dest="fl_cmd", required=True)
    p_ds = fl_sub.add_parser("build-dataset", help="build this node's sim demo dataset")
    p_ds.add_argument("--node-tag", required=True)
    p_ds.add_argument("--episodes", type=int, default=4)
    p_ds.add_argument("--seed", type=int, default=0)
    p_net = fl_sub.add_parser("start-network", help="run a local network (aggregator) on this machine")
    p_net.add_argument("--port", type=int, default=8797)
    p_net.add_argument("--enroll-port", type=int, default=8798)
    p_net.add_argument("--rounds", type=int, default=1)
    p_net.add_argument("--aggregator", default="fedprox", choices=["fedprox", "fedavg"])
    p_net.add_argument("--weighting", default="dynamic", choices=["dynamic", "plain"])
    p_net.add_argument("--mu", type=float, default=0.1)
    p_net.add_argument("--steps", type=int, default=8)
    p_net.add_argument("--min-clients", type=int, default=2)
    p_join = fl_sub.add_parser("join", help="join a local network and train a round")
    p_join.add_argument("--endpoint", required=True, help="aggregator host:port")
    p_join.add_argument("--enroll-url", required=True)
    p_join.add_argument("--token", required=True)
    p_join.add_argument("--dataset", default=None,
                        help="local LeRobot dataset root (required "
                             "unless --noop-dry-run)")
    p_join.add_argument("--noop-dry-run", action="store_true",
                        help="DRY-RUN PARTICIPANT: enroll and "
                             "contribute an identity/no-op update "
                             "(the round's initial parameters "
                             "unchanged, 0 examples), labeled as "
                             "such in every record it touches")
    p_join.add_argument("--steps", type=int, default=8)
    p_join.add_argument("--founder-override", default=None,
                        help="founder-scope only: bypass a refused gate with this stated reason (recorded)")
    p_stg = fl_sub.add_parser("start-staging",
                              help="run the STAGING aggregator (must run "
                                   "under the staging service home)")
    p_stg.add_argument("--rounds", type=int, default=1)
    p_stg.add_argument("--aggregator", default="fedprox", choices=["fedprox", "fedavg"])
    p_stg.add_argument("--weighting", default="dynamic", choices=["dynamic", "plain"])
    p_stg.add_argument("--mu", type=float, default=0.1)
    p_stg.add_argument("--steps", type=int, default=8)
    p_stg.add_argument("--min-clients", type=int, default=2)
    p_js = fl_sub.add_parser("join-staging",
                             help="join a STAGING round (node must be in "
                                  "staging mode)")
    p_js.add_argument("--endpoint", required=True, help="aggregator host:port")
    p_js.add_argument("--enroll-url", required=True)
    p_js.add_argument("--token", required=True)
    p_js.add_argument("--dataset", required=True, help="local LeRobot dataset root")
    p_js.add_argument("--steps", type=int, default=8)
    p_js.add_argument("--founder-override", default=None,
                      help="founder-scope only: bypass a refused gate with this stated reason (recorded)")

    p_flow = sub.add_parser("flow", help="task flow stages 7-11 "
                                         "(slice 4)")
    flow_sub = p_flow.add_subparsers(dest="flow_cmd", required=True)
    flow_sub.add_parser("admit", help="Stage 7: trajectory admission "
                                      "(decision 54) — generate "
                                      "candidates, run the envelope "
                                      "draws at rung 3, write the "
                                      "admitted dataset + signed "
                                      "admission record")
    p_ft = flow_sub.add_parser("finetune", help="Stage 8: local "
                                                "fine-tune on an "
                                                "admitted dataset")
    p_ft.add_argument("--dataset", required=True,
                      help="admitted LeRobot dataset root (Stage 7)")
    p_ft.add_argument("--steps", type=int, default=None)
    p_ft.add_argument("--lr", type=float, default=None)
    p_ft.add_argument("--mu", type=float, default=None,
                      help="FedProx proximal mu")
    p_ft.add_argument("--out", default=None)
    p_cmp = flow_sub.add_parser("compare", help="Stage 10: paired "
                                                "comparison from the "
                                                "six suite records")
    for a in ("--pre-sim1", "--pre-sim2", "--bench-sim1",
              "--bench-sim2", "--post-sim1", "--post-sim2"):
        p_cmp.add_argument(a, required=True,
                           help="eval_record.json path")
    p_cmp.add_argument("--round-record", default=None)
    p_cmp.add_argument("--out", default=None)
    p_dec = flow_sub.add_parser("declare", help="Stage 11: generate "
                                                "the win/no-win "
                                                "declaration FROM "
                                                "the records")
    p_dec.add_argument("--comparison", required=True,
                       help="comparison_record.json path")
    p_dec.add_argument("--round-record", default=None)
    p_dec.add_argument("--out", default=None)
    p_dry = flow_sub.add_parser("dry-run", help="stages 5-11 in one "
                                                "scripted pass from "
                                                "fresh node homes")
    p_dry.add_argument("--workdir", required=True)

    args = ap.parse_args(argv)
    cfg = load_config()
    if not config_path().exists():
        # First contact: persist the generated identity immediately so
        # the node_id is stable from the very first command (Stage 1
        # creates the node identity at install).
        save_config(cfg)

    if args.cmd == "doctor":
        bundle = run_doctor(cfg)
        if args.json:
            print(json.dumps(jsonable(bundle), indent=1))
        else:
            print(f"zk0 doctor — node {cfg['node_id'][:8]}…, mode {cfg['mode']}")
            _print_checks(bundle)
        return 1 if bundle["summary"]["fail"] else 0

    if args.cmd == "status":
        print(f"zk0 Node {__version__}")
        print(f"node_id:       {cfg['node_id']}")
        print(f"identity_kind: {cfg['identity_kind']}")
        print(f"mode:          {cfg['mode']}")
        print(f"network_scopes: {cfg['network_scopes'] or '[]'}")
        print(f"terms:         live={cfg['terms']['live']} "
              f"acceptance_record={cfg['terms']['acceptance_record']}")
        arm = cfg.get("arm") or {}
        print(f"arm port:      {arm.get('port') or 'not pinned'}")
        ep = (cfg.get("endpoints") or {}).get("staging")
        print(f"staging:       {ep}")
        return 0

    if args.cmd == "mode":
        if args.local_scope:
            notes = set_local_scope(cfg, args.local_scope == "join")
            save_config(cfg)
            for n in notes:
                print(n)
            return 0
        if not args.target:
            print(f"mode: {cfg['mode']} (scopes: {cfg['network_scopes']})")
            return 0
        try:
            notes = set_mode(cfg, args.target)
        except ProdEnrollmentBlocked as e:
            print("REFUSED — prod enrollment is hard-blocked for this node:")
            for r in e.reasons:
                print(f"  - {r}")
            return 3
        save_config(cfg)
        for n in notes:
            print(n)
        print(f"mode: {cfg['mode']}")
        return 0

    if args.cmd == "enroll":
        try:
            rec = enroll_prod(cfg)
        except ProdEnrollmentBlocked as e:
            print("REFUSED — prod enrollment is hard-blocked for this node:")
            for r in e.reasons:
                print(f"  - {r}")
            return 3
        print(f"prod enrollment accepted: {rec}")
        return 0

    if args.cmd == "sim" and args.sim_cmd == "run":
        from .sim.benchmark import run_sim1
        try:
            record = run_sim1(variant=args.arm_model, outdir=args.out,
                              cfg=cfg, servo=args.servo)
        except FileNotFoundError as e:
            print(f"sim run failed: {e}")
            return 2
        t1, t2 = record["results"]["tier1"], record["results"]["tier2"]
        print(f"SIM-1 (cube → box) — arm model: {args.arm_model}, "
              f"servo: {record['arm_model']['servo_profile'][:40]}…, "
              f"policy: scripted reference pilot (ORACLE-GUIDED, rung 0)")
        print(f"Tier 1: {'PASS' if t1['pass'] else 'FAIL'} "
              f"(containment {t1['containment']}, damage "
              f"{t1['damage']['pass']} — max single contact "
              f"{t1['damage']['max_single_contact_N_carry_window']} N vs "
              f"{t1['damage']['bound_N']} N PROVISIONAL bound, legal "
              f"actuation {t1['legal_actuation']['pass']})")
        print(f"Tier 2 (PROVISIONAL): gate "
              f"{'PASS' if t2['gate_pass'] else 'FAIL'}"
              + (f" — failed: {', '.join(t2['failed_gates'])}"
                 if t2["failed_gates"] else ""))
        print(f"records: {record['outdir']}")
        return 0

    if args.cmd == "arm" and args.arm_cmd == "scan":
        from .arm.bridge import ArmBridge, BridgeError
        from .arm.bus import RealBusAdapter, find_arm_port
        pinned = (cfg.get("arm") or {}).get("port")
        port, how = find_arm_port(pinned)
        if port is None:
            print(f"arm scan: NO ARM PRESENT — {how}. The bridge "
                  f"reports absence honestly; it never invents a "
                  f"bus. (The MOCK demonstration is `zk0 arm "
                  f"demo-mock`.)")
            return 5
        from .sim import assets
        bridge = ArmBridge(RealBusAdapter(port),
                           work_dir=assets.model_dir("realistic"))
        try:
            conn = bridge.connect()
            state = bridge.read_contract_state()
        except BridgeError as e:
            print(f"arm scan: bus audit FAILED — {e}")
            return 1
        finally:
            bridge.disconnect()
        print(f"arm scan: port {port} ({how})")
        print(f"servos answering: ids "
              f"{sorted(int(k) for k in conn['scan'])}")
        for m, a in conn["audit"].items():
            print(f"  {m:14} id {a['id']}  {a['model']}  "
                  f"firmware {a['firmware']}")
        print(f"positions (deg / gripper 0-100): "
              f"{ {k: round(v, 1) for k, v in state['positions'].items()} }")
        print(f"bus voltage: {state['voltage_v']}; status faults: "
              f"{state['status_faults'] or 'none'}")
        print("telemetry read over contract channels only: "
              + ", ".join(state["channels"]))
        return 0

    if args.cmd == "arm" and args.arm_cmd == "demo-mock":
        from .arm.bridge import run_mock_demo
        from .config import zk0_home
        from .identity import load_or_create_key
        from .records import write_signed_record
        from .sim import assets
        rec = run_mock_demo(zk0_home(), assets.model_dir("realistic"))
        rec["record_type"] = "arm_mock_demo"
        rec["schema"] = "zk0-node-arm/0.1"
        rec["node"] = {"node_id": cfg["node_id"]}
        from .records import utcnow_iso
        rec["created_utc"] = utcnow_iso()
        out = (zk0_home() / "records" / "arm" /
               f"mock-demo-{rec['created_utc'].replace(':', '').replace('-', '')}.json")
        write_signed_record(rec, out, load_or_create_key())
        print("MOCK bridge demonstration (simulated bus — NOT "
              "hardware evidence; hardware proof is the acceptance "
              "test on the founder's arm):")
        for s in rec["steps"]:
            if s["step"].startswith("safety events"):
                continue
            line = f"  [{s['outcome']}] {s['step']}"
            if s["outcome"] == "raised":
                line += f" — {s['error_type']}: {s['error']}"
            print(line)
        print(f"record (scope: mock, signed): {out}")
        return 0

    if args.cmd == "key" and args.key_cmd == "show":
        from .identity import key_path, load_or_create_key
        k = load_or_create_key()
        print(f"node_id:    {cfg['node_id']}")
        print(f"public key: {k.public_key_hex}")
        print(f"fingerprint: {k.fingerprint}")
        print(f"key file:   {key_path()} (mode 0600, never leaves this machine)")
        return 0

    if args.cmd == "verify":
        from .records import verify_record
        rec = json.loads(open(args.record).read())
        v = verify_record(rec)
        print(f"record: {rec.get('record_type', '?')} "
              f"(created {rec.get('created_utc', '?')})")
        print(f"content hash: {'OK' if v['hash_ok'] else 'MISMATCH'}")
        if v["signature_present"]:
            print(f"signature:    {'OK' if v['signature_ok'] else 'INVALID'} "
                  f"(signer fingerprint {v['signer']})")
        else:
            print("signature:    none (content-hash-only record)")
        # A content-hash-only record (the slice-1 form) verifies on
        # its hash alone; a signed record must also pass its
        # signature. An absent signature is not a failure — an
        # invalid one is.
        ok = v["hash_ok"] and (not v["signature_present"]
                               or v["signature_ok"])
        return 0 if ok else 1

    if args.cmd == "gate" and args.gate_cmd == "run":
        from .joingate import run_gate
        rec = run_gate(cfg)
        for c in rec["checks"]:
            print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['reason']}")
        if rec["pass"]:
            print(f"\nJOIN GATE: PASS — attestation {rec['path']}, "
                  f"expires {rec['expires_utc']}")
            return 0
        print(f"\nJOIN GATE: REFUSED — {'; '.join(rec['failed_checks'])}")
        print(f"refusal record: {rec['path']}")
        print("contribution stages stay disabled until the gate passes.")
        return 4

    if args.cmd == "gate" and args.gate_cmd == "status":
        from .joingate import current_attestation
        att = current_attestation(cfg)
        if att is None:
            print("no current attestation — run `zk0 gate run`")
            return 4
        print(f"attestation current: issued {att['created_utc']}, "
              f"expires {att['expires_utc']}, fingerprint "
              f"{att['calibration_fingerprint'][:16]}…")
        return 0

    if args.cmd == "trainpack" and args.tp_cmd == "prove":
        from .trainpack import prove
        try:
            res = prove()
        except Exception as e:
            print(f"train-pack proof FAILED: {e}")
            return 1
        print(f"train-pack round-trip: PASS — {res['frames_written']} frames "
              f"written with video (PyAV encode), {res['frames_reloaded']} "
              f"reloaded, action round-trip max err "
              f"{res['action_roundtrip_max_err']}")
        print(f"versions: {res['versions']}")
        return 0

    if args.cmd == "eval" and args.ev_cmd == "checkpoint":
        from .sim.policy_eval import run_suite
        try:
            rec = run_suite(args.task, trials=args.trials, cfg=cfg,
                            ckpt_dir=args.ckpt_dir)
        except FileNotFoundError as e:
            print(f"eval failed: {e}")
            return 2
        t1 = rec["results"]["tier1"]
        print(f"{rec['task']['id']} — policy: shared checkpoint "
              f"(SmolVLA), ablation rung: {rec['ablation_rung']}")
        print(f"Tier 1: {t1['successes']}/{t1['trials']} = "
              f"{t1['rate']:.0%} (95% Wilson CI "
              f"{t1['ci95_wilson'][0]:.0%}–{t1['ci95_wilson'][1]:.0%}); "
              f"bar (preregistration, rung 3): ≥60% — this run is "
              f"rung 0, the bar does not apply to it")
        print(f"records: {rec['outdir']}")
        return 0

    if args.cmd == "fl" and args.fl_cmd == "build-dataset":
        from .demodata import build_demo_dataset
        rec = build_demo_dataset(args.node_tag, episodes=args.episodes,
                                 seed=args.seed)
        print(f"dataset built: {rec['root']} — {rec['episodes']} episodes, "
              f"{rec['frames']} frames @ {rec['fps']} fps "
              f"(sim-generated, ORACLE-GUIDED; plan hash "
              f"{rec['plan_hash'][:16]}…)")
        return 0

    if args.cmd == "fl" and args.fl_cmd == "start-network":
        from .fl.network import start_network
        rec = start_network(port=args.port, enroll_port=args.enroll_port,
                            rounds=args.rounds, aggregator=args.aggregator,
                            weighting=args.weighting, proximal_mu=args.mu,
                            steps=args.steps, min_clients=args.min_clients,
                            cfg=cfg)
        print(f"round complete — network {rec['network_id']}, "
              f"participants: "
              f"{rec['participants'][0]['accepted_node_ids'] if rec['participants'] else []}")
        print(f"action-expert subset: {rec['before']['action_expert_subset_sha256'][:16]}… "
              f"→ {rec['after']['action_expert_subset_sha256'][:16]}…")
        print(f"round record (LOCAL, signed): {rec['path']}")
        return 0

    if args.cmd == "fl" and args.fl_cmd == "join":
        from .fl.client import run_client
        if not args.noop_dry_run and not args.dataset:
            print("REFUSED — --dataset is required unless "
                  "--noop-dry-run is given")
            return 2
        if not _gate_or_refuse(cfg, args.founder_override):
            return 4
        part = run_client(args.endpoint, args.enroll_url, args.token,
                          args.dataset, cfg=cfg, steps=args.steps,
                          noop=args.noop_dry_run)
        print(f"round joined and completed — network {part['network_id']}")
        print(f"shared model received: {part['received_model_dir']} "
              f"(sha256 verified against the round record)")
        print(f"participation record (LOCAL, signed): {part['path']}")
        return 0

    if args.cmd == "fl" and args.fl_cmd == "start-staging":
        from .config import zk0_home
        from .fl.staging import run_staging_aggregator, staging_service_home
        svc = staging_service_home()
        if zk0_home().resolve() != svc.resolve():
            print("REFUSED — the staging aggregator is a separate "
                  "service with its own home, key, and records; it "
                  "never runs inside a node's home.")
            print(f"run it as the staging service:")
            print(f"  ZK0_HOME={svc} ZK0_STAGING_HOME={svc} "
                  f"zk0 fl start-staging")
            print("(production: that home is the zk0bot account's "
                  "home on the Spark — key authorization still "
                  "outstanding; see the slice-3 report)")
            return 2
        rec = run_staging_aggregator(
            rounds=args.rounds, aggregator=args.aggregator,
            weighting=args.weighting, proximal_mu=args.mu,
            steps=args.steps, min_clients=args.min_clients, cfg=cfg)
        print(f"STAGING round complete — network {rec['network_id']}, "
              f"participants: "
              f"{rec['participants'][0]['accepted_node_ids'] if rec['participants'] else []}")
        print(f"round record (STAGING, signed): {rec['path']}")
        return 0

    if args.cmd == "fl" and args.fl_cmd == "join-staging":
        from .fl.client import run_client
        if cfg.get("mode") != "staging":
            print(f"REFUSED — staging rounds enroll only "
                  f"staging-mode nodes; this node is in mode "
                  f"{cfg.get('mode')!r}. Run `zk0 mode staging` "
                  f"first (spec §2: scopes are not blurred).")
            return 3
        if not _gate_or_refuse(cfg, args.founder_override):
            return 4
        part = run_client(args.endpoint, args.enroll_url, args.token,
                          args.dataset, cfg=cfg, steps=args.steps,
                          scope="staging")
        print(f"STAGING round joined and completed — network "
              f"{part['network_id']}")
        print(f"shared model received: {part['received_model_dir']} "
              f"(sha256 verified against the round record)")
        print(f"participation record (STAGING, signed): {part['path']}")
        return 0

    if args.cmd == "flow" and args.flow_cmd == "admit":
        from .flow import run_admission
        rec = run_admission(cfg=cfg)
        s = rec["summary"]
        print(f"Stage 7 admission: {s['admitted']}/{s['candidates']} "
              f"candidates admitted "
              f"(SIM-1 {s['per_task']['SIM-1']['admitted']}/"
              f"{s['per_task']['SIM-1']['candidates']}, "
              f"SIM-2 {s['per_task']['SIM-2']['admitted']}/"
              f"{s['per_task']['SIM-2']['candidates']}); "
              f"dataset: {(rec['dataset'] or {}).get('root')}")
        print(json.dumps({"record": rec["path"],
                          "admitted": s["admitted"],
                          "candidates": s["candidates"]}))
        return 0

    if args.cmd == "flow" and args.flow_cmd == "finetune":
        from .flow import run_finetune
        rec = run_finetune(args.dataset, steps=args.steps,
                           lr=args.lr, proximal_mu=args.mu,
                           out_dir=args.out, cfg=cfg)
        b = rec["budget"]
        print(f"Stage 8 local fine-tune: {b['steps']} steps @ lr "
              f"{b['lr']} in {b['wall_clock_s']} s on {b['hardware']}; "
              f"train loss {rec['training_metrics']['train_loss']:.4f}")
        print(json.dumps({"record": rec["path"],
                          "model_dir": rec["model_dir"]}))
        return 0

    if args.cmd == "flow" and args.flow_cmd == "compare":
        from .flow import run_comparison
        rec = run_comparison(
            {"SIM-1": args.pre_sim1, "SIM-2": args.pre_sim2},
            {"SIM-1": args.bench_sim1, "SIM-2": args.bench_sim2},
            {"SIM-1": args.post_sim1, "SIM-2": args.post_sim2},
            round_record_path=args.round_record,
            out_dir=args.out, cfg=cfg)
        for task, t in rec["tasks"].items():
            print(f"{task}: pre {t['pre_contribution']['rate']:.0%} "
                  f"-> benchmark {t['stage8_benchmark']['rate']:.0%} "
                  f"-> post {t['post_contribution']['rate']:.0%} "
                  f"(parity margin "
                  f"{t['parity']['margin_pp']:+.1f}pp)")
        print(json.dumps({"record": rec["path"]}))
        return 0

    if args.cmd == "flow" and args.flow_cmd == "declare":
        from .flow import run_declaration
        rec = run_declaration(args.comparison,
                              round_record_path=args.round_record,
                              out_dir=args.out, cfg=cfg)
        print(rec["declaration_text"])
        print(json.dumps({"record": rec["path"],
                          "win": rec["verdict"]["win"]}))
        return 0

    if args.cmd == "flow" and args.flow_cmd == "dry-run":
        from .flow import run_dry_run
        rec = run_dry_run(args.workdir, cfg=cfg)
        print(f"full-loop dry run: status={rec['status']}, "
              f"{len(rec['chain'])} records in chain, all verified: "
              f"{rec['chain_all_verified']}, total "
              f"{rec['total_wall_clock_s']} s")
        print(json.dumps({"record": rec["path"],
                          "status": rec["status"],
                          "chain_all_verified":
                              rec["chain_all_verified"]}))
        return 0 if rec["status"] == "complete" else 5

    return 2


if __name__ == "__main__":
    sys.exit(main())
