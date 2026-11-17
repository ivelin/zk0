"""The `zk0` CLI — a thin skin over zk0-core (spec §6.2).

Slice 1 commands: doctor, status, mode, enroll, sim run. Human-facing
output leads with the verdict in plain words; records and hashes are
one flag away (--json), not the default view.
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
    p_run.add_argument("--out", default=None, help="output directory")

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
            record = run_sim1(variant=args.arm_model, outdir=args.out, cfg=cfg)
        except FileNotFoundError as e:
            print(f"sim run failed: {e}")
            return 2
        t1, t2 = record["results"]["tier1"], record["results"]["tier2"]
        print(f"SIM-1 (cube → box) — arm model: {args.arm_model}, "
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

    return 2


if __name__ == "__main__":
    sys.exit(main())
