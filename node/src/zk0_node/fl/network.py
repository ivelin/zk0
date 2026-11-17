"""The operator-run LOCAL aggregator (spec section 2.3).

`zk0 fl start-network` runs, on the operator's own machine:
- a Flower server (the aggregator), and
- a small enrollment HTTP service on the LAN: nodes present the
  network's join token + their signed M0 gate attestation; the
  server verifies both (the attestation's hash and node-key
  signature, its pass verdict, its expiry) and issues a signed
  enrollment credential.

Aggregation (spec section 5.3 + preregistration Q7):
- Default aggregator: **FedProx + dynamic weighting**. The proximal
  term is applied client-side (the server passes proximal_mu in the
  round config — that is where FedProx lives); the weighting is
  applied here, exactly as stated in every round record:

    weight_i  ∝  n_i * clip(median_loss / loss_i, 0.5, 2.0)

  where n_i is the node's example count and loss_i its reported
  mean training loss for the round. FedAvg is the same code with
  mu = 0 and plain n_i weighting, selectable per network.
- Only updates from enrolled nodes whose update signature verifies
  against the enrolled node key are aggregated. Anything else is
  excluded and named in the round record.

The round record is scope-marked LOCAL, signed by the server node's
key, and held locally. It is not a hosted connector record.
"""
import json
import secrets
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from ..config import load_config, zk0_home
from ..identity import NodeKey, load_or_create_key
from ..records import (manifest_sha256, runtime_manifest_path,
                       utcnow_iso, verify_record, write_signed_record)
from .common import (arrays_hash, full_model_sha256, get_subset,
                     save_updated_checkpoint, set_subset)

DYNAMIC_WEIGHTING_RULE = (
    "weight_i ∝ n_i * clip(median_loss / loss_i, 0.5, 2.0), "
    "loss_i = the node's reported mean training loss this round")
FEDAVG_RULE = "weight_i ∝ n_i (plain example-count weighting)"


class Registry:
    def __init__(self):
        self.nodes = {}   # node_id -> {public_key_hex, via}
        self.lock = threading.Lock()

    def add(self, node_id, pub_hex, via):
        with self.lock:
            self.nodes[node_id] = {"public_key_hex": pub_hex, "via": via}

    def get(self, node_id):
        with self.lock:
            return self.nodes.get(node_id)

    def ids(self):
        with self.lock:
            return sorted(self.nodes)


def _verify_attestation(att: dict, node_id: str, pub_hex: str):
    """Server-side gate verification: the attestation must be a
    passing, unexpired, signature-valid record from this node — or a
    founder override record (admitted, and marked as such)."""
    v = verify_record(att)
    if not (v["hash_ok"] and v["signature_ok"]):
        return False, f"attestation failed verification: {v}"
    if att.get("signature", {}).get("public_key_hex") != pub_hex:
        return False, "attestation signer != presented node key"
    if att.get("record_type") == "join_gate_founder_override":
        if att.get("node", {}).get("node_id") == node_id:
            return True, "founder_override"
        return False, "founder override record failed verification"
    if att.get("node", {}).get("node_id") != node_id:
        return False, "attestation node_id mismatch"
    if not att.get("pass"):
        return False, "attestation is not a passing gate record"
    if att.get("expires_utc", "") <= utcnow_iso():
        return False, "attestation expired"
    return True, "gate_attestation"


def make_strategy(registry, sink, aggregator, weighting, proximal_mu,
                  steps, lr, min_clients=2):
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedAvg

    class Zk0Strategy(FedAvg):
        def configure_fit(self, server_round, parameters, client_manager):
            out = super().configure_fit(server_round, parameters,
                                        client_manager)
            for _, ins in out:
                ins.config["steps"] = steps
                ins.config["lr"] = lr
                ins.config["proximal_mu"] = (
                    proximal_mu if aggregator == "fedprox" else 0.0)
            return out

        def aggregate_fit(self, server_round, results, failures):
            from flwr.common import FitRes

            accepted, excluded, per_node = [], [], []
            losses = {}
            for cp, res in results:
                m = res.metrics or {}
                nid = m.get("node_id")
                rec = registry.get(nid) if nid else None
                uh, sig = m.get("update_hash"), m.get("update_sig")
                ok = False
                if rec and uh and sig:
                    arrs = parameters_to_ndarrays(res.parameters)
                    ok = (arrays_hash(arrs) == uh and NodeKey.verify(
                        rec["public_key_hex"], bytes.fromhex(uh), sig))
                if ok:
                    accepted.append((cp, res))
                    losses[nid] = float(m.get("train_loss", 0.0))
                else:
                    excluded.append(
                        {"node_id": nid,
                         "reason": "not enrolled, or update signature "
                                   "did not verify against the "
                                   "enrolled node key"})
            med = float(np.median(list(losses.values()))) if losses \
                else 0.0
            weighted = []
            for cp, res in accepted:
                nid = res.metrics["node_id"]
                li = losses[nid]
                if weighting == "dynamic" and li > 0 and med > 0:
                    f = float(np.clip(med / li, 0.5, 2.0))
                else:
                    f = 1.0
                eff = int(round(res.num_examples * f))
                per_node.append({
                    "node_id": nid, "n_examples": res.num_examples,
                    "train_loss": li, "weight_factor": round(f, 4),
                    "effective_weight": eff,
                    "update_hash": res.metrics["update_hash"]})
                weighted.append((cp, FitRes(
                    status=res.status, parameters=res.parameters,
                    num_examples=eff, metrics=res.metrics)))
            sink.setdefault("rounds", []).append({
                "server_round": server_round,
                "median_train_loss": med,
                "accepted_node_ids": [r.metrics.get("node_id")
                                      for _, r in accepted],
                "excluded": excluded,
                "per_node": per_node,
                "failures": len(failures)})
            if not weighted:
                return None, {}
            params, metrics = super().aggregate_fit(
                server_round, weighted, failures)
            if params is not None:
                sink["final_arrays"] = parameters_to_ndarrays(params)
                sink["final_round"] = server_round
            return params, metrics

    return Zk0Strategy(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=min_clients, min_available_clients=min_clients)


class _State:
    def __init__(self):
        self.round_record = None       # dict after finalization
        self.shared_model_path = None  # Path after finalization
        self.model_fetches = 0         # /shared-model downloads served
        self.finalized_at = None       # time.time() at finalization


def _enrollment_server(port, token, network_id, server_key, registry,
                       state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/enroll":
                return self._json(404, {"ok": False})
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                return self._json(400, {"ok": False, "error": str(e)})
            if body.get("token") != token:
                return self._json(403, {"ok": False,
                                        "error": "bad join token"})
            nid = body.get("node_id")
            pub = body.get("public_key_hex")
            att = body.get("attestation") or {}
            ok, via = _verify_attestation(att, nid, pub)
            if not ok:
                return self._json(403, {"ok": False, "error": via})
            registry.add(nid, pub, via)
            cred_msg = f"{nid}|{network_id}|{pub}".encode()
            import hashlib
            cred = {"node_id": nid, "network_id": network_id,
                    "server_public_key_hex": server_key.public_key_hex,
                    "signature": server_key.sign_hex(
                        hashlib.sha256(cred_msg).digest())}
            return self._json(200, {"ok": True, "network_id": network_id,
                                    "credential": cred,
                                    "enrolled_via": via})

        def do_GET(self):
            if self.path == "/status":
                return self._json(200, {
                    "network_id": network_id,
                    "enrolled": registry.ids(),
                    "round_done": state.round_record is not None})
            if self.path == "/round-record":
                if state.round_record is None:
                    return self._json(404, {"ok": False})
                return self._json(200, state.round_record)
            if self.path == "/shared-model":
                if state.shared_model_path is None:
                    return self._json(404, {"ok": False})
                data = Path(state.shared_model_path).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type",
                                 "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
                # Count only COMPLETED downloads: the file is ~1.8 GB,
                # and counting at handler start let the watchdog exit
                # mid-transfer on the first full round.
                state.model_fetches += 1
                return
            return self._json(404, {"ok": False})

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def start_network(port: int = 8797, enroll_port: int = 8798,
                  rounds: int = 1, aggregator: str = "fedprox",
                  weighting: str = "dynamic", proximal_mu: float = 0.1,
                  steps: int = 8, lr: float = 1e-5,
                  min_clients: int = 2, cfg=None) -> dict:
    """Run a local network until its rounds complete; write the
    signed round record; return it."""
    import flwr as fl
    from flwr.common import ndarrays_to_parameters

    from ..sim.policy_eval import checkpoint_dir, load_policy

    cfg = cfg or load_config()
    server_key = load_or_create_key()
    network_id = uuid.uuid4().hex[:12]
    net_dir = zk0_home() / "fl" / "networks" / network_id
    net_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    (net_dir / "join-token.txt").write_text(token + "\n")
    (net_dir / "join-token.txt").chmod(0o600)

    registry = Registry()
    sink: dict = {}
    state = _State()
    _enrollment_server(enroll_port, token, network_id, server_key,
                       registry, state)
    print(f"[zk0 network {network_id}] enrollment on :{enroll_port}, "
          f"flower on :{port}")
    print(f"[zk0 network {network_id}] join token: {token}")
    print(f"[zk0 network {network_id}] aggregator={aggregator} "
          f"weighting={weighting} mu={proximal_mu} rounds={rounds}")

    policy = load_policy().policy
    policy.float()  # fp32 federation (see fl/common.py)
    initial = get_subset(policy)
    before_hash = arrays_hash(initial)
    ckpt = checkpoint_dir()
    before_file_sha = full_model_sha256(ckpt / "model.safetensors")
    strategy = make_strategy(registry, sink, aggregator, weighting,
                             proximal_mu, steps, lr,
                             min_clients=min_clients)
    strategy.initial_parameters = ndarrays_to_parameters(initial)

    finalized = threading.Event()

    def finalize():
        if finalized.is_set() or "final_arrays" not in sink:
            return
        finalized.set()
        set_subset(policy, sink["final_arrays"])
        out_dir = net_dir / "shared_model_after"
        save_updated_checkpoint(policy, ckpt, out_dir)
        after_file_sha = full_model_sha256(out_dir / "model.safetensors")
        record = {
            "record_type": "fl_round_record",
            "schema": "zk0-node-fl/0.1",
            "created_utc": utcnow_iso(),
            "scope": "local",
            "network_id": network_id,
            "node": {"node_id": cfg["node_id"],
                     "runtime_manifest_sha256": manifest_sha256(
                         runtime_manifest_path())},
            "aggregator": {
                "name": aggregator,
                "weighting": weighting,
                "weighting_rule": DYNAMIC_WEIGHTING_RULE
                if weighting == "dynamic" else FEDAVG_RULE,
                "proximal_mu": proximal_mu
                if aggregator == "fedprox" else 0.0,
                "note": "FedProx's proximal term is applied "
                        "client-side from the round config, as the "
                        "algorithm defines; the server passes mu and "
                        "applies the weighting rule in aggregation.",
            },
            "rounds": rounds,
            "participants": sink.get("rounds", []),
            "enrolled_node_ids": registry.ids(),
            "before": {"action_expert_subset_sha256": before_hash,
                       "model_safetensors_sha256": before_file_sha},
            "after": {"action_expert_subset_sha256": arrays_hash(
                          sink["final_arrays"]),
                      "model_safetensors_sha256": after_file_sha},
            "shared_model_after_dir": str(out_dir),
            "note": "LOCAL round on the operator's own network "
                    "(spec 2.3) — held locally, not a hosted "
                    "connector record.",
        }
        out = zk0_home() / "records" / "fl-rounds" / \
            f"round-{network_id}.json"
        write_signed_record(record, out, server_key)
        record["path"] = str(out)
        state.round_record = record
        state.shared_model_path = out_dir / "model.safetensors"
        state.finalized_at = time.time()
        print(f"[zk0 network {network_id}] round record: {out}")

    def watchdog():
        # Legacy start_server returns when rounds complete; if the
        # transport lingers, finalize from the sink. The process must
        # NOT exit before the clients have fetched the shared model
        # from the enrollment service (an earlier version exited ~12 s
        # after finalizing and both clients' fetches were refused) —
        # exit once every enrolled node has downloaded the model, or
        # after a 10-minute grace period.
        import os
        while True:
            if "final_arrays" in sink and not finalized.is_set():
                time.sleep(5)
                finalize()
            if finalized.is_set() and state.finalized_at:
                expected = max(1, len(registry.ids()))
                served = state.model_fetches >= expected
                expired = time.time() - state.finalized_at > 600
                if served or expired:
                    time.sleep(10)
                    os._exit(0)
            time.sleep(2)

    threading.Thread(target=watchdog, daemon=True).start()
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy, grpc_max_message_length=1 << 30)
    finalize()
    if state.round_record is None:
        raise RuntimeError("rounds completed without an aggregated "
                           "result — see the strategy sink")
    # The clients fetch the round record + shared model from this
    # process's enrollment service AFTER their Flower connection
    # closes. Returning here would exit the process and refuse
    # those fetches (this exact failure killed two demonstration
    # rounds) — so wait until every enrolled node has downloaded
    # the shared model, with a bounded grace period.
    expected = max(1, len(registry.ids()))
    deadline = time.time() + 600
    while state.model_fetches < expected and time.time() < deadline:
        time.sleep(2)
    print(f"[zk0 network {network_id}] shared model fetched by "
          f"{state.model_fetches}/{expected} enrolled nodes; "
          f"server exiting")
    return state.round_record
