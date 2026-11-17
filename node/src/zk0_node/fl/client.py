"""A zk0 node as a Flower client (spec sections 2.3, 5.3).

Trains ONLY the action-expert subset on the node's local dataset;
the SmolVLM backbone never leaves the frozen state and never
receives gradient updates here. Each fit result is signed: the
client signs the hash of its updated subset with its node key, and
the server aggregates only updates whose signature verifies against
the enrolled key (see network.py) — an unenrolled or unsigned
update is excluded and the exclusion is recorded.

FedProx: when the server's round config carries proximal_mu > 0,
the local objective gains the proximal term
mu/2 * ||w - w_global||^2 over the subset — the standard FedProx
penalty, applied client-side exactly as the algorithm defines.
"""
import json
import urllib.request
from pathlib import Path

import numpy as np

from ..config import load_config, save_config, zk0_home
from ..identity import load_or_create_key
from ..records import utcnow_iso, write_signed_record
from ..sim.policy_eval import load_policy
from .common import (arrays_hash, full_model_sha256, get_subset,
                     save_updated_checkpoint, set_subset, subset_keys)


def _make_trainable(policy):
    """Freeze everything except the action-expert subset; return the
    trainable parameters in subset order."""
    sd_keys = set(subset_keys(policy.state_dict()))
    trainable = {}
    for name, p in policy.named_parameters():
        p.requires_grad_(name in sd_keys)
        if name in sd_keys:
            trainable[name] = p
    return [trainable[k] for k in sorted(trainable)]


def _dataset_batches(root, batch_size=4, chunk=50, fps=30, seed=0):
    """An endless batch iterator over a local LeRobot dataset with
    action chunks (the training shape SmolVLA consumes)."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id="local", root=root,
        delta_timestamps={"action": [i / fps for i in range(chunk)]})
    n = len(ds)

    def gen():
        order = np.random.default_rng(seed).permutation(n)
        pos = 0
        while True:
            idx = []
            for _ in range(batch_size):
                idx.append(int(order[pos % n]))
                pos += 1
            items = [ds[i] for i in idx]
            batch = {}
            for k in items[0]:
                if k == "task":
                    batch[k] = [it[k] for it in items]
                elif hasattr(items[0][k], "shape"):
                    batch[k] = torch.stack([it[k] for it in items])
            yield batch

    return gen(), n


def _policy_loss(policy, pre, batch):
    out = policy.forward(pre(batch))
    return out[0] if isinstance(out, tuple) else out


class Zk0NumPyClient:
    """Duck-typed flwr NumPyClient (subclassed at runtime to avoid a
    hard import at module load)."""

    def __init__(self, policy, pre, dataset_root, node_id, key,
                 steps=8, lr=1e-5, batch_size=4):
        import torch

        self.policy = policy
        self.pre = pre
        self.node_id = node_id
        self.key = key
        self.steps = steps
        self.lr = lr
        self.trainable = _make_trainable(policy)
        self.opt = torch.optim.AdamW(self.trainable, lr=lr)
        self.batches, self.n_examples = _dataset_batches(
            dataset_root, batch_size=batch_size)
        self.received_initial_hash = None

    # -- Flower NumPyClient interface ----------------------------------
    def get_parameters(self, config):
        return get_subset(self.policy)

    def fit(self, parameters, config):
        import torch

        set_subset(self.policy, parameters)
        if self.received_initial_hash is None:
            self.received_initial_hash = arrays_hash(parameters)
        mu = float(config.get("proximal_mu", 0.0))
        steps = int(config.get("steps", self.steps))
        global_t = [torch.from_numpy(np.asarray(a, dtype=np.float32)).to(
            next(self.policy.parameters()).device)
            for a in parameters]
        # Map global arrays onto trainable params by subset order.
        sd = self.policy.state_dict()
        keys = subset_keys(sd)
        by_key = dict(zip(keys, global_t))
        prox_refs = [by_key[k] for k in sorted(
            k for k in sd if not k.startswith("model.vlm_with_expert.vlm."))]
        losses = []
        self.policy.train()
        for _ in range(steps):
            batch = next(self.batches)
            self.opt.zero_grad()
            loss = _policy_loss(self.policy, self.pre, batch)
            if mu > 0:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(self.trainable, prox_refs))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            self.opt.step()
            losses.append(float(loss.detach().cpu()))
        updated = get_subset(self.policy)
        uh = arrays_hash(updated)
        return updated, self.n_examples, {
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "node_id": self.node_id,
            "update_hash": uh,
            "update_sig": self.key.sign_hex(bytes.fromhex(uh)),
        }

    def evaluate(self, parameters, config):
        set_subset(self.policy, parameters)
        self.policy.eval()
        losses = []
        for _ in range(2):
            batch = next(self.batches)
            with __import__("torch").no_grad():
                losses.append(float(_policy_loss(
                    self.policy, self.pre, batch).detach().cpu()))
        return float(np.mean(losses)), self.n_examples, {}


def make_client_class():
    from flwr.client import NumPyClient

    class _Client(Zk0NumPyClient, NumPyClient):
        pass

    return _Client


def enroll(enroll_url: str, token: str, cfg: dict, attestation: dict,
           key) -> dict:
    """POST the node's enrollment: join token + signed gate
    attestation + node public key. Returns the server credential."""
    body = json.dumps({
        "node_id": cfg["node_id"], "token": token,
        "public_key_hex": key.public_key_hex,
        "attestation": attestation}).encode()
    req = urllib.request.Request(
        enroll_url.rstrip("/") + "/enroll", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _get(enroll_url: str, path: str, binary=False):
    with urllib.request.urlopen(enroll_url.rstrip("/") + path,
                                timeout=120) as r:
        data = r.read()
    return data if binary else json.loads(data)


def _get_when_ready(enroll_url: str, path: str, binary=False,
                    timeout_s: int = 180):
    """GET with retries: the server finalizes the round record a
    few seconds after aggregation completes, so a client asking
    immediately can legitimately get a 404 first."""
    import time
    import urllib.error

    deadline = time.time() + timeout_s
    while True:
        try:
            return _get(enroll_url, path, binary=binary)
        except urllib.error.HTTPError as e:
            if e.code != 404 or time.time() > deadline:
                raise
            time.sleep(3)
        except (urllib.error.URLError, ConnectionError):
            # Server still finalizing, not yet listening, or a large
            # download was cut mid-transfer — retry from scratch.
            if time.time() > deadline:
                raise
            time.sleep(3)


def run_client(server_address: str, enroll_url: str, token: str,
               dataset_root, cfg=None, steps: int = 8,
               lr: float = 1e-5) -> dict:
    """Join a local network and participate in its rounds.

    Precondition (enforced by the CLI before this is called): the M0
    gate — a current attestation, or a recorded founder override.
    Enrollment itself re-verifies the attestation server-side.
    """
    import flwr as fl

    cfg = cfg or load_config()
    key = load_or_create_key()
    from .. import joingate
    att = joingate.current_attestation(cfg)
    override = None
    if att is None:
        # A recorded founder override is the only other way in.
        d = joingate.attestations_dir()
        overs = sorted(d.glob("override-*.json")) if d.exists() else []
        if not overs:
            raise joingate.GateRefused(
                ["no attestation and no founder override on file"])
        override = json.loads(overs[-1].read_text())
        att = {"record_type": "join_gate_founder_override",
               **override}
    cred = enroll(enroll_url, token, cfg, att, key)
    cfg.setdefault("network_scopes", [])
    if "local" not in cfg["network_scopes"]:
        cfg["network_scopes"].append("local")
    cfg.setdefault("endpoints", {})["local"] = (
        f"{server_address} (enroll {enroll_url})")
    save_config(cfg)

    bundle = load_policy()
    # Federated training runs in fp32 (see fl/common.py): the expert
    # loads partly as bfloat16, which would make AdamW updates and
    # the averaged result needlessly coarse.
    bundle.policy.float()
    client = make_client_class()(bundle.policy, bundle.pre, dataset_root,
                                 cfg["node_id"], key, steps=steps, lr=lr)
    try:
        fl.client.start_client(
            server_address=server_address, client=client,
            grpc_max_message_length=1 << 30)
    except TypeError:
        fl.client.start_client(server_address=server_address,
                               client=client)

    # The round is done: fetch the shared model + round record from
    # the operator's own server (the local network's coordination
    # channel) and verify the model against the record.
    net_id = cred["network_id"]
    record = _get_when_ready(enroll_url, "/round-record")
    model_bytes = _get_when_ready(enroll_url, "/shared-model", binary=True)
    recv_dir = zk0_home() / "fl" / "received" / net_id
    recv_dir.mkdir(parents=True, exist_ok=True)
    (recv_dir / "model.safetensors").write_bytes(model_bytes)
    from ..sim.policy_eval import checkpoint_dir
    import shutil
    for f in checkpoint_dir().iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy2(f, recv_dir / f.name)
    got_sha = full_model_sha256(recv_dir / "model.safetensors")
    expect_sha = record["after"]["model_safetensors_sha256"]
    if got_sha != expect_sha:
        raise RuntimeError(
            f"received shared model hash {got_sha} != round record "
            f"{expect_sha}")
    part = {
        "record_type": "fl_participation_record",
        "schema": "zk0-node-fl/0.1",
        "created_utc": utcnow_iso(),
        "scope": "local",
        "network_id": net_id,
        "node": {"node_id": cfg["node_id"]},
        "enrollment": "gate attestation" if not override
                      else "FOUNDER OVERRIDE (see override record)",
        "dataset_root": str(dataset_root),
        "before": {"action_expert_subset_sha256":
                   client.received_initial_hash},
        "after": {"action_expert_subset_sha256":
                  record["after"]["action_expert_subset_sha256"],
                  "model_safetensors_sha256": got_sha},
        "round_record_content_sha256": record.get("content_sha256"),
        "received_model_dir": str(recv_dir),
        "note": "LOCAL round on the operator's own network (spec "
                "2.3). This record is held locally; it is not a "
                "hosted connector record and claims no hosted round.",
    }
    out = zk0_home() / "records" / "fl-rounds" / f"participation-{net_id}.json"
    write_signed_record(part, out, key)
    part["path"] = str(out)
    part["round_record"] = record
    return part
