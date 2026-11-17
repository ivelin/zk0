"""Node configuration: modes, network scopes, and the prod gate.

Implements spec §2: the three modes are configurations of one install,
never separate builds; switching down is always allowed; switching up
requires the target tier's gates. Prod enrollment for external
operators is HARD-BLOCKED — the client refuses, it does not warn —
until (a) the contributor terms are live (a server-side flag) and
(b) the operator's own terms-acceptance record exists (acceptance is
human-click-only, server rule SM-5B). Founder nodes proceed under the
founder's own acceptance record (spec §2.1 / §8 Q3).

Every install also carries a local-network scope (spec §2.3):
membership in an operator-run LAN federation is held alongside
staging/prod membership, never instead of the mode gates.
"""
import json
import os
import uuid
from pathlib import Path

MODES = ("sim", "staging", "prod")
_MODE_ORDER = {"sim": 0, "staging": 1, "prod": 2}
NETWORK_SCOPES = ("local", "staging", "prod")
IDENTITY_KINDS = ("external", "founder")


class ProdEnrollmentBlocked(Exception):
    """Prod enrollment refused. Carries the named reasons (spec §2.1)."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("prod enrollment refused: " + "; ".join(self.reasons))


def zk0_home() -> Path:
    """The node's local state directory (config, assets, records)."""
    return Path(os.environ.get("ZK0_HOME", Path.home() / ".zk0-node"))


def config_path() -> Path:
    return zk0_home() / "config.json"


def default_config() -> dict:
    return {
        "node_id": str(uuid.uuid4()),
        "identity_kind": "external",
        "mode": "sim",
        "network_scopes": [],
        # Stage 2 calibration state (canonical shape:
        # calibration.default_calibration). Must be a known key —
        # load_config drops unknown keys by design.
        "calibration": {"generation": 0, "offsets_deg": [0.0] * 6,
                        "action_norm": {"type": "identity"},
                        "fingerprint": None},
        "terms": {
            # `live` mirrors the server-side flag that removes the
            # external block when the contributor terms ship (Q3);
            # `acceptance_record` is the id of the human-click
            # acceptance (SM-5B) — the agent cannot create it.
            "live": False,
            "acceptance_record": None,
        },
        "endpoints": {
            "prod_connector": "https://mcp.zk0.pirin.ai",
            "staging": "DGX Spark dev federation (zk0bot account)",
            "local": None,
        },
    }


def load_config() -> dict:
    p = config_path()
    if p.exists():
        cfg = json.loads(p.read_text())
        base = default_config()
        base.update({k: cfg[k] for k in cfg if k in base})
        base["terms"] = {**default_config()["terms"], **cfg.get("terms", {})}
        base["endpoints"] = {**default_config()["endpoints"],
                             **cfg.get("endpoints", {})}
        base["calibration"] = {**default_config()["calibration"],
                               **(cfg.get("calibration") or {})}
        return base
    return default_config()


def save_config(cfg: dict) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=1, sort_keys=True))
    return p


def prod_block_reasons(cfg: dict) -> list:
    """Why prod enrollment is currently refused for this node ([] = allowed)."""
    reasons = []
    terms = cfg.get("terms", {})
    if cfg.get("identity_kind") not in IDENTITY_KINDS:
        reasons.append(f"unknown identity_kind {cfg.get('identity_kind')!r}")
    if cfg.get("identity_kind") != "founder" and not terms.get("live"):
        reasons.append(
            "contributor terms are not live — external prod enrollment is "
            "hard-blocked in the client until the terms ship (spec §2.1/Q3); "
            "the block is lifted by the server-side terms flag, not by a "
            "client update")
    if not terms.get("acceptance_record"):
        reasons.append(
            "no terms-acceptance record for this node — acceptance is a "
            "human click (server rule SM-5B); the agent cannot accept on "
            "the operator's behalf (spec §3, Stage 1)")
    return reasons


def enroll_prod(cfg: dict) -> dict:
    """Attempt prod enrollment. Refuses (raises) or returns the record."""
    reasons = prod_block_reasons(cfg)
    if reasons:
        raise ProdEnrollmentBlocked(reasons)
    return {"node_id": cfg["node_id"], "identity_kind": cfg["identity_kind"],
            "acceptance_record": cfg["terms"]["acceptance_record"],
            "scope": "prod"}


def set_mode(cfg: dict, target: str) -> list:
    """Switch mode per §2.2. Returns human-readable notes.

    Down-switches always succeed immediately. Up-switches run the
    target tier's gates: prod goes through enroll_prod (hard block);
    staging notes its doctor subset. A mode switch never moves raw
    data (§2.2 rule 4) — this function only edits configuration.
    """
    if target not in MODES:
        raise ValueError(f"unknown mode {target!r} (modes: {', '.join(MODES)})")
    current = cfg.get("mode", "sim")
    notes = []
    if _MODE_ORDER[target] < _MODE_ORDER[current]:
        notes.append(f"mode {current} → {target}: down-switch, always allowed (§2.2)")
    elif _MODE_ORDER[target] > _MODE_ORDER[current]:
        if target == "prod":
            enroll_prod(cfg)  # raises ProdEnrollmentBlocked when gated
            notes.append("prod gates passed (identity + terms, Stage 1)")
        if target == "staging":
            notes.append("staging: dev-federation membership is configuration; "
                         "the staging doctor subset (checks 3, 11) applies")
    cfg["mode"] = target
    if target in ("staging", "prod") and target not in cfg["network_scopes"]:
        cfg["network_scopes"].append(target)
    return notes


def set_local_scope(cfg: dict, enabled: bool) -> list:
    """Join/leave the local-network scope (§2.3) — orthogonal to mode."""
    scopes = cfg.setdefault("network_scopes", [])
    if enabled and "local" not in scopes:
        scopes.append("local")
        return ["local-network scope joined: this node can enroll in an "
                "operator-run LAN federation (zk0.fl_join) independent of "
                "its mode (§2.3)"]
    if not enabled and "local" in scopes:
        scopes.remove("local")
        return ["local-network scope left"]
    return ["local-network scope unchanged"]
