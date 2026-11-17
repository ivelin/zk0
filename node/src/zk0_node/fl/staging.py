"""The staging aggregator deployment (spec section 2.2; build
slice 3).

Staging is a DEPLOYMENT of the same aggregator code (fl/network.py)
under its own service home — its own node key, its own records —
run with scope="staging" and require_mode="staging": it enrolls
only nodes that are themselves in staging mode, and every record
it issues is scope-marked STAGING.

Production deployment (week plan): the service runs on the Spark
under the `zk0bot` account, whose home is the service home. That
account's SSH-key authorization is STILL OUTSTANDING (a
founder/admin action), so as of this slice the service has been
run only as a REHEARSAL under the build account (its own service
home directory, the same code path) — the rehearsal proves the
wiring; it does not claim the zk0bot deployment.

Ports differ from the local network's (8797/8798) so a staging
service and a local network can coexist on one host: staging uses
8897 (Flower) / 8898 (enrollment).
"""
import os
from pathlib import Path

STAGING_FLOWER_PORT = 8897
STAGING_ENROLL_PORT = 8898


def staging_service_home() -> Path:
    """The staging service's home (its ZK0_HOME when it runs).
    ZK0_STAGING_HOME overrides; default ~/.zk0-node-staging-service.
    In the production deployment this is the zk0bot account's home
    directory on the Spark."""
    return Path(os.environ.get(
        "ZK0_STAGING_HOME", "~/.zk0-node-staging-service")
    ).expanduser()


def run_staging_aggregator(rounds: int = 1, min_clients: int = 2,
                           **kw) -> dict:
    """Run the staging aggregator in THIS process. The caller (the
    `zk0 fl start-staging` CLI) has already verified that this
    process's ZK0_HOME is the staging service home — the service's
    key and records must never land in a node's home."""
    from .network import start_network
    return start_network(port=STAGING_FLOWER_PORT,
                         enroll_port=STAGING_ENROLL_PORT,
                         rounds=rounds, min_clients=min_clients,
                         scope="staging", require_mode="staging",
                         **kw)
