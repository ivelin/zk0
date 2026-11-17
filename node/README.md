# zk0 Node

The operator distribution package for the [zk0](https://github.com/ivelin/zk0)
federated-learning network: everything an operator needs to run a node —
**join, calibrate, train, evaluate, contribute** — as one install with
three mode configurations (sim / staging / prod) plus a local-network
scope, instead of a repository to assemble by hand.

Design basis: the zk0 Node v0.1 design spec (maintained privately by the
maintainer; this package implements it).

## Status — build slice 1 (v0.1.0-dev)

Built and exercised on the dev host:

- **Launcher + modes** — `bin/zk0` runs the package on its own pinned
  runtime. Modes are configurations of one install (`zk0 mode …`);
  switching down is always allowed, switching up runs the tier's gates.
  **Prod enrollment for external operators is hard-blocked** — the
  client refuses, it does not warn — until the contributor terms are
  live and the operator's own human-click acceptance record exists.
  A local-network scope (§2.3) can be joined alongside any mode.
- **`zk0 doctor`** — the 12 self-checks: runtime integrity, dependency
  consistency against the pinned manifest, compute profile, arm bus /
  port / camera / calibration / servo checks (degrading honestly to
  "not present" with no arm attached), sim stack, safety layer,
  network, storage. Output is also written as a diagnostic bundle.
- **Sim mode end-to-end** — `zk0 sim run` executes the SIM-1 benchmark
  task (cube → box) on the certified simulation harness and writes a
  state log + eval record (Tier 1 judged from the state log; Tier 2
  PROVISIONAL). The slice-1 policy is the scripted reference pilot
  (ORACLE-GUIDED, rung 0) — not the shared checkpoint.

Not yet built (later slices): local FL rounds and staging wiring, the
train pack (LeRobot 0.7.0 + PyAV 15.x), the arm bridge, the local
plugin + MCP surface, node-key record signing.

## Quick start (dev host)

```bash
dev/bootstrap.sh        # runtime venv (Python 3.12 + pins) + staged assets
bin/zk0 doctor
bin/zk0 status
bin/zk0 sim run                       # SIM-1 on so101-realistic
bin/zk0 sim run --arm-model stock     # the certified T1 control config
bin/zk0 mode prod                     # → REFUSED (hard block), by design
```

## Pins

Every runtime pin and its reason lives in
[`runtime-manifest.json`](runtime-manifest.json) — including the hard
pin **PyAV 15.x** (PyAV 19 breaks LeRobot 0.7.0 dataset capture) and
**LeRobot 0.7.0**. The manifest's SHA-256 is embedded in every eval
record the package produces.

## Layout

```
bin/zk0                  launcher (spec §1.1)
runtime-manifest.json    every pin + reason (§1.3)
src/zk0_node/            zk0-core (§1.1): config, doctor, safety, records
src/zk0_node/sim/        sim mode; harness/ = vendored certified code
                         (see harness/PROVENANCE.md for sources + patches)
dev/bootstrap.sh         dev-host runtime + asset staging
tests/                   stdlib-unittest mode/gate/clamp tests
runtime/                 installed runtime (gitignored, never source)
```

Node state (config, staged assets, records, diagnostics) lives under
`~/.zk0-node/` (override with `ZK0_HOME`).
