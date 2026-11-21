"""Unit check for adaptation #6 (D-11/G-3): the substrate's
dynamic_weighted_average must reproduce the study driver's
aggregate_round output on identical synthetic inputs (same arrays,
same losses, same num_examples) to fp32 rounding, including the
per-client factors and effective weights. The driver path exercises
its real hash + Ed25519 signature verification with throwaway keys.
"""

import sys

import numpy as np

sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-20261119/run/scripts")

import fl_driver
from zk0_node.fl.common import arrays_hash
from zk0_node.identity import NodeKey

sys.path.insert(0, "/home/openclaw/muse-work/zk0-study001-fl-20261121")
from src.server.strategy import dynamic_weighted_average

rng = np.random.default_rng(7)


def run_case(name, losses, num_examples):
    n_clients = len(losses)
    keys = [NodeKey(bytes([i + 1]) * 32) for i in range(n_clients)]
    enrolled = {f"node_{i}": keys[i].public_key_hex for i in range(n_clients)}
    driver_results = []
    substrate_entries = []
    for i in range(n_clients):
        arrays = [
            rng.normal(0, 0.05, size=(6, 4)).astype(np.float32),
            rng.normal(0.5, 0.05, size=(5,)).astype(np.float32),
            rng.normal(-0.2, 0.05, size=(3, 3, 2)).astype(np.float32),
        ]
        uh = arrays_hash(arrays)
        driver_results.append(
            (
                arrays,
                num_examples[i],
                {
                    "node_id": f"node_{i}",
                    "train_loss": losses[i],
                    "update_hash": uh,
                    "update_sig": keys[i].sign_hex(bytes.fromhex(uh)),
                },
            )
        )
        substrate_entries.append((arrays, num_examples[i], losses[i]))
    drv_global, drv_record = fl_driver.aggregate_round(
        driver_results, enrolled, dynamic=True
    )
    sub_global, sub_record = dynamic_weighted_average(substrate_entries)
    assert len(drv_global) == len(sub_global)
    max_diff = 0.0
    for a, b in zip(drv_global, sub_global):
        assert a.shape == b.shape
        max_diff = max(max_diff, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    assert max_diff < 1e-5, f"{name}: max abs diff {max_diff}"
    drv_eff = {p["node_id"]: p["effective_weight"] for p in drv_record["per_node"]}
    drv_fac = {p["node_id"]: p["weight_factor"] for p in drv_record["per_node"]}
    for row in sub_record["per_client"]:
        nid = f"node_{row['client_index']}"
        assert row["effective_weight"] == drv_eff[nid], (name, nid, row, drv_eff)
        assert abs(row["weight_factor"] - drv_fac[nid]) < 1e-9, (name, nid)
    assert sub_record["total_effective_weight"] == sum(drv_eff.values())
    assert abs(sub_record["median_policy_loss"] - drv_record["median_train_loss"]) < 1e-12
    print(f"CASE {name}: PASS (max abs diff {max_diff:.2e})")


run_case("typical", [0.90, 1.10, 0.70, 1.30, 1.00], [12975, 9127, 11679, 20806, 27647])
run_case("clip-bounds", [0.10, 1.00, 1.00, 1.00, 5.00], [405, 285, 364, 650, 864])
run_case("zero-loss", [0.0, 1.00, 0.80, 1.20, 0.90], [405, 285, 364, 650, 864])
run_case("all-equal", [1.0, 1.0, 1.0, 1.0, 1.0], [100, 200, 300, 400, 500])
print("DYNAMIC WEIGHTING UNIT CHECK PASS")
