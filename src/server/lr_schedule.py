"""Registered LR schedule for Study 001 (deviation D-11, gap G-4).

Transcribed verbatim from the study driver
(~/muse-work/zk0-study001-20261119/run/scripts/fl_driver.py, lr_at):
warmup 100 steps to 1e-4, then cosine decay to 2.5e-6 at the node's
3000-step budget (30 rounds x 100 local steps).
"""

import math

TOTAL_LOCAL_STEPS = 3000  # per node: 30 rounds x 100 steps


def lr_at(s: int) -> float:
    """Registered schedule at local-step offset s (server-side, per
    round): warmup 100 steps to 1e-4, then cosine decay to 2.5e-6 at
    the node's 3000-step budget."""
    if s < 100:
        return 1e-4 * s / 100
    return 2.5e-6 + 0.5 * (1e-4 - 2.5e-6) * (
        1 + math.cos(math.pi * (s - 100) / (TOTAL_LOCAL_STEPS - 100)))
