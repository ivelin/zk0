"""The hard safety layer's configuration and command clamping (spec §4.1).

The layer sits below the learning: it inspects every command between
any policy and the arm, and no weight file, config flag, or operator
setting can disable it or loosen it past the certified envelope; an
operator may only tighten it. Build slice 1 carries the layer's
configuration and its command clamp (the S6b lesson, SSOT decision
27: an out-of-range close command held at stall produced ~47x
object-weight clamp force — out-of-range commands are clamped and
counted, and sustained saturation aborts the run). The arm-side
enforcement lands with the arm bridge (build slice 2).
"""

# Defaults are the certified envelope, not inventions: the saturation
# fraction is the grasp gate's SAT_FRAC_MAX; the damage bound is the
# preregistration's PROVISIONAL ceramic-chip low end (§2).
DEFAULT_LIMITS = {
    "gripper_saturation_abort_frac": 0.50,
    "damage_bound_N": 25.0,
    "damage_bound_status": "PROVISIONAL",
    # Declared workspace box around the node's work area (metres).
    # Must contain the benchmark scenes; doctor verifies (§1.4 #10).
    "workspace_box_m": {"x": [0.05, 0.45], "y": [-0.35, 0.35], "z": [0.0, 0.45]},
}


def clamp_command(value, joint_range):
    """Clamp a command into the joint's own range.

    Returns (applied, was_clamped). This is the same rule the
    certified pilot applies (Pilot._clamp_grip); it lives here so
    every command path in the package shares one implementation.
    """
    lo, hi = float(joint_range[0]), float(joint_range[1])
    applied = min(max(float(value), lo), hi)
    return applied, applied != float(value)


def workspace_contains(limits, point) -> bool:
    box = limits["workspace_box_m"]
    return all(box[axis][0] <= float(point[i]) <= box[axis][1]
               for i, axis in enumerate(("x", "y", "z")))
