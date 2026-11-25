"""Study 001 twin-replay: global parameters (Builder C, Phase 4).

Every value traces to docs/physics-parameters.md (Builder A) unless marked
LOCAL-CHOICE (a Phase-4 modelling choice, logged in the validation report).
World frame: origin at arm base centre, x forward, y left, z up, tabletop z=0.
"""
from pathlib import Path

TWIN = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
RECORDS = Path("/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122/records")
SPLIT = Path("/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122/split/split.json")
ARM_XML = "/home/openclaw/muse-work/zk0-sim-v01-20261115/sim/model/so101_new_calib.xml"
ARM_MESHDIR = "assets"  # scenes/assets is a symlink to the vendored model assets (meshdir resolves relative to the scene file)
ARM_ASSETS_TARGET = "/home/openclaw/muse-work/zk0-sim-v01-20261115/sim/model/assets"

# FPS RESOLUTION (v1.1, F-05): the pre-registration text says "30 fps", but
# the registered rate is the DATASET's own rate. Resolved from the dataset's
# own metadata: /home/openclaw/muse-work/zk0-onarm-20261117/data/
# armnetbench_v01_lerobot_so101/meta/info.json field "fps" = 20 (verified
# 2026-11-22 by Builder 1). v1 ran at 20 fps (correct value, unproven
# provenance); v1.1 runs at 20 fps, read from that metadata file at import
# so the rate can never silently diverge from the dataset again.
import json as _json
DATASET_META = Path("/home/openclaw/muse-work/zk0-onarm-20261117/data/armnetbench_v01_lerobot_so101/meta/info.json")
try:
    FPS = float(_json.loads(DATASET_META.read_text())["fps"])
except Exception:
    FPS = 20.0        # fallback = the verified metadata value above
TIMESTEP = 0.0005     # 2000 Hz physics (v1.1 global fix, changelog #14:
                      # the cable ball-joint chain + stiff weld solref is
                      # integrator-unstable at v1's 500 Hz; 1000 Hz fixed
                      # only 1 of 2 exploded test episodes, 2000 Hz fixed
                      # both. Applied globally to all tasks/variants.)
# SETTLE_S (the former fixed 1.0 s settle wait) was REMOVED 2026-11-25
# (founder ruling, criteria A+B change): _finish now settles until the
# physics engine reports rest (replay._bodies_at_rest -- every dynamic
# object body's speed below |g| * dt read from the model), so no fixed
# wait gates a score. See criteria-provenance.json code.settle_rest.

# --- grasp device (docs/physics-parameters.md section 7) ---
# GRASP_DIST_M (the former 3.0 cm proximity gate) was REMOVED 2026-11-25
# (founder ruling, criteria A+B change): the attach gate now reads
# pad-object CONTACT (replay.contact_dist -- the gripper site reaches
# the object's material), not a hand-made proximity band. See
# criteria-provenance.json code.grasp_contact.
ATTACH_CLOSEDNESS = 0.47      # attach threshold (normalised closedness)
RELEASE_CLOSEDNESS = 0.35     # release threshold (hysteresis)
# Gripper-trace normalisation (CONCRETE, global algorithm; see report):
#   The recorded gripper channel is in degrees and is INVERTED vs the naive
#   reading (calibration finding, changelog #4): LOW values = jaws closed
#   (holding), HIGH values = jaws open; replay.closedness_trace implements
#   closedness = clip((hi - v)/(hi - lo), 0, 1) on the p5/p99 refs below.
#   The peak excursion is object-dependent (carton ~40 deg,
#   cube ~26-44, ring ~15-28, tool ~11-22) because the jaw stops on the
#   object. An absolute 50%-of-range threshold can therefore NEVER fire for
#   ring/tool episodes. The normalisation is one global algorithm applied
#   identically to every episode (no per-episode hand tuning):
#     closed_ref = 5th percentile of the episode gripper channel
#     open_ref   = 99th percentile of the episode gripper channel
#     closedness = clip((open_ref - g) / (open_ref - closed_ref), 0, 1)
#   If (open_ref - closed_ref) < 4.4 deg the episode has no grasp excursion
#   and no attach is possible.
GRIP_MIN_EXCURSION_DEG = 4.4

# --- common workspace ---
MAT_SIZE = (0.55, 0.40, 0.003)        # x, y, thickness; top surface at z=0
MAT_CENTER = (0.30, 0.0)
MAT_FRICTION = (0.9, 0.005, 0.0001)
FLOOR_FRICTION = (0.8, 0.005, 0.0001)
WALL_FRICTION = (0.5, 0.005, 0.0001)

# --- per-task geometry / physics (physics-parameters.md) ---
CUBE_EDGE = 0.030
CUBE_MASS = 0.020
CUBE_FRICTION = (0.6, 0.005, 0.0001)

CARTON_SIZE = (0.090, 0.035, 0.035)   # x,y,z full sizes
CARTON_MASS = 0.030
CARTON_FRICTION = (0.5, 0.005, 0.0001)
PLATFORM_TOP = (0.200, 0.160, 0.004)  # plate sizes
PLATFORM_HEIGHT = 0.090               # top surface height above mat
PLATFORM_FRICTION = (0.4, 0.005, 0.0001)
BASKET_OUTER = (0.260, 0.180)
BASKET_WALL_H = 0.070
BASKET_WALL_T = 0.003
BASKET_FRICTION = (0.4, 0.005, 0.0001)

BOARD_R = 0.085
BOARD_H = 0.015
BOARD_FRICTION = (0.5, 0.005, 0.0001)
PEG_R = 0.008
PEG_H = 0.070                          # above board top
PEG_FRICTION = (0.4, 0.005, 0.0001)
RING_R_OUT = 0.0225
RING_R_IN = 0.011
RING_THICK = 0.0115   # built torus tube diameter (11.5 mm): the build code
                      # caps the tube radius at (RING_R_OUT - RING_R_IN)/2
                      # = 5.75 mm; adopted as the criterion parameter for
                      # the modelled tube by founder ruling 2026-11-24
                      # 06:43 CT (ring-peg-evidence-20261124.md section 6)
                      # -- NOT a measurement of the real ring (a flat
                      # washer, axial ~12.0 mm measured); 0.012 was that
                      # axial thickness in the wrong role --
                      # criteria-provenance.json param.RING_THICK)
RING_MASS = 0.010
RING_FRICTION = (0.5, 0.005, 0.0001)
RING_SEGMENTS = 12                     # capsule ring approximation

TRAY_SIZE = (0.385, 0.270, 0.045)
TRAY_FRICTION = (0.6, 0.005, 0.0001)
SLOT_SIZE = (0.081, 0.029)             # target slot footprint (x,y), MEASURED
                                       # 2026-11-24 (P3 second opinion): channel
                                       # length 81 mm, width plateau 29 mm (waist
                                       # ~23 mm is a local narrowing the
                                       # rectangular model cannot express)
SLOT_DEPTH = 0.015                     # slot recess depth, MEASURED working
                                       # figure 2026-11-24 (bound >= ~15 mm at
                                       # the socket section; the former 5 mm
                                       # was an estimate, >=3x too shallow)
TOOL_LEN = 0.100
TOOL_R = 0.005
TOOL_MASS = 0.070
TOOL_FRICTION = (0.4, 0.005, 0.0001)

BLOCK_SIZE = (0.110, 0.070, 0.055)
BLOCK_FRICTION = (0.6, 0.005, 0.0001)
CLIP_CHANNEL_W = 0.008
CLIP_FRICTION = (0.5, 0.005, 0.0001)
CABLE_N = 24
CABLE_SEG_LEN = 0.030
CABLE_R = 0.0025
CABLE_SEG_MASS = 0.045 / 24
CABLE_FRICTION = (0.8, 0.005, 0.001)
CABLE_JOINT_DAMPING = 0.2
USB_SIZE = (0.025, 0.012, 0.008)
USB_MASS = 0.006

# Default fixture poses (x, y) in world frame -- PLACEHOLDERS used only until
# Builder B's scenes/fixtures.json lands; every use is logged per episode.
DEFAULT_FIXTURES = {
    "eye_drops_to_basket": {
        "platform": {"pos": [0.32, -0.06, 0.0]},
        "basket": {"pos": [0.30, 0.15, 0.0]},
    },
    "ring_insert": {
        "board": {"pos": [0.30, 0.0, 0.0]},
    },
    "tool_insert": {
        "tray": {"pos": [0.40, 0.0, 0.0]},
        "slot": {"pos": [0.35, 0.06, 0.045], "yaw_deg": 0.0},
    },
    "cable_clip": {
        "clip_block": {"pos": [0.30, 0.05, 0.0], "yaw_deg": 0.0},
    },
}

# --- Amendment 1: suite versioning + variants ---
SUITE_VERSION = "twin-suite-v1.1"
VARIANT_SEED = 20261122
VARIANT_K = 5
# Pose jitter (founder ruling 2026-11-25, adopting the commissioned
# pose-error measurement, jitter-pose-error-measurement-20261125.md):
# the former per-tier placeholders (high +-5 mm / +-3 deg, medium
# +-10 mm / +-6 deg -- a logged SAMPLING CHOICE standing in for an error
# estimate that was never measured) are replaced by the MEASURED
# distribution. The xy bound is a SINGLE pooled tier: held-out radial
# p90 = 0.088 m (n=73; the measured per-tier p90s, high 0.087 /
# medium 0.090, coincide and their medians invert, so the data does not
# support separate tiers). Yaw is UNMEASURED -- the value is None, the
# 3/6 deg placeholders are removed rather than carried forward, and the
# sampler draws no yaw jitter until a yaw-error measurement exists.
JITTER = {"pooled": (0.088, None)}
# Per-axis sigma of the same measured distribution (pooled sigma/axis
# 4.26 cm, n=73 held-out): the Gaussian sigma build_scenes.make_draws
# samples xy jitter with, for every tier (the measured per-tier sigmas,
# high 0.046 / medium 0.039, do not stratify; the pose files' rubric
# sigmas were a budget, not a measurement).
JITTER_SIGMA_XY_M = 0.0426
# Mass sampling ranges (kg) from Builder A's sourced ranges.
MASS_RANGES = {"cube_a": (0.010, 0.035), "cube_b": (0.010, 0.035),
               "carton": (0.020, 0.045), "ring": (0.004, 0.015),
               "tool": (0.050, 0.100), "cable_total": (0.040, 0.060)}
FRICTION_MULT_RANGE = (0.8, 1.2)   # +-20% where Builder A gives no range
                                   # (founder-decided convention 2026-11-25:
                                   # a robustness stress envelope, not a
                                   # measurement claim)

# --- v1.1: secondary criterion + cable solver-failure bounds ---
# Near-miss reporting thresholds (founder-decided convention 2026-11-25:
# a reporting label; continuity of the recorded results is the point).
SECONDARY_THRESH_M = {"block_stack": 0.030, "eye_drops_to_basket": 0.030,
                      "tool_insert": 0.030, "ring_insert": 0.020,
                      "cable_clip": None}   # cable exempt (deformable)
# Cable explosion envelope (see replay.evaluate + changelog v1.1 #10):
# workspace/table within ~0.6 m of origin; total cable length 0.72 m.
# Cable channel tolerances (v1.1 fixpass FP-B / V11-02): REGISTERED here,
# values UNCHANGED from v1/v1.1 replay.py literals. Basis (built jaw
# geometry, per V-Code's analysis): the clip jaw geoms clip_jaw_a/b in
# build_scenes.py sit at centres y = +/-(CLIP_CHANNEL_W/2 + 0.004) with
# half-extents (0.012, 0.004, 0.006) m. The x bound equals the jaw
# half-length exactly (0.012 m); the y bound is the channel half-width
# plus 0.002 m, reaching exactly the jaw inner faces' outer edge; the
# z bound equals the jaw half-height exactly (+/-0.006 m), like the x
# bound (converted 2026-11-25: the former +/-0.012 m was twice the jaw
# half-height -- a x2 multiplier choice, not a measurement).
CABLE_CHAN_X_HALF_M = 0.012
CABLE_CHAN_Y_EXTRA_M = 0.002
CABLE_CHAN_Z_HALF_M = 0.006
CABLE_MAX_RADIAL_M = 1.0
CABLE_Z_MIN_M = -0.05
CABLE_Z_MAX_M = 0.50
