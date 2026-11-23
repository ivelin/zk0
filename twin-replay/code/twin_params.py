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

FPS = 20.0            # dataset recording rate (extract_frames.py FPS=20).
                      # NOTE: prereg text also says 30 fps in one place and the
                      # task brief says 30 fps; 20 is the dataset's own rate.
                      # Flagged as ambiguity in the validation report.
TIMESTEP = 0.002      # 500 Hz physics (>=240 Hz required)
SETTLE_S = 1.0        # pre-registered settle time

# --- grasp device (docs/physics-parameters.md section 7) ---
GRASP_DIST_M = 0.030          # object centre within 3.0 cm of gripper centre
ATTACH_CLOSEDNESS = 0.50      # attach threshold (normalised closedness)
RELEASE_CLOSEDNESS = 0.30     # release threshold (hysteresis)
# Gripper-trace normalisation (CONCRETE, global algorithm; see report):
#   The recorded gripper channel is in degrees, baseline ~1-3 deg (open/rest)
#   and rises during a grasp; the peak is object-dependent (carton ~40 deg,
#   cube ~26-44, ring ~15-28, tool ~11-22) because the jaw stops on the
#   object. An absolute 50%-of-range threshold can therefore NEVER fire for
#   ring/tool episodes. The normalisation is one global algorithm applied
#   identically to every episode (no per-episode hand tuning):
#     open_ref   = 5th percentile of the episode gripper channel
#     closed_ref = 99th percentile of the episode gripper channel
#     closedness = clip((g - open_ref) / (closed_ref - open_ref), 0, 1)
#   If (closed_ref - open_ref) < 5 deg the episode has no grasp excursion
#   and no attach is possible.
GRIP_MIN_EXCURSION_DEG = 5.0

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
RING_THICK = 0.012
RING_MASS = 0.010
RING_FRICTION = (0.5, 0.005, 0.0001)
RING_SEGMENTS = 12                     # capsule ring approximation

TRAY_SIZE = (0.385, 0.270, 0.045)
TRAY_FRICTION = (0.6, 0.005, 0.0001)
SLOT_SIZE = (0.110, 0.032)             # target slot footprint (x,y) LOCAL-CHOICE
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
SUITE_VERSION = "twin-suite-v1"
VARIANT_SEED = 20261122
VARIANT_K = 5
# Pose jitter scales by Builder B confidence tier. Builder B's pose files
# carry NO per-pose numeric error field, so these tier scales are a logged
# SAMPLING CHOICE standing in for "B's quantitative error estimate"
# (flagged in the validation report): high +-5 mm / +-3 deg, medium
# +-10 mm / +-6 deg (uniform draws).
JITTER = {"high": (0.005, 3.0), "medium": (0.010, 6.0)}
# Mass sampling ranges (kg) from Builder A's sourced ranges.
MASS_RANGES = {"cube_a": (0.010, 0.035), "cube_b": (0.010, 0.035),
               "carton": (0.020, 0.045), "ring": (0.004, 0.015),
               "tool": (0.050, 0.100), "cable_total": (0.040, 0.060)}
FRICTION_MULT_RANGE = (0.8, 1.2)   # +-20% where Builder A gives no range
