"""Sim mode (spec §2): Stack A — MuJoCo + the certified harness."""

# Headless rendering: on a machine with no DISPLAY, MuJoCo's default
# GLFW backend cannot create the OpenGL context its Renderer needs.
# EGL is the headless path on the GPU build hosts; set it before
# mujoco is first imported in this process (any zk0_node.sim import
# precedes the mujoco imports in the package's own modules).
import os as _os

if not _os.environ.get("DISPLAY"):
    _os.environ.setdefault("MUJOCO_GL", "egl")
