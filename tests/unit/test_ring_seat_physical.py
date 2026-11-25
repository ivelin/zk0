"""Regression test for the ring-insert seat criterion (founder ruling
2026-11-25 11:50 CT; ring band-edge postmortem 2026-11-25).

The former criterion gated the seat on an invented height band
(0.0 <= ring_bottom - board_top <= 0.010). The band-edge probe
(2026-11-24) showed every proven-flat seat settles ~3 um BELOW the
band's lower edge and fails on that field alone, while the one
criteria-passing end state on record (Stage A2, ep48 v2) sat 1.0 mm
HIGH. The ruling replaced the band with a physical test: the ring
passes iff it is ENGAGED on the peg (peg through the hole), in
seat-support CONTACT with the board or peg at settle end, and SETTLED
(not device-held; speed below one integration step of gravitational
velocity gain, |g| * dt read from the model).

These tests run replay.evaluate() itself against a minimal MuJoCo
scene built with the same construction as build_scenes.py's ring
branch (board cylinder, peg cylinder, 12-capsule ring, twin_params
geometry):

- a genuinely settled flat seat -- whose bottom_above_board_m reads
  NEGATIVE, i.e. below the old band edge -- must PASS;
- the ep48 v2 end state (recorded pose: planar 2.5 mm, centre z
  0.0217 m, 1.0 mm above flat rest, no support contact) must FAIL;
- a seat still held by the grasp weld must FAIL;
- a ring still moving at settle end must FAIL;
- a ring resting on the board away from the peg must FAIL.

Drilled 2026-11-25: against the pre-ruling replay.py this file fails
(the flat seat fails the band; the floating ep48 v2 state passes it);
against the physical test it passes. Requires mujoco (the twin-replay
runtime dependency); skipped where mujoco is unavailable.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

CODE_DIR = Path(__file__).resolve().parents[2] / "twin-replay" / "code"
sys.path.insert(0, str(CODE_DIR))

import twin_params as P  # noqa: E402

# build_scenes reads its calibration data from P.TWIN at import time;
# point it at this repo's own twin-replay tree (which carries the
# calibration files) instead of the DGX live-tree path.
P.TWIN = Path(__file__).resolve().parents[2] / "twin-replay"

import replay  # noqa: E402

BT = P.BOARD_H
RMID = (P.RING_R_OUT + P.RING_R_IN) / 2
TUBE = min((P.RING_R_OUT - P.RING_R_IN) / 2, P.RING_THICK / 2)
PEG_XY = (0.30, 0.0)


def _scene_xml(ring_pos, weld=False):
    geoms = []
    for k in range(P.RING_SEGMENTS):
        a0 = 2 * math.pi * k / P.RING_SEGMENTS
        a1 = 2 * math.pi * (k + 1) / P.RING_SEGMENTS
        p0 = (RMID * math.cos(a0), RMID * math.sin(a0), 0)
        p1 = (RMID * math.cos(a1), RMID * math.sin(a1), 0)
        geoms.append(
            f'<geom type="capsule" fromto="{p0[0]} {p0[1]} {p0[2]} {p1[0]} {p1[1]} {p1[2]}" '
            f'size="{TUBE}" mass="{P.RING_MASS / P.RING_SEGMENTS}" '
            f'friction="{P.RING_FRICTION[0]} {P.RING_FRICTION[1]} {P.RING_FRICTION[2]}"/>'
        )
    weld_xml = (
        '<equality><weld name="weld_ring_0" body1="gripper" body2="ring" active="true"/></equality>'
        if weld
        else ""
    )
    gripper = (
        f'<body name="gripper" pos="{ring_pos[0]} {ring_pos[1]} {ring_pos[2] + 0.05}"/>'
        if weld
        else ""
    )
    return f"""
<mujoco>
  <option timestep="{P.TIMESTEP}"/>
  <worldbody>
    <geom name="mat" type="plane" size="0 0 0.01"
      friction="{P.MAT_FRICTION[0]} {P.MAT_FRICTION[1]} {P.MAT_FRICTION[2]}"/>
    <body name="board" pos="{PEG_XY[0]} {PEG_XY[1]} 0">
      <geom name="board_geom" type="cylinder" pos="0 0 {BT / 2}" size="{P.BOARD_R} {BT / 2}"
        friction="{P.BOARD_FRICTION[0]} {P.BOARD_FRICTION[1]} {P.BOARD_FRICTION[2]}"/>
    </body>
    <body name="peg" pos="{PEG_XY[0]} {PEG_XY[1]} 0">
      <geom name="peg_geom" type="cylinder" pos="0 0 {BT + P.PEG_H / 2}" size="{P.PEG_R} {P.PEG_H / 2}"
        friction="{P.PEG_FRICTION[0]} {P.PEG_FRICTION[1]} {P.PEG_FRICTION[2]}"/>
    </body>
    {gripper}
    <body name="ring" pos="{ring_pos[0]} {ring_pos[1]} {ring_pos[2]}">
      <freejoint/>
      {''.join(geoms)}
    </body>
  </worldbody>
  {weld_xml}
</mujoco>"""


def _evaluate(monkeypatch, model, data):
    monkeypatch.setattr(replay.B, "load_episode_poses", lambda task, epi: None)
    monkeypatch.setattr(
        replay.B,
        "load_fixtures",
        lambda task, poses=None: {"board": {"pos": [PEG_XY[0], PEG_XY[1], 0.0]}},
    )
    mujoco.mj_forward(model, data)
    return replay.evaluate(model, data, "ring_insert", 48, mujoco)


def _settled_seat(monkeypatch, weld=False):
    model = mujoco.MjModel.from_xml_string(
        _scene_xml([PEG_XY[0], PEG_XY[1], BT + TUBE], weld=weld)
    )
    data = mujoco.MjData(model)
    for _ in range(int(2.0 / P.TIMESTEP)):
        mujoco.mj_step(model, data)
    return model, data


def test_settled_flat_seat_below_the_old_band_edge_passes(monkeypatch):
    model, data = _settled_seat(monkeypatch)
    res = _evaluate(monkeypatch, model, data)
    # The regression essence: this seat's bottom reads BELOW the old
    # band's 0.0 edge (contact settling), yet it is a true seat.
    raw_bottom = res["objects_final"]["ring"][2] - P.RING_THICK / 2 - BT
    assert raw_bottom < 0
    assert res["checks"]["bottom_above_board_m"] <= 0.0
    assert res["checks"]["engaged"] is True
    # The verdict itself, asserted before the new diagnostic fields so
    # the drill against the pre-ruling code fails HERE (the band scores
    # this seat FAIL) and not on a missing key.
    assert res["sim_success"] is True
    assert res["checks"]["seat_support_contact"] is True
    assert res["checks"]["settled"] is True


def test_floating_high_ring_ep48_v2_state_fails(monkeypatch):
    # Stage A2 ep48 v2 recorded end state: planar 2.5 mm, engaged,
    # ring centre z 0.0217 m (1.0 mm above flat rest). No stepping --
    # evaluate the recorded pose as-is.
    model = mujoco.MjModel.from_xml_string(
        _scene_xml([PEG_XY[0] + 0.0025, PEG_XY[1], 0.0217])
    )
    data = mujoco.MjData(model)
    res = _evaluate(monkeypatch, model, data)
    assert res["checks"]["engaged"] is True
    # Verdict first (drill hygiene, as above): the old band scores this
    # floating state PASS -- that inversion is the defect.
    assert res["sim_success"] is False
    assert res["checks"]["seat_support_contact"] is False


def test_weld_held_seat_fails(monkeypatch):
    model, data = _settled_seat(monkeypatch, weld=True)
    res = _evaluate(monkeypatch, model, data)
    assert res["checks"]["held_by_device"] is True
    assert res["checks"]["settled"] is False
    assert res["sim_success"] is False


def test_moving_ring_is_not_settled(monkeypatch):
    model, data = _settled_seat(monkeypatch)
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ring")
    jid = model.body_jntadr[rid]
    adr = model.jnt_dofadr[jid]
    data.qvel[adr] = 0.05  # lateral slide, far above the |g|*dt rest bound
    res = _evaluate(monkeypatch, model, data)
    assert res["checks"]["settled"] is False
    assert res["sim_success"] is False


def test_ring_resting_away_from_peg_fails(monkeypatch):
    model = mujoco.MjModel.from_xml_string(
        _scene_xml([PEG_XY[0] + 0.05, PEG_XY[1], BT + TUBE])
    )
    data = mujoco.MjData(model)
    for _ in range(int(2.0 / P.TIMESTEP)):
        mujoco.mj_step(model, data)
    res = _evaluate(monkeypatch, model, data)
    assert res["checks"]["engaged"] is False
    assert res["sim_success"] is False
