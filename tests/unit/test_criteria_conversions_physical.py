"""Regression tests for the criteria A+B conversions (founder ruling
2026-11-25; criteria-provenance manifest, criteria A+B change).

Three hand-made rules were converted to physics-state reads, mirroring
the ring seat test's structure (test_ring_seat_physical.py):

- B1 SETTLE: the fixed SETTLE_S = 1.0 s wait is gone. _finish settles
  until the engine reports rest (every dynamic object body's speed
  below |g| * dt read from the model). A scene already at rest takes
  ZERO settle steps; a scene still moving at the old 1.0 s mark is
  stepped until it actually rests -- the old code scored the
  mid-transient state.
- B2 BLOCK: the 25% footprint inset band is gone. block_stack scores
  contact with the lower cube + sole support + settled. A physically
  stacked cube whose centre overhangs the former inset PASSES; a cube
  inside the former inset, in contact, but still moving FAILS (the old
  code passed it -- it had no settled read).
- B3 GRASP: the GRASP_DIST_M 3.0 cm proximity gate is gone. The weld
  attach gate reads pad-object CONTACT (the gripper site reaches the
  object's material). A near pass 1.3 cm from the material (inside the
  old proximity band, centre distance 2.8 cm < the old 3.0 cm bound)
  must NOT attach; a true contact pass must still attach and drag the
  object (the conversion must not over-tighten the device).

The B1/B2 tests run replay.evaluate() / replay._finish() against
minimal MuJoCo scenes built with twin_params geometry, as in the ring
seat tests. The B3 tests run replay.run_episode() end-to-end on a
synthetic hinge-chain arm scene with a crafted recorded trajectory.
Requires mujoco (the twin-replay runtime dependency); skipped where
mujoco is unavailable.

Drilled 2026-11-25: against the pre-change tree (base b2a407c) five of
these six tests fail -- both block verdicts at the verdict assertions,
both settle tests at the converted _finish signature, and the grasp
near-pass on its attach assertion (the old gate DID attach) -- while
the true-contact grasp test passes on both trees, as designed.
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

E = P.CUBE_EDGE
FORMER_INSET = (E / 2) * 0.75   # the abolished band's edge, for drill assertions


def _no_poses(monkeypatch):
    monkeypatch.setattr(replay.B, "load_episode_poses", lambda task, epi: None)
    monkeypatch.setattr(replay.B, "load_fixtures", lambda task, poses=None: {})


def _cubes_xml(cube_a_pos, cube_b_pos=(0.30, 0.0), friction=None):
    fr = friction or P.CUBE_FRICTION
    fr_s = f"{fr[0]} {fr[1]} {fr[2]}"
    return f"""
<mujoco>
  <option timestep="{P.TIMESTEP}"/>
  <worldbody>
    <geom name="mat" type="plane" size="0 0 0.01" friction="{fr_s}"/>
    <body name="cube_b" pos="{cube_b_pos[0]} {cube_b_pos[1]} {E / 2}">
      <freejoint/>
      <geom name="cube_b_geom" type="box" size="{E/2} {E/2} {E/2}" mass="{P.CUBE_MASS}" friction="{fr_s}"/>
    </body>
    <body name="cube_a" pos="{cube_a_pos[0]} {cube_a_pos[1]} {cube_a_pos[2]}">
      <freejoint/>
      <geom name="cube_a_geom" type="box" size="{E/2} {E/2} {E/2}" mass="{P.CUBE_MASS}" friction="{fr_s}"/>
    </body>
  </worldbody>
</mujoco>"""


def _step(model, data, seconds):
    for _ in range(int(seconds / P.TIMESTEP)):
        mujoco.mj_step(model, data)


def _evaluate_cubes(monkeypatch, model, data):
    _no_poses(monkeypatch)
    mujoco.mj_forward(model, data)
    return replay.evaluate(model, data, "block_stack", 1, mujoco)


# ---------------------------------------------------------------- B2 --

def test_block_overhanging_the_former_inset_but_stacked_passes(monkeypatch):
    # Upper cube resting on the lower with centre offset 13 mm: beyond
    # the abolished inset (11.25 mm) but inside the lower cube's top
    # face (15 mm), so physics holds it. The old band scored this FAIL.
    model = mujoco.MjModel.from_xml_string(
        _cubes_xml((0.30 + 0.013, 0.0, E + E / 2)))
    data = mujoco.MjData(model)
    _step(model, data, 2.0)
    res = _evaluate_cubes(monkeypatch, model, data)
    assert res["sim_success"] is True
    assert abs(res["checks"]["centre_offset_xy_m"][0]) > FORMER_INSET
    assert res["checks"]["contact_with_lower"] is True
    assert res["checks"]["no_other_support_contact"] is True
    assert res["checks"]["settled"] is True


def test_block_within_the_former_inset_but_moving_fails(monkeypatch):
    # Upper cube centred within the former inset and in contact, but
    # still sliding at the read: the old code scored this PASS (it had
    # no settled read); the physical test must FAIL it.
    model = mujoco.MjModel.from_xml_string(
        _cubes_xml((0.30 + 0.005, 0.0, E + E / 2)))
    data = mujoco.MjData(model)
    _step(model, data, 2.0)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_a")
    adr = model.jnt_dofadr[model.body_jntadr[bid]]
    data.qvel[adr] = 0.05  # lateral slide, far above the |g|*dt rest bound
    res = _evaluate_cubes(monkeypatch, model, data)
    assert res["sim_success"] is False
    assert abs(res["checks"]["centre_offset_xy_m"][0]) < FORMER_INSET
    assert res["checks"]["contact_with_lower"] is True
    assert res["checks"]["settled"] is False


# ---------------------------------------------------------------- B1 --

def _finish_cubes(monkeypatch, tmp_path, model, data, max_settle_steps=100000):
    _no_poses(monkeypatch)
    return replay._finish(model, data, "block_stack", 1, [], 0.0, 1.0, False,
                          None, False, tmp_path, mujoco,
                          max_settle_steps=max_settle_steps)


def test_settle_takes_zero_steps_when_already_at_rest(monkeypatch, tmp_path):
    model = mujoco.MjModel.from_xml_string(
        _cubes_xml((0.30, 0.10, E / 2), cube_b_pos=(0.30, 0.0)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    res = _finish_cubes(monkeypatch, tmp_path, model, data)
    # The old fixed wait burned int(1.0 / TIMESTEP) = 2000 steps here.
    assert res["settle_steps"] == 0
    assert res["settle_at_rest"] is True


def test_settle_steps_until_rest_past_the_old_fixed_wait(monkeypatch, tmp_path):
    # Cube sliding on a low-friction plane: at the old 1.0 s mark it is
    # still moving at ~0.1 m/s; it comes to rest only at ~1.5 s. The
    # old code scored the mid-slide state; the rest read must step past
    # the old wait and return the resolved state.
    low = (0.02, 0.005, 0.0001)
    model = mujoco.MjModel.from_xml_string(
        _cubes_xml((0.05, 0.10, E / 2), cube_b_pos=(0.30, -0.15), friction=low))
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_a")
    adr = model.jnt_dofadr[model.body_jntadr[bid]]
    data.qvel[adr] = 0.30
    x0 = float(data.xpos[bid][0])
    res = _finish_cubes(monkeypatch, tmp_path, model, data)
    assert res["settle_steps"] > int(1.0 / P.TIMESTEP)
    assert res["settle_at_rest"] is True
    at_rest, _, _ = replay._body_at_rest(model, data, mujoco, bid)
    assert at_rest is True
    assert float(data.xpos[bid][0]) - x0 > 0.15   # it really slid


# ---------------------------------------------------------------- B3 --

ARM_SCENE = """
<mujoco>
  <option timestep="{ts}"/>
  <worldbody>
    <geom name="mat" type="plane" size="0 0 0.01" friction="0.9 0.005 0.0001"/>
    <body name="base" pos="0 0 0.10">
      <body name="shoulder" pos="0 0 0">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.005" mass="0.01" contype="0" conaffinity="0"/>
        <body name="upper_arm" pos="0 0 0">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.005" mass="0.01" contype="0" conaffinity="0"/>
          <body name="lower_arm" pos="0.25 0 0">
            <joint name="elbow_flex" type="hinge" axis="0 1 0"/>
            <geom type="sphere" size="0.005" mass="0.01" contype="0" conaffinity="0"/>
            <body name="wrist" pos="0 0 0">
              <joint name="wrist_flex" type="hinge" axis="0 1 0"/>
              <geom type="sphere" size="0.005" mass="0.01" contype="0" conaffinity="0"/>
              <body name="gripper" pos="0 0 0">
                <joint name="wrist_roll" type="hinge" axis="1 0 0"/>
                <geom type="sphere" size="0.005" mass="0.01" contype="0" conaffinity="0"/>
                <site name="gripperframe" pos="0 0 -0.02"/>
                <body name="moving_jaw_so101_v1" pos="0 0 0">
                  <joint name="gripper" type="hinge" axis="0 1 0"/>
                  <geom type="sphere" size="0.004" mass="0.005" contype="0" conaffinity="0"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="cube_a" pos="{ax} {ay} {hz}">
      <freejoint/>
      <geom name="cube_a_geom" type="box" size="{h} {h} {h}" mass="0.02" friction="0.6 0.005 0.0001"/>
    </body>
    <body name="cube_b" pos="0.10 -0.25 {hz}">
      <freejoint/>
      <geom name="cube_b_geom" type="box" size="{h} {h} {h}" mass="0.02" friction="0.6 0.005 0.0001"/>
    </body>
  </worldbody>
  <equality>
    {welds}
  </equality>
</mujoco>"""


def _run_grasp_episode(monkeypatch, tmp_path, cube_a_xy):
    """One synthetic recorded episode: the gripper site sweeps an arc of
    radius 0.25 m past cube_a while the gripper closes, holds, reopens.
    Returns (result, cube_a spawn xy)."""
    twin = tmp_path / "twin"
    (twin / "scenes").mkdir(parents=True)
    records = tmp_path / "records"
    records.mkdir()
    welds = "".join(
        f'<weld name="weld_{obj}_{k}" body1="gripper" body2="{obj}" active="false" '
        f'solref="0.005 1" solimp="0.95 0.99 0.001"/>'
        for obj in ("cube_a", "cube_b") for k in range(4))
    xml = ARM_SCENE.format(ts=P.TIMESTEP, ax=cube_a_xy[0], ay=cube_a_xy[1],
                           hz=E / 2, h=E / 2, welds=welds)
    (twin / "scenes" / "block_stack__ep990_v0.xml").write_text(xml)
    T = 200
    gt = np.zeros((T, 6))
    gt[:, 0] = np.linspace(-25.0, 25.0, T)          # shoulder_pan, degrees
    grip = np.full(T, 40.0)                          # recorded channel: high = open
    grip[40:80] = np.linspace(40.0, 2.0, 40)         # close
    grip[80:160] = 2.0                               # hold closed
    grip[160:200] = np.linspace(2.0, 40.0, 40)       # reopen (release)
    gt[:, 5] = grip
    np.savez(records / "base__block_stack__ep990.npz", gt=gt)
    monkeypatch.setattr(P, "TWIN", twin)
    monkeypatch.setattr(P, "RECORDS", records)
    _no_poses(monkeypatch)
    res = replay.run_episode("block_stack", 990, render=False,
                             out_dir=tmp_path / "out")
    return res, cube_a_xy


def test_grasp_near_pass_inside_the_old_proximity_band_does_not_attach(
        monkeypatch, tmp_path):
    # Cube centre 28 mm from the site's closest approach: planar
    # distance to the cube's material 13 mm, centre distance 28 mm --
    # BOTH inside the abolished 3.0 cm gate, so the old code attached
    # and dragged the cube. The contact read must not attach.
    res, spawn = _run_grasp_episode(monkeypatch, tmp_path, (0.278, 0.0))
    attaches = [e for e in res["grasp_events"]
                if e["event"] == "attach" and e["object"] == "cube_a"]
    assert attaches == []
    final = res["objects_final"]["cube_a"]
    assert math.hypot(final[0] - spawn[0], final[1] - spawn[1]) < 0.01


def test_grasp_true_contact_still_attaches_and_drags(monkeypatch, tmp_path):
    # Cube squarely on the site's arc: the site reaches the material
    # (planar shape distance 0), so the device must attach, drag the
    # cube along the sweep, and release it at the reopen.
    res, spawn = _run_grasp_episode(monkeypatch, tmp_path, (0.25, 0.0))
    attaches = [e for e in res["grasp_events"]
                if e["event"] == "attach" and e["object"] == "cube_a"]
    assert len(attaches) == 1
    final = res["objects_final"]["cube_a"]
    assert math.hypot(final[0] - spawn[0], final[1] - spawn[1]) > 0.04
