"""Object factory for the description->scene test. Every object returns
(assets_xml, body_xml, info) where info carries the spec-sheet data:
source library, mass + mass provenance, placed dims, grasp profile.

Mass provenance labels: 'published' = measured value from the YCB
literature (YCB-Handovers Table I, arXiv 2512.20847); 'regulation' =
ITF tennis-ball spec; 'measured-eval' = the 2026-11-15 TripoSG eval;
'class-estimate' = realistic class value assigned by the spec layer
(no library anywhere ships measured friction/mass for these).
"""
import glob
import numpy as np
import mujoco
from harness import (obj_audit, load_obj_verts, pca_align_quat, prep_mesh,
                     quat_from_euler, quat_axis_angle, mesh_body, prim_body,
                     ycb_obj, ycb_tex, box_inertia, TRIPO, YCB)

IDENT = np.array([1.0, 0, 0, 0])

# name -> (ycb id, mass kg, mass source, quat mode)
YCB_DEFS = {
    "mug":        ("025_mug", 0.118, "published", "yup_handle_y"),
    "apple":      ("013_apple", 0.160, "class-estimate", "zup"),
    "pear":       ("016_pear", 0.150, "class-estimate", "yup_down"),
    "banana":     ("011_banana", 0.120, "class-estimate", "pca"),
    "sponge":     ("026_sponge", 0.020, "class-estimate", "zup"),
    "marker":     ("040_large_marker", 0.015, "published", "pca"),
    "screwdriver": ("043_phillips_screwdriver", 0.095, "published", "pca2d"),
    "chef_can":   ("002_master_chef_can", 0.410, "published", "zup"),
    "tuna_can":   ("007_tuna_fish_can", 0.160, "class-estimate", "zup"),
    "foam_brick": ("061_foam_brick", 0.030, "class-estimate", "zup"),
    "tennis_ball": ("056_tennis_ball", 0.058, "regulation", "zup"),
    "orange":     ("017_orange", 0.160, "class-estimate", "zup"),
    "plum":       ("018_plum", 0.060, "class-estimate", "zup"),
    "bowl":       ("024_bowl", 0.149, "published", "zup"),
}

def _quat_for(mode, verts):
    if mode == "zup":
        return IDENT.copy()
    if mode == "yup":      # mesh +Y is up -> Rx(+90deg): y axis -> z
        return quat_from_euler(np.pi / 2, 0, 0)
    if mode == "yup_down":  # mesh -Y is up -> Rx(-90deg)
        return quat_from_euler(-np.pi / 2, 0, 0)
    if mode == "yup_handle_y":  # Rx(+90): y->z (up); then Rz(+90): x->y (handle -> +Y)
        from harness import quat_mul
        return quat_mul(quat_from_euler(0, 0, np.pi / 2), quat_from_euler(np.pi / 2, 0, 0))
    if mode == "zup_handle_y":  # rotate about Z by 90 deg: handle (native +X) -> +Y
        return quat_from_euler(0, 0, np.pi / 2)
    if mode == "pca":
        return pca_align_quat(verts)
    if mode == "pca2d":
        # robust 2D alignment for flat-lying tools: drop outlier verts, take
        # the XY-plane principal axis, rotate about Z so PC1 -> +Y
        lo = np.percentile(verts, 1, axis=0); hi = np.percentile(verts, 99, axis=0)
        Vr = verts[np.all((verts >= lo) & (verts <= hi), axis=1)]
        C = np.cov((Vr[:, :2] - Vr[:, :2].mean(0)).T)
        w, E = np.linalg.eigh(C)
        pc1 = E[:, int(np.argmax(w))]
        ang = np.arctan2(pc1[1], pc1[0])
        return quat_from_euler(0, 0, np.pi / 2 - ang)
    raise ValueError(mode)

def ycb_object(inst, kind, pos_xy, mu, rgba="1 1 1 1", use_texture=True):
    yid, mass, msrc, mode = YCB_DEFS[kind]
    path = ycb_obj(yid)
    verts, _ = load_obj_verts(path)
    q = _quat_for(mode, verts)
    prep = prep_mesh([path], q)
    di = box_inertia(mass, prep["dims"])
    assets, body = mesh_body(
        inst, path, (pos_xy[0], pos_xy[1], prep["rest_z"]), q, mass, di, rgba, mu,
        texfile=ycb_tex(yid) if use_texture else None,
        geom_pos=tuple(prep["geom_pos"]))
    info = dict(kind=kind, source=f"YCB {yid} (google_16k scan)",
                mass_kg=mass, mass_source=msrc,
                dims_mm=np.round(prep["dims"] * 1000, 1).tolist(),
                rest_z=round(float(prep["rest_z"]), 4),
                grasp_width_mm=round(float(min(prep["dims"][0], prep["dims"][1])) * 1000, 1),
                prep=prep, quat=q)
    return assets, body, info

def ycb_object_with_pose(inst, kind, pos3, quat, mu, rgba="1 1 1 1"):
    """ycb_object but spawning the body at an explicit measured pose (pos3,
    quat) instead of the canonical alignment pose — used for objects whose
    natural rest pose differs from their scan alignment (the screwdriver:
    spawn = the pose a settle probe measured it coming to rest in)."""
    yid, mass, msrc, mode = YCB_DEFS[kind]
    path = ycb_obj(yid)
    verts, _ = load_obj_verts(path)
    q = _quat_for(mode, verts)
    prep = prep_mesh([path], q)
    di = box_inertia(mass, prep["dims"])
    assets, body = mesh_body(
        inst, path, tuple(pos3), tuple(quat), mass, di, rgba, mu,
        texfile=ycb_tex(yid), geom_pos=tuple(prep["geom_pos"]))
    info = dict(kind=kind, source=f"YCB {yid} (google_16k scan), measured rest pose",
                mass_kg=mass, mass_source=msrc,
                dims_mm=np.round(prep["dims"] * 1000, 1).tolist(),
                rest_z=round(float(pos3[2]), 4),
                grasp_width_mm=round(float(min(prep["dims"][0], prep["dims"][1])) * 1000, 1),
                prep=prep, quat=np.array(quat, dtype=float))
    return assets, body, info

def triposg_mug(inst, pos_xy, mu, rgba="0.85 0.45 0.2 1"):
    """The 2026-11-15 eval mug: CoACD hulls (24) + decimated visual, already
    z-upred and spec-scaled in the eval's frame (85 mm tall, body 68.2 mm,
    mass 0.151 kg at ceramic density)."""
    hulls = sorted(glob.glob(TRIPO + "/mug_hull_*.obj"))
    prep = prep_mesh(hulls, IDENT)
    scale = 0.085 / prep["dims"][2]
    # rescale prep to spec height
    prep2 = prep_mesh(hulls, IDENT)
    dims = prep2["dims"] * scale
    mass = 0.151
    di = box_inertia(mass, dims)
    assets, body = mesh_body(
        inst, TRIPO + "/mug_decimated.obj",
        (pos_xy[0], pos_xy[1], prep2["rest_z"] * scale), IDENT, mass, di, rgba, mu,
        hull_files=hulls, mesh_scale=round(float(scale), 5),
        geom_pos=tuple(prep2["geom_pos"] * scale))
    info = dict(kind="mug_triposg", source="TripoSG generation (2026-11-15 eval assets)",
                mass_kg=mass, mass_source="measured-eval",
                dims_mm=np.round(dims * 1000, 1).tolist(),
                rest_z=round(float(prep2["rest_z"] * scale), 4),
                grasp_width_mm=round(float(min(dims[0], dims[1])) * 1000, 1),
                prep=prep2, quat=IDENT)
    return assets, body, info

def triposg_apple(inst, pos_xy, mu, rgba="0.8 0.15 0.1 1"):
    """TripoSG apple, spec-scaled to 68 mm diameter (in-envelope instance)."""
    path = TRIPO + "/apple_decimated.obj"
    prep = prep_mesh([path], IDENT)
    scale = 0.068 / prep["dims"][0]
    dims = prep["dims"] * scale
    mass = 0.140
    di = [0.4 * mass * (dims[0] / 2) ** 2] * 3
    assets, body = mesh_body(
        inst, path, (pos_xy[0], pos_xy[1], prep["rest_z"] * scale), IDENT, mass, di,
        rgba, mu, mesh_scale=round(float(scale), 5),
        geom_pos=tuple(prep["geom_pos"] * scale))
    info = dict(kind="apple_triposg", source="TripoSG generation (2026-11-15 eval assets)",
                mass_kg=mass, mass_source="class-estimate",
                dims_mm=np.round(dims * 1000, 1).tolist(),
                rest_z=round(float(prep["rest_z"] * scale), 4),
                grasp_width_mm=round(float(dims[0]) * 1000, 1),
                prep=prep, quat=IDENT)
    return assets, body, info

def spec_cube(inst, pos_xy, mu, rgba, size=0.04, mass=0.064):
    h = size / 2
    body = prim_body(inst, "box", (h, h, h), (pos_xy[0], pos_xy[1], h), IDENT, mass, rgba, mu)
    info = dict(kind="cube", source="zk0 spec-layer primitive",
                mass_kg=mass, mass_source="battery-proven spec",
                dims_mm=[size * 1000] * 3, rest_z=h, grasp_width_mm=size * 1000,
                prep=None, quat=IDENT)
    return "", body, info

def spec_book(inst, pos_xy, mu, yaw=0.0, rgba="0.55 0.2 0.15 1"):
    q = quat_from_euler(0, 0, yaw)
    body = prim_body(inst, "box", (0.115, 0.08, 0.016), (pos_xy[0], pos_xy[1], 0.016),
                     q, 0.450, rgba, mu)
    info = dict(kind="book", source="zk0 spec-layer primitive",
                mass_kg=0.450, mass_source="class-estimate",
                dims_mm=[230, 160, 32], rest_z=0.016, grasp_width_mm=None,
                prep=None, quat=q)
    return "", body, info

def spec_ball(inst, pos_xy, mu, rgba, r=0.03, mass=0.05):
    body = prim_body(inst, "sphere", r, (pos_xy[0], pos_xy[1], r), IDENT, mass, rgba, mu)
    info = dict(kind="ball", source="zk0 spec-layer primitive",
                mass_kg=mass, mass_source="class-estimate",
                dims_mm=[2 * r * 1000] * 3, rest_z=r, grasp_width_mm=2 * r * 1000,
                prep=None, quat=IDENT)
    return "", body, info
