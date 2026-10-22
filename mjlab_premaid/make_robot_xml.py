"""Build the robot-only MJCF for mjlab from mujoco_sim/assets/premaidai.xml.

- removes the floor, debris, lights, actuators and sensors (mjlab adds terrain, actuators, sensors)
- replaces the convex-hull mesh colliders with fitted capsules (limbs) and boxes (torso, head, feet):
  much cheaper for MuJoCo Warp's GPU collision, especially for get-up's many self/ground contacts
- every robot collider gets a name "<body>_col" and full self-collision (parent/child pairs excluded)

    python mjlab_premaid/make_robot_xml.py        # writes mjlab_premaid/assets/premaid_mjlab.xml
"""

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "mujoco_sim", "assets", "premaidai.xml")
OUT = os.path.join(HERE, "assets", "premaid_mjlab.xml")

ROBOT_FRICTION = 0.3
CAPSULE_BODIES = ["l_uleg", "r_uleg", "l_lleg", "r_lleg", "l_uarm", "r_uarm", "l_larm", "r_larm", "l_hand", "r_hand"]
BOX_BODIES = ["base_link", "head", "l_foot", "r_foot", "l_ankle", "r_ankle"]


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def body_points(m, body):
    """Vertices of the body's visual mesh, in the body frame."""
    bid = m.body(body).id
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(v @ quat_to_mat(m.geom_quat[g]).T + m.geom_pos[g])
    return np.concatenate(pts)


def fit_capsule(p, radius_pct=85):
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    ax = vt[0]
    t = (p - c) @ ax
    perp = np.linalg.norm((p - c) - np.outer(t, ax), axis=1)
    r = np.percentile(perp, radius_pct)
    t0, t1 = np.percentile(t, 3) + r, np.percentile(t, 97) - r
    if t1 < t0:
        t0 = t1 = 0.5 * (t0 + t1)
    return c + t0 * ax, c + t1 * ax, r


def fit_box(p):
    lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
    return (lo + hi) / 2, (hi - lo) / 2


def main():
    spec = mujoco.MjSpec.from_file(SRC)
    m = spec.compile()

    # drop world clutter
    for g in list(spec.worldbody.geoms):
        spec.delete(g)
    for b in list(spec.worldbody.bodies):
        if b.name.startswith("debris"):
            spec.delete(b)
    for l in list(spec.worldbody.lights):
        spec.delete(l)
    for a in list(spec.actuators):
        spec.delete(a)
    for s in list(spec.sensors):
        spec.delete(s)
    for t in list(spec.textures):
        spec.delete(t)
    for mat in list(spec.materials):
        spec.delete(mat)

    # remove old mesh colliders (sole boxes are kept and renamed later)
    for b in spec.bodies:
        for g in list(b.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_MESH and (g.contype or g.conaffinity):
                spec.delete(g)

    # MuJoCo combines friction with max(): keep the robot low so the (randomized) floor decides
    def add(body, **kw):
        g = spec.body(body).add_geom(**kw)
        g.contype, g.conaffinity, g.group = 1, 1, 3
        g.rgba = [0.8, 0.3, 0.3, 0.4]
        g.density = 0
        g.friction = [ROBOT_FRICTION, 0.005, 0.0001]
        return g

    for b in CAPSULE_BODIES:
        a, z, r = fit_capsule(body_points(m, b), radius_pct=60 if "leg" in b else 85)
        add(b, name=f"{b}_col", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=np.concatenate([a, z]), size=[r, 0, 0])
    for b in BOX_BODIES:
        c, h = fit_box(body_points(m, b))
        if b.endswith("foot"):  # top part of the foot; the sole box below handles ground contact
            h = h.copy()
            h[2] *= 0.6
            c = c.copy()
            c[2] += 0.004
        add(b, name=f"{b}_col", type=mujoco.mjtGeom.mjGEOM_BOX, pos=c, size=h)

    # soles: keep as the ground-contact geoms, same friction/condim as before
    for s in ("l", "r"):
        g = spec.geom(f"{s}_sole")
        g.contype, g.conaffinity = 1, 1
        g.friction = [ROBOT_FRICTION, g.friction[1], g.friction[2]]

    # links two joints apart overlap at the joint itself (shoulder, neck, elbow): not real contacts
    for b1, b2 in [("base_link", "l_uarm"), ("base_link", "r_uarm"), ("base_link", "head"),
                   ("l_uarm", "l_larm"), ("r_uarm", "r_larm"), ("base_link", "l_uleg"), ("base_link", "r_uleg")]:
        spec.add_exclude(bodyname1=b1, bodyname2=b2)
    spec.compiler.meshdir = os.path.relpath(os.path.join(HERE, "..", "mujoco_sim", "assets", "meshes"), os.path.dirname(OUT))
    xml = spec.to_xml()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(xml)

    m2 = mujoco.MjModel.from_xml_path(OUT)
    d2 = mujoco.MjData(m2)
    print(f"wrote {OUT}: nbody={m2.nbody} njnt={m2.njnt} ngeom={m2.ngeom} nu={m2.nu} "
          f"colliders={sum(1 for g in range(m2.ngeom) if m2.geom_contype[g])}")
    return m2, d2


if __name__ == "__main__":
    main()
