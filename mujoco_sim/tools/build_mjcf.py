"""Build the MuJoCo (MJCF) model of Premaid AI.

Kinematics, masses and inertias come from chikuta/premaidai_description
(Apache-2.0, meshes (c) 2019 kuroiwasi). Joint names match the servo map in
chikuta/premaidai_controller/config/joint_config.yaml.

Usage:
    git clone https://github.com/chikuta/premaidai_description /tmp/premaidai_description
    pip install trimesh pycollada
    python mujoco_sim/tools/build_mjcf.py --description /tmp/premaidai_description

Writes mujoco_sim/assets/premaidai.xml and mujoco_sim/assets/meshes/*.stl.
The generated files are committed, so you only need this to regenerate.

Notes on the source URDF:
  * Inertia values are written with an "e-5" suffix but are really in units of
    1e-7 kg*m^2 (checked against the convex hull of each mesh at the given
    mass). We scale them by 1e-7 here.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets"))

D2R = math.pi / 180.0
I_SCALE = 1e-7

# Servo torque limits from the URDF (N*m): legs use the stronger servo.
N_DEBRIS = 16

LEG_EFFORT = 1.3712
ARM_EFFORT = 0.9114

# ----------------------------------------------------------------------------
# Link inertials: name -> (mass, com, (ixx, iyy, izz, ixy, ixz, iyz)) in URDF units.
# "sign" is +1 for left, -1 for right (applied to mirrored terms).
# ----------------------------------------------------------------------------

def leg_links(s):
    return {
        "hip_joint": (0.0143, (0, s * 0.000183, -0.010532), (20.138887, 53.757824, 46.127151, -0.000001, 0.0, s * -0.518773)),
        "uleg_root": (0.0648, (0, s * 0.013420, -0.004373), (246.003845, 191.922195, 238.454895, s * -0.001517, -0.001688, s * -6.165664)),
        "uleg": (0.1134, (0, 0.000001, -0.032601), (884.390869, 825.489990, 222.801773, 0.000002, -0.000046, -0.000031)),
        "lleg": (0.0719, (-0.001624, s * 0.002494, -0.034776), (735.568420, 434.662354, 480.555054, s * -1.871153, -35.647034, s * 0.247592)),
        "ankle": (0.0571, (-0.009629, 0.000379, -0.003702), (175.244446, 257.425598, 224.547821, -0.000004, 11.289989, -0.000002)),
        "foot": (0.0941, (0.005014, s * 0.008793, -0.004123), (383.406982, 1013.430359, 1206.411377, s * 11.222215, 12.157411, s * -1.649494)),
    }


def arm_links(s):
    return {
        "uarm_joint": (0.0131, (0, s * 0.005109, 0.000865), (39.090332, 116.438423, 132.243393, 0.000021, 0.000003, s * -4.044059)),
        "uarm": (0.0712, (-0.000001, s * 0.021640, 0.000670), (356.620514, 126.890160, 414.000244, -0.000020, 0.000026, s * 4.816492)),
        "larm_joint": (0.0124, (-0.008596, s * 0.015762, 0.002150), (70.485497, 188.648819, 236.494141, s * 104.342163, -0.000005, s * -3.846500)),
        "larm": (0.0812, (-0.008596, s * 0.027572, 0.000420), (342.361237, 141.587677, 414.116486, s * -0.000036, 0.000012, s * 0.382193)),
        "hand": (0.0046, (0.001817, s * 0.018911, 0.002717), (8.193970, 3.111708, 9.081884, s * -0.178483, 0.194812, s * -2.003531)),
    }


BODY_LINKS = {
    "body": (0.4426, (0.008509, 0.000002, 0.070698), (2871.143311, 2672.078857, 1188.332153, -0.001151, 237.094467, -0.003379)),
    "neck_pitch": (0.01, (-0.006449, 0.000002, 0.008964), (6.489763, 2.296302, 6.197822, -0.000003, -0.024047, -0.000005)),
    "head_yaw": (0.01, (0.008100, 0.000002, 0.017871), (12.636111, 23.689224, 17.372910, -0.000001, 0.0, 0.0)),
    "head": (0.1449, (0.014213, 0.000001, 0.011813), (650.041687, 598.413635, 588.398682, -0.000280, 49.702934, -0.000379)),
}

MESH_FOR = {"head_yaw": "neck_yaw"}  # link -> mesh file stem, when they differ


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def inertial_xml(mass, com, I):
    ixx, iyy, izz, ixy, ixz, iyz = (x * I_SCALE for x in I)
    full = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    w = np.linalg.eigvalsh(full)
    assert w.min() > 0, f"non positive-definite inertia {w}"
    return f'<inertial pos="{fmt(com)}" mass="{mass:.6g}" fullinertia="{fmt((ixx, iyy, izz, ixy, ixz, iyz))}"/>'


def convert_meshes(desc_dir):
    import trimesh

    src = os.path.join(desc_dir, "meshes")
    dst = os.path.join(ASSETS, "meshes")
    os.makedirs(dst, exist_ok=True)
    bounds = {}
    for f in sorted(os.listdir(src)):
        if not f.endswith(".dae"):
            continue
        stem = f[:-4]
        mesh = trimesh.load(os.path.join(src, f), force="scene").to_mesh()
        mesh.export(os.path.join(dst, stem + ".stl"))
        bounds[stem] = mesh.bounds.copy()
    return bounds


class Builder:
    def __init__(self, bounds):
        self.bounds = bounds
        self.joints = []  # (name, lo, hi, effort)

    def link(self, name, inertial, pos, joint, children, geoms_extra="", indent=2):
        """joint = (name, axis, lo_deg, hi_deg, effort) or None"""
        pad = "  " * indent
        mesh = MESH_FOR.get(name, name)
        lines = [f'{pad}<body name="{name}" pos="{fmt(pos)}">']
        if joint is not None:
            jn, axis, lo, hi, eff = joint
            lines.append(f'{pad}  <joint name="{jn}" axis="{fmt(axis)}" range="{lo * D2R:.5f} {hi * D2R:.5f}"/>')
            self.joints.append((jn, lo * D2R, hi * D2R, eff))
        lines.append(f"{pad}  {inertial_xml(*inertial)}")
        lines.append(f'{pad}  <geom class="visual" mesh="{mesh}"/>')
        if geoms_extra:
            lines.append(f"{pad}  {geoms_extra}")
        for c in children:
            lines.append(c)
        lines.append(f"{pad}</body>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def leg(self, p, s, indent):
        L = leg_links(s)
        side_lim = lambda r, l: r if s < 0 else l  # noqa: E731
        fb = self.bounds[f"{p}_foot"]
        # Flat sole plate under the foot mesh (1 cm thick).
        sole_lo = np.array([fb[0][0], fb[0][1], fb[0][2]])
        sole_hi = np.array([fb[1][0], fb[1][1], fb[0][2] + 0.01])
        c, h = (sole_lo + sole_hi) / 2, (sole_hi - sole_lo) / 2
        foot_geom = f'<geom name="{p}_sole" class="foot" type="box" pos="{fmt(c)}" size="{fmt(h)}"/>'
        foot_site = f'<site name="{p}_foot_site" pos="{fmt((c[0], c[1], sole_lo[2]))}" size="0.005"/>'
        leg_col = lambda n: f'<geom class="leg_col" mesh="{n}"/>'  # noqa: E731

        foot = self.link(f"{p}_foot", L["foot"], (0, 0, -0.040677),
                         (f"{p}_akx_joint", (1, 0, 0), *side_lim((-38.23875, 37.29375), (-37.29375, 38.23875)), LEG_EFFORT),
                         [], foot_geom + "\n" + "  " * (indent + 6) + foot_site, indent + 6)
        ankle = self.link(f"{p}_ankle", L["ankle"], (0, 0, -0.065),
                          (f"{p}_aky_joint", (0, 1, 0), -61.3575, 24.3, LEG_EFFORT), [foot], "", indent + 5)
        lleg = self.link(f"{p}_lleg", L["lleg"], (0, 0, -0.0652),
                         (f"{p}_kny_joint", (0, 1, 0), -2.0925, 119.64, LEG_EFFORT), [ankle], leg_col(f"{p}_lleg"), indent + 4)
        uleg = self.link(f"{p}_uleg", L["uleg"], (0, s * 0.0128525, -0.034358),
                         (f"{p}_hpy_joint", (0, 1, 0), -91.60, 90.213, LEG_EFFORT), [lleg], leg_col(f"{p}_uleg"), indent + 3)
        uroot = self.link(f"{p}_uleg_root", L["uleg_root"], (0, s * -0.0021, -0.028),
                          (f"{p}_hpx_joint", (1, 0, 0), *side_lim((-25.35, 22.713), (-22.713, 25.35)), LEG_EFFORT), [uleg], "", indent + 2)
        hip = self.link(f"{p}_hip_joint", L["hip_joint"], (0.004225, s * 0.0195245, 0.001412),
                        (f"{p}_hpz_joint", (0, 0, 1), -81.0, 102.43, LEG_EFFORT), [uroot], "", indent + 1)
        return hip

    def arm(self, p, s, indent):
        A = arm_links(s)
        side_lim = lambda r, l: r if s < 0 else l  # noqa: E731
        col = lambda n: f'<geom class="body_col" mesh="{n}"/>'  # noqa: E731
        hand = self.link(f"{p}_hand", A["hand"], (0, s * 0.061348, 0),
                         (f"{p}_wry_joint", (0, 1, 0), -135.0, 118.125, ARM_EFFORT), [], col(f"{p}_hand"), indent + 5)
        larm = self.link(f"{p}_larm", A["larm"], (0, s * 0.024054, 0),
                         (f"{p}_elx_joint", (1, 0, 0), *side_lim((-13.5, 118.96875), (-118.96875, 13.5)), ARM_EFFORT), [hand], col(f"{p}_larm"), indent + 4)
        ljoint = self.link(f"{p}_larm_joint", A["larm_joint"], (0.0088013, s * 0.060007, 0),
                           (f"{p}_ely_joint", (0, 1, 0), -128.68875, 134.325, ARM_EFFORT), [larm], "", indent + 3)
        uarm = self.link(f"{p}_uarm", A["uarm"], (0, s * 0.0189, 0),
                         (f"{p}_shx_joint", (1, 0, 0), *side_lim((-3.0375, 91.56375), (-91.56375, 3.0375)), ARM_EFFORT), [ljoint], col(f"{p}_uarm"), indent + 2)
        ujoint = self.link(f"{p}_uarm_joint", A["uarm_joint"], (0, s * 0.0359, 0.113043),
                           (f"{p}_shy_joint", (0, 1, 0), -108.0, 128.25, ARM_EFFORT), [uarm], "", indent + 1)
        return ujoint

    def head(self, indent):
        B = BODY_LINKS
        head = self.link("head", B["head"], (0, 0, 0.0129),
                         ("head_nx_joint", (1, 0, 0), -21.9375, 28.6875, ARM_EFFORT), [], '<geom class="body_col" mesh="head"/>', indent + 3)
        yaw = self.link("head_yaw", B["head_yaw"], (-0.0103485, 0, 0.011103),
                        ("head_nz_joint", (0, 0, 1), -47.25, 47.25, ARM_EFFORT), [head], "", indent + 2)
        pitch = self.link("neck_pitch", B["neck_pitch"], (0.0038738, 0, 0.132809),
                          ("head_ny_joint", (0, 1, 0), -10.125, 23.625, ARM_EFFORT), [yaw], "", indent + 1)
        return pitch


# Actuator order = servo ID order on the real robot (premaidai_controller joint_config.yaml).
SERVO_ORDER = [
    "r_shy_joint", "head_ny_joint", "l_shy_joint", "head_nz_joint", "r_hpz_joint", "head_nx_joint",
    "l_hpz_joint", "r_shx_joint", "r_hpx_joint", "l_shx_joint", "l_hpx_joint", "r_ely_joint",
    "r_hpy_joint", "l_ely_joint", "l_hpy_joint", "r_elx_joint", "r_kny_joint", "l_elx_joint",
    "l_kny_joint", "r_wry_joint", "r_aky_joint", "l_wry_joint", "l_aky_joint", "r_akx_joint",
    "l_akx_joint",
]


def build(bounds, kp_leg, kp_arm, kv):
    b = Builder(bounds)
    ind = 2
    children = [b.leg("r", -1, ind), b.leg("l", 1, ind), b.arm("r", -1, ind), b.arm("l", 1, ind), b.head(ind)]
    body_inertial = inertial_xml(*BODY_LINKS["body"])
    pad = "  " * ind
    base = "\n".join([
        f'{pad}<body name="base_link" pos="0 0 0.26" childclass="premaid">',
        f'{pad}  <freejoint name="root"/>',
        f"{pad}  {body_inertial}",
        f'{pad}  <geom class="visual" mesh="body"/>',
        f'{pad}  <geom class="body_col" mesh="body"/>',
        f'{pad}  <site name="imu" pos="0 0 0.07" size="0.006"/>',
        *children,
        f"{pad}</body>",
    ])

    assert sorted(SERVO_ORDER) == sorted(j[0] for j in b.joints), "joint set mismatch"
    jinfo = {j[0]: j for j in b.joints}
    acts = []
    for jn in SERVO_ORDER:
        _, lo, hi, eff = jinfo[jn]
        leg = any(k in jn for k in ("hp", "kn", "ak"))
        kp = kp_leg if leg else kp_arm
        acts.append(
            f'    <position name="{jn[:-6]}" joint="{jn}" kp="{kp}" kv="{kv}" ctrlrange="{lo:.5f} {hi:.5f}" '
            f'forcerange="{-eff:.4f} {eff:.4f}"/>'
        )

    # Floor debris: mocap (kinematic) boxes, parked under the floor. premaid_env.py moves them around the
    # robot and sets how far each one sticks up (curriculum). Footprints vary; all are 2 cm thick
    # so any protrusion up to 2 cm can be had by sinking them.
    rng = np.random.default_rng(0)
    debris = "\n".join(
        f'    <body name="debris{i}" mocap="true" pos="0 0 -1"><geom name="debris{i}" type="box" size="{sx:.4f} {sy:.4f} 0.0100" '
        f'rgba="0.55 0.42 0.3 1" contype="1" conaffinity="1" friction="0.8 0.02 0.002" condim="3"/></body>'
        for i, (sx, sy) in enumerate(rng.uniform([0.004, 0.004], [0.03, 0.06], size=(N_DEBRIS, 2)))
    )

    meshes = "\n".join(f'    <mesh name="{m}" file="{m}.stl"/>' for m in sorted(bounds))
    excludes = "\n".join(
        f'    <exclude body1="{p}_{a}" body2="{p}_{c}"/>'
        for p in "rl" for a, c in [("uleg", "ankle"), ("uleg", "foot"), ("lleg", "foot")]
    )

    xml = f"""<!-- AUTO-GENERATED by mujoco_sim/tools/build_mjcf.py. Edit the builder, not this file. -->
<!-- Premaid AI (DMM.make ROBOTS). Kinematics/inertials/meshes: chikuta/premaidai_description, Apache-2.0. -->
<mujoco model="premaidai">
  <compiler angle="radian" meshdir="meshes" autolimits="true" balanceinertia="true"/>
  <option timestep="0.005" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <default>
    <default class="premaid">
      <joint damping="0.05" armature="0.002" frictionloss="0.01"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" rgba="0.92 0.92 0.95 1"/>
      </default>
      <!-- collision bits: 1 = floor, 2 = legs/feet (self-collision between legs) -->
      <default class="body_col">
        <geom type="mesh" contype="1" conaffinity="0" group="3" density="0" rgba="0.8 0.3 0.3 0.4"/>
      </default>
      <default class="leg_col">
        <geom type="mesh" contype="3" conaffinity="2" group="3" density="0" rgba="0.3 0.3 0.8 0.4"/>
      </default>
      <default class="foot">
        <geom contype="3" conaffinity="3" group="3" density="0" friction="0.9 0.02 0.002" condim="4"
              solref="0.01 1" rgba="0.2 0.2 0.2 1"/>
      </default>
    </default>
  </default>

  <asset>
{meshes}
    <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.28 0.32" rgb2="0.32 0.36 0.4" width="512" height="512" mark="edge" markrgb="0.5 0.5 0.5"/>
    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true" reflectance="0.1"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.6 0.75 0.9" rgb2="0.1 0.12 0.15" width="256" height="256"/>
  </asset>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1" friction="0.9 0.02 0.002" condim="3"/>
{debris}
{base}
  </worldbody>

  <contact>
{excludes}
  </contact>

  <actuator>
{chr(10).join(acts)}
  </actuator>

  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
    <velocimeter name="local_linvel" site="imu"/>
    <framequat name="orientation" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""
    return xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--description", required=True, help="path to a clone of chikuta/premaidai_description")
    # Fitted to the RCB-4 step responses of servos 10, 13 and 14 (tools/fit_servo.py). Arm and leg
    # servos responded the same; they differ only in torque limit. The ~220 deg/s slew limit and
    # command/sensor delays live in premaid_env.py.
    ap.add_argument("--kp-leg", type=float, default=43.6)
    ap.add_argument("--kp-arm", type=float, default=43.6)
    ap.add_argument("--kv", type=float, default=0.30)
    args = ap.parse_args()

    bounds = convert_meshes(args.description)
    xml = build(bounds, args.kp_leg, args.kp_arm, args.kv)
    out = os.path.join(ASSETS, "premaidai.xml")
    with open(out, "w") as f:
        f.write(xml)

    import mujoco

    m = mujoco.MjModel.from_xml_path(out)
    print(f"wrote {out}: nbody={m.nbody} njnt={m.njnt} nu={m.nu} robot mass={m.body_subtreemass[m.body("base_link").id]:.3f} kg")


if __name__ == "__main__":
    main()
