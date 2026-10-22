"""Premaid AI as an mjlab entity (robot-only MJCF from make_robot_xml.py)."""

from __future__ import annotations

import os
import sys

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "assets", "premaid_mjlab.xml")
sys.path.insert(0, os.path.join(HERE, "..", "mujoco_sim"))

from premaid_env import DEFAULT_POSE, LEG_JOINTS, SERVO_ORDER  # noqa: E402  (shared with the CPU version)

JOINT_NAMES = [f"{n}_joint" for n in SERVO_ORDER]  # policy / servo order
LEG_REGEX = ".*_(hp[xyz]|kny|ak[xy])_joint"
UPPER_REGEX = "(.*_(sh[xy]|el[xy]|wry)_joint|head_.*)"

# Servo model fitted to the RCB-4 step responses (mujoco_sim/tools/fit_servo.py).
SERVO_KP = 43.6
SERVO_KD = 0.30
SERVO_ARMATURE = 0.002
LEG_EFFORT = 1.3712
UPPER_EFFORT = 0.9114


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(XML)


def premaid_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),
            joint_pos={f"{n}_joint": DEFAULT_POSE.get(n, 0.0) for n in SERVO_ORDER},
        ),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(LEG_REGEX,), stiffness=SERVO_KP, damping=SERVO_KD,
                    effort_limit=LEG_EFFORT, armature=SERVO_ARMATURE,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(UPPER_REGEX,), stiffness=SERVO_KP, damping=SERVO_KD,
                    effort_limit=UPPER_EFFORT, armature=SERVO_ARMATURE,
                ),
            ),
            soft_joint_pos_limit_factor=0.95,
        ),
    )


__all__ = ["premaid_robot_cfg", "JOINT_NAMES", "SERVO_ORDER", "DEFAULT_POSE", "LEG_JOINTS"]
