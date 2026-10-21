"""Joint <-> RCB-4 servo mapping for the real Premaid AI.

>>> CHECK THIS BEFORE ENABLING TORQUE <<<   (python deploy_policy.py --mode scan / --mode check)

Defaults come from chikuta/premaidai_controller/config/joint_config.yaml (id, direction, offset),
written for the original Premaid board. On an RCB-4 each servo is addressed by (ICS ID 0-17, SIO
group 1 or 2). servo_test.py used ID 10/13/14 on SIO 1, so the IDs below assume SIO 1; if
`--mode scan` finds some servos on SIO 2, or under different IDs, edit this table.

Angle convention (same as the premaidai_controller read path):
    joint_angle = (servo_angle + offset) * direction        servo_angle in degrees, 0 = 7500 counts
"""

# name: (ics_id, sio, direction, offset_deg)
SERVO_MAP = {
    "r_shy": (2, 1, 1, -90),
    "head_ny": (3, 1, 1, 0),
    "l_shy": (4, 1, -1, 90),
    "head_nz": (5, 1, -1, 0),
    "r_hpz": (6, 1, 1, 0),
    "head_nx": (7, 1, -1, 0),
    "l_hpz": (8, 1, 1, 0),
    "r_shx": (9, 1, 1, 0),
    "r_hpx": (10, 1, 1, 0),
    "l_shx": (11, 1, 1, 0),
    "l_hpx": (12, 1, 1, 0),
    "r_ely": (13, 1, 1, 0),
    "r_hpy": (14, 1, 1, 0),
    "l_ely": (15, 1, -1, 0),
    "l_hpy": (16, 1, -1, 0),
    "r_elx": (17, 1, 1, 0),
    # IDs above 17 can't be addressed through an RCB-4 SIO group as-is. If these servos were
    # re-numbered (e.g. moved to SIO 2), put their real (id, sio) here after running --mode scan.
    "r_kny": (18, 1, -1, 0),
    "l_elx": (19, 1, 1, 0),
    "l_kny": (20, 1, 1, 0),
    "r_wry": (21, 1, 1, 90),
    "r_aky": (22, 1, 1, 0),
    "l_wry": (23, 1, -1, -90),
    "l_aky": (24, 1, -1, 0),
    "r_akx": (26, 1, 1, 0),
    "l_akx": (28, 1, 1, 0),
}

# Kondo ICS position counts: 3500..11500 spans -135..+135 deg (premaidai_controller).
# NOTE servo_test.py used 33.333 counts/deg; if knee angles read ~12% off in --mode check, change this.
COUNTS_CENTER = 7500
COUNTS_PER_DEG = 8000.0 / 270.0
COUNTS_MIN, COUNTS_MAX = 3500, 11500
