"""Hardware access for the real Premaid AI (RCB-4 + backpack IMU) and a MuJoCo stand-in for testing.

Both backends expose the same interface, in the policy's joint order (premaid_env.SERVO_ORDER),
angles in radians in the simulator's convention:

    read()  -> dict(q=(25,), gyro=(3,) rad/s body frame, grav=(3,) unit vector body frame, t=float)
    write(q_target)     send position targets
    free()              all servos limp
    close()
"""

from __future__ import annotations

import math
import os
import struct
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # mujoco_sim/

from premaid_env import SERVO_ORDER  # noqa: E402
from servo_map import COUNTS_CENTER, COUNTS_MAX, COUNTS_MIN, COUNTS_PER_DEG, SERVO_MAP  # noqa: E402

ICS_RAM_BASE = 0x0090  # RCB-4 RAM: 36 ICS device slots x 20 bytes (KRR = slot 35 -> button data at 0x0350)
ICS_SLOT_SIZE = 20
ICS_POS_OFFSET = 4  # "measured position" field inside a slot
MAX_READ = 240  # bytes per RAM read (reply length byte must stay < 256)


def euler_to_gravity(roll, pitch, yaw):
    """Gravity direction in the sensor frame for R = Rz(yaw) Ry(pitch) Rx(roll) (Isaac's quat_from_euler_xyz)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    return R.T @ np.array([0.0, 0.0, -1.0])


def imu_to_body(v):
    """Backpack-mounted IMU axes -> robot body axes (x forward, y left, z up). From imu_noise_sample.py."""
    return np.array([-v[2], -v[0], v[1]])


# ======================================================================================
# Real robot
# ======================================================================================
class ImuReader(threading.Thread):
    """Reads the ESP32 dongle's binary stream: 0xAA 0xBB + '<ffffffBBBB'
    (gyro x,y,z deg/s, pitch, roll, yaw deg, 4 status bytes). Keeps only the newest packet."""

    PACKET = struct.Struct("<ffffffBBBB")

    def __init__(self, port, baud):
        super().__init__(daemon=True)
        import serial

        self.ser = serial.Serial(port, baud, timeout=0.01)
        self.latest = None
        self.count = 0
        self._stop = False

    def run(self):
        buf = bytearray()
        n = self.PACKET.size
        while not self._stop:
            buf += self.ser.read(max(1, self.ser.in_waiting))
            while True:
                i = buf.find(b"\xaa\xbb")
                if i < 0 or len(buf) < i + 2 + n:
                    if i > 0:
                        del buf[:i]
                    break
                d = self.PACKET.unpack_from(buf, i + 2)
                del buf[: i + 2 + n]
                gyro_deg = d[0:3]
                pitch, roll, _yaw = (math.radians(x) for x in d[3:6])
                g_imu = euler_to_gravity(roll, pitch, 0.0)  # yaw doesn't affect gravity
                gyro_imu = np.radians(gyro_deg)
                self.latest = (imu_to_body(gyro_imu), imu_to_body(g_imu), time.perf_counter())
                self.count += 1

    def close(self):
        self._stop = True
        self.join(timeout=0.5)
        self.ser.close()


class RealRobot:
    def __init__(self, rcb4_port, rcb4_baud, imu_port, imu_baud, lib_dir=None):
        for p in filter(None, [lib_dir, os.path.join(HERE, "..", ".."), HERE]):
            if os.path.exists(os.path.join(p, "Rcb4BaseLib.py")):
                sys.path.insert(0, os.path.abspath(p))
                break
        from Rcb4BaseLib import Rcb4BaseLib

        self.lib = Rcb4BaseLib
        self.rcb4 = Rcb4BaseLib()
        if not self.rcb4.open(rcb4_port, rcb4_baud, 0.1):
            raise RuntimeError(f"could not open RCB-4 on {rcb4_port} @ {rcb4_baud}")
        self.map = [SERVO_MAP[n] for n in SERVO_ORDER]
        bad = [n for n, (i, s, _, _) in zip(SERVO_ORDER, self.map) if not (0 <= i <= 17 and s in (1, 2))]
        if bad:
            raise ValueError(f"servo_map.py: these joints have IDs an RCB-4 can't address (ID 0-17, SIO 1/2): {bad}. "
                             f"Run --mode scan and fix the table.")
        self.ics = np.array([i * 2 + (s - 1) for i, s, _, _ in self.map])
        self.dirs = np.array([d for _, _, d, _ in self.map], dtype=float)
        self.offs = np.array([o for _, _, _, o in self.map], dtype=float)
        self._plan_reads()
        self.imu = None
        if imu_port:
            self.imu = ImuReader(imu_port, imu_baud)
            self.imu.start()

    # ---------------- conversions
    def joint_to_counts(self, q):
        servo_deg = np.degrees(q) * self.dirs - self.offs
        return np.clip(np.round(COUNTS_CENTER + servo_deg * COUNTS_PER_DEG), COUNTS_MIN, COUNTS_MAX).astype(int)

    def counts_to_joint(self, c):
        servo_deg = (np.asarray(c, dtype=float) - COUNTS_CENTER) / COUNTS_PER_DEG
        return np.radians((servo_deg + self.offs) * self.dirs)

    # ---------------- joint reads
    def _plan_reads(self):
        """Group servo slots into as few contiguous RAM reads as possible."""
        slots = sorted(set(self.ics.tolist()))
        self.reads, cur = [], [slots[0]]
        for s in slots[1:]:
            if (s - cur[0] + 1) * ICS_SLOT_SIZE <= MAX_READ:
                cur.append(s)
            else:
                self.reads.append((cur[0], cur[-1]))
                cur = [s]
        self.reads.append((cur[0], cur[-1]))  # (first_slot, last_slot) pairs

    def read_counts_bulk(self):
        counts = {}
        for a, b in self.reads:
            addr = ICS_RAM_BASE + a * ICS_SLOT_SIZE
            size = (b - a + 1) * ICS_SLOT_SIZE
            ok, data = self.rcb4.moveRamToComCmdSynchronize(addr, size)
            if not ok:
                raise IOError(f"RCB-4 RAM read failed @0x{addr:04x}")
            for s in range(a, b + 1):
                off = (s - a) * ICS_SLOT_SIZE + ICS_POS_OFFSET
                counts[s] = struct.unpack_from("<h", data, off)[0]
        return np.array([counts[s] for s in self.ics])

    def read_counts_single(self):
        out = []
        for i, s, _, _ in self.map:
            ok, v = self.rcb4.getSinglePos(i, s)
            out.append(v if ok else -1)
        return np.array(out)

    def verify_bulk(self):
        bulk, single = self.read_counts_bulk(), self.read_counts_single()
        return bulk, single, bool(np.all(np.abs(bulk - single) <= 20) and np.all(single > 0))

    # ---------------- interface
    def read(self, joints=True):
        q = self.counts_to_joint(self.read_counts_bulk()) if joints else None
        gyro = grav = None
        t_imu = None
        if self.imu is not None and self.imu.latest is not None:
            gyro, grav, t_imu = self.imu.latest
        return dict(q=q, gyro=gyro, grav=grav, t=time.perf_counter(), t_imu=t_imu)

    def _send(self, counts):
        D = self.lib.ServoData
        datas = [D(i, s, int(c)) for (i, s, _, _), c in zip(self.map, counts)]
        if not self.rcb4.setServoPos(datas, 1):
            raise IOError("RCB-4 setServoPos failed")

    def write(self, q_target):
        self._send(self.joint_to_counts(q_target))

    def free(self):
        self._send([0x8000] * len(self.map))

    def close(self):
        try:
            self.free()
        finally:
            if self.imu is not None:
                self.imu.close()
            self.rcb4.close()


# ======================================================================================
# MuJoCo stand-in (same servo model, delays and staleness as training)
# ======================================================================================
class SimRobot:
    def __init__(self, randomize=False, seed=0, render=False):
        from premaid_env import PremaidWalkEnv, WalkCfg

        cfg = WalkCfg(randomize=randomize, obs_noise=False, debris=False)
        self.env = PremaidWalkEnv(cfg, seed=seed)
        self.env.manual_commands = True
        self.env.commands[:] = 0.0
        self.env.reset()
        self.dt = self.env.dt
        self.viewer = None
        if render:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)

    def read(self, joints=True):
        e, d = self.env, self.env.data
        q = e.joint_hist[-1 - e.obs_delay][0].copy()  # stale like an RCB-4 read
        R = d.xmat[e.base_id].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = d.sensordata[e._sid["gyro"]: e._sid["gyro"] + 3].copy()
        return dict(q=q, gyro=gyro, grav=grav, t=time.perf_counter(), t_imu=time.perf_counter())

    def write(self, q_target):
        e = self.env
        action = (np.asarray(q_target) - e.default_q) / e.action_scale
        e.step(action)
        if self.viewer is not None:
            self.viewer.sync()

    def free(self):
        pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
