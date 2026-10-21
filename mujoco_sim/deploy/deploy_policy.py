"""Run the trained walking policy on the real Premaid AI (RCB-4 + backpack IMU).

Work through the modes in order the first time:

  python deploy_policy.py --mode scan               find servos on the RCB-4 (read-only)
  python deploy_policy.py --mode check              torque OFF; live joint angles + IMU. Pose the robot by hand
                                                    and confirm every joint's sign/zero and the gravity vector
  python deploy_policy.py --mode bench              time joint reads / writes; can we hold 50 Hz?
  python deploy_policy.py --mode stand              torque ON: ease into the default crouch and hold (no policy)
  python deploy_policy.py --mode run --checkpoint ../runs/premaid_mujoco/<run>/checkpoints/agent_200000.pt
                                                    the policy, robot in a harness first! keys: w/s fwd/back,
                                                    a/d strafe, q/e turn, space = stop walking, x = quit (go limp)

  python deploy_policy.py --mode getup --getup_checkpoint ../runs/premaid_getup/<run>/checkpoints/agent_N.pt
                                    [--checkpoint <walking agent>]
                                                    from lying down: get-up policy until the robot is upright and
                                                    settled in the crouch, then hand over to the walking policy
                                                    (standing still) or, without --checkpoint, hold the crouch

  add --sim to any of stand/run/getup to drive a MuJoCo copy of the robot instead (same servo model and
  delays as training) -- a full end-to-end test of this script without hardware.

Ports: --rcb4 COM5 (115200 default; 1250000 recommended, see bench) --imu COM4 (1 Mbps binary stream).
Ctrl+C or a fall (torso tilted > ~45 deg) makes every servo go limp.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402

from premaid_env import ASSET_XML, DEFAULT_POSE, LEG_JOINTS, SERVO_ORDER, WalkCfg  # noqa: E402

CTRL_DT = WalkCfg().control_dt  # 0.02 s, 50 Hz like training
GAIT_PERIOD = WalkCfg().gait_period
FALL_TILT = 0.7  # |gravity_xy| in body frame (~44 deg)


# ---------------------------------------------------------------------------------- policy side
class Policy:
    """Builds exactly the 86-value observation premaid_env.PremaidWalkEnv gives the actor."""

    def __init__(self, path, task="walk"):
        import torch

        self.torch = torch
        try:
            self.net = torch.jit.load(path, map_location="cpu")  # play_mujoco.py --export output
        except Exception:
            from play_mujoco import DeterministicPolicy, load_agent

            self.net = DeterministicPolicy(load_agent(path)).eval()
        if task == "getup":
            from getup_env import GetUpCfg

            cfg = GetUpCfg()
        else:
            cfg = WalkCfg()
        self.default_q = np.array([DEFAULT_POSE.get(n, 0.0) for n in SERVO_ORDER])
        self.scale = np.array([cfg.action_scale_leg if n in LEG_JOINTS else cfg.action_scale_upper for n in SERVO_ORDER])
        self.clip = cfg.action_clip
        m = mujoco.MjModel.from_xml_path(ASSET_XML)
        self.lo = np.array([m.actuator(n).ctrlrange[0] for n in SERVO_ORDER])
        self.hi = np.array([m.actuator(n).ctrlrange[1] for n in SERVO_ORDER])
        self.reset()

    def reset(self):
        self.last_action = np.zeros(25)
        self.q_hist = collections.deque(maxlen=3)
        self.steps = 0

    def observe(self, s, cmd):
        q = s["q"]
        self.q_hist.append(q)
        if len(self.q_hist) >= 3:  # reads refresh every ~20 ms, so difference over 2 steps
            dq = (self.q_hist[-1] - self.q_hist[0]) / (2 * CTRL_DT)
        else:
            dq = np.zeros(25)
        cmd = np.asarray(cmd, dtype=float)
        if np.any(cmd != 0.0):
            ph = (self.steps * CTRL_DT / GAIT_PERIOD) % 1.0
            clock = np.array([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])
        else:
            clock = np.zeros(2)
        return np.concatenate([
            s["gyro"] * 0.25, s["grav"], cmd * np.array([4.0, 4.0, 1.0]),
            q - self.default_q, dq * 0.05, self.last_action, clock,
        ]).astype(np.float32)

    def act(self, obs):
        with self.torch.no_grad():
            a = self.net(self.torch.as_tensor(obs)[None])[0].numpy().astype(np.float64)
        a = np.clip(a, -self.clip, self.clip)
        self.last_action = a
        self.steps += 1
        return np.clip(self.default_q + self.scale * a, self.lo, self.hi)


# ---------------------------------------------------------------------------------- helpers
class Keyboard:
    def __init__(self):
        try:
            import msvcrt

            self.m = msvcrt
        except ImportError:
            self.m = None

    def keys(self):
        out = []
        while self.m is not None and self.m.kbhit():
            out.append(self.m.getwch().lower())
        return out


def make_robot(args, task="walk"):
    from robot_io import RealRobot, SimRobot

    if args.sim:
        return SimRobot(randomize=args.sim_randomize, render=args.render, task=task, seed=args.seed)
    return RealRobot(args.rcb4, args.rcb4_baud, args.imu, args.imu_baud, args.lib_dir)


def tick(t_next, realtime):
    if not realtime:
        return time.perf_counter()
    now = time.perf_counter()
    if t_next > now:
        time.sleep(t_next - now)
    return t_next


def ease_to(robot, q_from, q_to, seconds, realtime=True):
    n = max(1, int(seconds / CTRL_DT))
    t = time.perf_counter()
    for i in range(1, n + 1):
        a = 0.5 - 0.5 * np.cos(np.pi * i / n)
        robot.write(q_from + a * (q_to - q_from))
        t = tick(t + CTRL_DT, realtime)


def confirm(msg):
    return input(f"{msg} Type 'yes' to continue: ").strip().lower() == "yes"


# ---------------------------------------------------------------------------------- modes
def mode_scan(args):
    import robot_io

    for p in filter(None, [args.lib_dir, os.path.join(HERE, "..", "..")]):
        sys.path.insert(0, os.path.abspath(p))
    from Rcb4BaseLib import Rcb4BaseLib

    r = Rcb4BaseLib()
    if not r.open(args.rcb4, args.rcb4_baud, 0.1):
        raise SystemExit(f"could not open RCB-4 on {args.rcb4}")
    print("ID/SIO pairs that report a position (read-only, torque unchanged):")
    found = []
    for sio in (1, 2):
        for sid in range(18):
            ok, v = r.getSinglePos(sid, sio)
            if ok and v > 0:
                found.append((sid, sio, v))
                print(f"  ID {sid:2d}  SIO {sio}  counts {v:5d}  ({(v - 7500) / robot_io.COUNTS_PER_DEG:+6.1f} deg)")
    r.close()
    want = {(i, s) for i, s, _, _ in robot_io.SERVO_MAP.values()}
    have = {(i, s) for i, s, _ in found}
    print(f"\n{len(found)} servos found; servo_map.py expects {len(want)}.")
    missing = sorted(want - have)
    if missing:
        names = [n for n, v in robot_io.SERVO_MAP.items() if (v[0], v[1]) in missing]
        print(f"servo_map.py entries NOT found: {names}\n-> edit deploy/servo_map.py with the right (id, sio).")


def mode_check(args):
    robot = make_robot(args)
    pol_default = np.array([DEFAULT_POSE.get(n, 0.0) for n in SERVO_ORDER])
    if not args.sim:
        robot.free()
    print("Torque is OFF. Move joints by hand: each angle should follow the simulator's convention.\n"
          "Robot upright -> gravity ~ (0, 0, -1). Tilt it forward -> gravity x goes positive; lean it to its\n"
          "left -> gravity y goes positive. Ctrl+C to stop.\n")
    try:
        while True:
            s = robot.read()
            lines = [f"{n:8s} {np.degrees(q):+7.1f} deg   (default {np.degrees(d):+6.1f})"
                     for n, q, d in zip(SERVO_ORDER, s["q"], pol_default)]
            half = (len(lines) + 1) // 2
            print("\033[2J\033[H", end="")
            for a, b in zip(lines[:half], lines[half:] + [""]):
                print(f"{a:44s}{b}")
            if s["grav"] is not None:
                print(f"\ngravity (body) {np.round(s['grav'], 2)}   gyro (rad/s) {np.round(s['gyro'], 2)}")
            else:
                print("\nIMU: no packets yet")
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


def mode_bench(args):
    robot = make_robot(args)
    try:
        if not args.sim:
            bulk, single, ok = robot.verify_bulk()
            print(f"bulk RAM read matches single reads: {ok}")
            if not ok:
                for n, b, s in zip(SERVO_ORDER, bulk, single):
                    print(f"  {n:8s} bulk {b:6d} single {s:6d}")
                print("-> joint reads would be wrong; send me this output.")
            print(f"RAM reads per step: {len(robot.reads)}  {robot.reads}")
        tr, tw = [], []
        for _ in range(50):
            t0 = time.perf_counter()
            robot.read()
            t1 = time.perf_counter()
            if not args.sim:
                robot.free()  # same packet size as a position write, keeps torque off
            tw.append(time.perf_counter() - t1)
            tr.append(t1 - t0)
        tr, tw = np.array(tr) * 1e3, np.array(tw) * 1e3
        total = np.median(tr) + np.median(tw)
        print(f"read  {np.median(tr):6.1f} ms (p90 {np.percentile(tr, 90):.1f})")
        print(f"write {np.median(tw):6.1f} ms (p90 {np.percentile(tw, 90):.1f})")
        print(f"=> {total:.1f} ms of a {CTRL_DT * 1e3:.0f} ms step: " + (
            "OK for 50 Hz" if total < CTRL_DT * 1e3 * 0.8 else
            "TOO SLOW for 50 Hz. Set the RCB-4 COM speed to 1.25 Mbps (HeartToHeart) and use --rcb4_baud 1250000, "
            "and set the USB-serial latency timer to 1 ms."))
        if not args.sim and robot.imu is not None:
            n0 = robot.imu.count
            time.sleep(1.0)
            print(f"IMU packets/s: {robot.imu.count - n0}")
    finally:
        robot.close()


def mode_stand(args):
    robot = make_robot(args)
    default_q = np.array([DEFAULT_POSE.get(n, 0.0) for n in SERVO_ORDER])
    if not args.sim and not confirm("Torque will turn ON and the robot will move to its crouch pose."):
        return
    try:
        q0 = robot.read()["q"]
        ease_to(robot, q0, default_q, 2.0, realtime=not args.sim)
        print("Holding the default pose. Ctrl+C to go limp.")
        t = time.perf_counter()
        for _ in range(int(args.seconds / CTRL_DT)):
            robot.write(default_q)
            t = tick(t + CTRL_DT, not args.sim)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


def mode_run(args):
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for --mode run")
    pol = Policy(args.checkpoint)
    robot = make_robot(args)
    if not args.sim and not confirm("Policy control with torque ON. Is the robot in a harness / held?"):
        robot.close()
        return
    kb = Keyboard()
    cmd = np.array(args.cmd, dtype=float)
    realtime = (not args.sim) or args.render
    log = collections.defaultdict(list)
    try:
        s = robot.read()
        if s["grav"] is None:
            raise RuntimeError("no IMU data -- check --imu port")
        ease_to(robot, s["q"], pol.default_q, 2.0, realtime=realtime)
        pol.reset()
        print("Running. w/s a/d q/e change the command, space stops, x quits.")
        t = time.perf_counter()
        x0 = robot.env.data.qpos[:2].copy() if args.sim else None
        for _ in range(int(args.seconds / CTRL_DT)):
            t_loop = time.perf_counter()
            for k in kb.keys():
                if k == "x":
                    raise KeyboardInterrupt
                cmd += {"w": [0.03, 0, 0], "s": [-0.03, 0, 0], "a": [0, 0.03, 0], "d": [0, -0.03, 0],
                        "q": [0, 0, 0.15], "e": [0, 0, -0.15]}.get(k, [0, 0, 0])
                if k == " ":
                    cmd[:] = 0.0
                cmd = np.clip(cmd, [-0.10, -0.08, -0.6], [0.20, 0.08, 0.6])
                print(f"command vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} yaw={cmd[2]:+.2f}")
            s = robot.read()
            if np.linalg.norm(s["grav"][:2]) > FALL_TILT:
                print("FALL detected -> limp")
                log["fell"].append(1)
                break
            target = pol.act(pol.observe(s, cmd))
            robot.write(target)
            log["loop_ms"].append((time.perf_counter() - t_loop) * 1e3)
            if s.get("t_imu") is not None:
                log["imu_age_ms"].append((s["t"] - s["t_imu"]) * 1e3)
            t = tick(t + CTRL_DT, realtime)
        if args.sim:
            dist = robot.env.data.qpos[:2] - x0
            print(f"[sim] travelled {np.linalg.norm(dist):.2f} m in {pol.steps * CTRL_DT:.1f} s "
                  f"(commanded {cmd[0]:+.2f} m/s), fell: {bool(log['fell'])}")
    except KeyboardInterrupt:
        print("stopping -> limp")
    finally:
        robot.close()
        if log["loop_ms"]:
            lm = np.array(log["loop_ms"])
            print(f"loop time median {np.median(lm):.1f} ms, p95 {np.percentile(lm, 95):.1f} ms, "
                  f"over budget {np.mean(lm > CTRL_DT * 1e3) * 100:.0f}% of steps")


def mode_getup(args):
    if not args.getup_checkpoint:
        raise SystemExit("--getup_checkpoint is required for --mode getup")
    up = Policy(args.getup_checkpoint, task="getup")
    walk = Policy(args.checkpoint) if args.checkpoint else None
    robot = make_robot(args, task="getup")
    if not args.sim and not confirm("Get-up with torque ON. Clear space around the robot, hand near the power switch?"):
        robot.close()
        return
    realtime = (not args.sim) or args.render
    zero = np.zeros(3)
    settled, phase, t_up = 0, "getup", None
    t = time.perf_counter()
    try:
        s = robot.read()
        if s["grav"] is None:
            raise RuntimeError("no IMU data -- check --imu port")
        robot.write(s["q"])  # torque on, holding the current pose
        up.reset()
        for step in range(int(args.seconds / CTRL_DT)):
            s = robot.read()
            if phase == "getup":
                robot.write(up.act(up.observe(s, zero)))
                tilt = np.linalg.norm(s["grav"][:2])
                pose_err = np.linalg.norm(s["q"] - up.default_q)
                ok = tilt < 0.3 and np.linalg.norm(s["gyro"]) < 0.6 and pose_err < 0.35
                settled = settled + 1 if ok else 0
                if settled >= int(0.5 / CTRL_DT):
                    t_up = step * CTRL_DT
                    print(f"upright and settled after {t_up:.1f} s -> " + ("walking policy (standing)" if walk else "holding crouch"))
                    phase = "walk" if walk else "hold"
                    if walk:
                        walk.reset()
            elif phase == "walk":
                if np.linalg.norm(s["grav"][:2]) > FALL_TILT:
                    print("fell again -> back to get-up")
                    phase, settled = "getup", 0
                    up.reset()
                    continue
                robot.write(walk.act(walk.observe(s, zero)))
            else:
                robot.write(up.default_q)
            t = tick(t + CTRL_DT, realtime)
            if args.sim and phase != "getup" and step * CTRL_DT > t_up + args.hold_seconds:
                break
        if args.sim:
            print(f"[sim] {'stood up in %.1f s' % t_up if t_up is not None else 'did NOT stand up'}; "
                  f"final tilt {np.degrees(np.arcsin(min(1, np.linalg.norm(robot.read()['grav'][:2])))):.0f} deg")
    except KeyboardInterrupt:
        print("stopping -> limp")
    finally:
        robot.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["scan", "check", "bench", "stand", "run", "getup"])
    ap.add_argument("--checkpoint", help="walking policy: skrl .pt checkpoint or TorchScript from play_mujoco.py --export")
    ap.add_argument("--getup_checkpoint", help="get-up policy checkpoint (runs/premaid_getup/...)")
    ap.add_argument("--hold_seconds", type=float, default=3.0, help="sim getup: keep running this long after standing")
    ap.add_argument("--seed", type=int, default=0, help="sim: which random start (lying pose for getup)")
    ap.add_argument("--rcb4", default="COM5")
    ap.add_argument("--rcb4_baud", type=int, default=115200)
    ap.add_argument("--imu", default="COM4")
    ap.add_argument("--imu_baud", type=int, default=1_000_000)
    ap.add_argument("--lib_dir", default=None, help="folder containing Rcb4BaseLib.py")
    ap.add_argument("--cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "YAW"))
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--sim", action="store_true", help="use the MuJoCo robot instead of hardware")
    ap.add_argument("--sim_randomize", action="store_true", help="sim: random delays/servo params/pushes")
    ap.add_argument("--render", action="store_true", help="sim: show a viewer (runs in real time)")
    args = ap.parse_args()
    {"scan": mode_scan, "check": mode_check, "bench": mode_bench, "stand": mode_stand, "run": mode_run,
     "getup": mode_getup}[args.mode](args)


if __name__ == "__main__":
    main()
