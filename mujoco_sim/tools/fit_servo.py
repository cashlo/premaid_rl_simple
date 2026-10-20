"""Fit the MuJoCo servo model to Kondo/RCB-4 step-response recordings (servo_test.py output).

    python tools/fit_servo.py --data_dir .. [--plot fit.png]

Servo model (the same one premaid_env.py uses):
    command --(delay)--> reference that moves toward the command at <= vmax --> PD position servo
    (kp, kv) with the URDF torque limit
Measurement model: the RCB-4 position read is refreshed every ~20 ms (sample-and-hold), so the
simulated angle is sampled the same way before comparing.

Each recording is replayed on the full robot model with the torso held in the air, only the tested
servo powered and no contacts, which is how servo_test.py runs it.
"""

from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "..", "assets", "premaidai.xml")

# servo ID -> joint (premaidai_controller/config/joint_config.yaml)
RECORDINGS = {
    10: "r_hpx_joint",
    13: "r_ely_joint",
    14: "r_hpy_joint",
}
TARGET_DEG = 20.0
SIM_DT = 0.001
HOLD_PERIOD = 0.0205  # RCB-4 position refresh seen in the logs


def load_recording(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    t, a = d[:, 0], d[:, 1]
    # the servo settles ~1 deg past the commanded 20 deg (deadband/calibration), so fit the
    # dynamics against where it actually settles
    return t, a, float(np.mean(a[-20:]))


def hanging_model():
    xml = open(XML).read().replace('<freejoint name="root"/>', "")  # torso welded in the air
    xml = xml.replace('<body name="base_link" pos="0 0 0.26"', '<body name="base_link" pos="0 0 0.6"')
    cwd = os.getcwd()
    os.chdir(os.path.dirname(XML))
    try:
        m = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(cwd)
    m.opt.timestep = SIM_DT
    # robot hangs in the air: no contacts (in the real test the other, unpowered leg just gets pushed aside)
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    return m


def simulate(m, joint, params, t_end=1.0, target_deg=TARGET_DEG):
    kp, kv, vmax, delay = params
    d = mujoco.MjData(m)
    jid = m.joint(joint).id
    qadr, dadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    act = [i for i in range(m.nu) if m.actuator_trnid[i, 0] == jid][0]
    # only the tested servo is powered (servo_test.py holds just that one); the rest are limp
    m.actuator_gainprm[:, 0] = 0.0
    m.actuator_biasprm[:, 1] = 0.0
    m.actuator_biasprm[:, 2] = 0.0
    m.actuator_gainprm[act, 0] = kp
    m.actuator_biasprm[act, 1] = -kp
    m.actuator_biasprm[act, 2] = -kv
    mujoco.mj_forward(m, d)
    target = np.deg2rad(target_deg)
    ref = 0.0
    n = int(t_end / SIM_DT)
    ts, qs = np.arange(n) * SIM_DT, np.zeros(n)
    for i in range(n):
        cmd = target if ts[i] >= delay else 0.0
        ref += np.clip(cmd - ref, -vmax * SIM_DT, vmax * SIM_DT)
        d.ctrl[:] = 0.0
        d.ctrl[act] = ref
        mujoco.mj_step(m, d)
        qs[i] = d.qpos[qadr]
    return ts, np.rad2deg(qs)


def measured(ts, qs, t_samples, hold_phase):
    """Sample-and-hold the simulated angle like the RCB-4 position read."""
    t_hold = np.floor((t_samples - hold_phase) / HOLD_PERIOD) * HOLD_PERIOD + hold_phase
    return np.interp(np.clip(t_hold, 0, None), ts, qs)


def loss(x, m, data):
    kp, kv, vmax, delay = x
    if kp <= 0 or kv < 0 or vmax <= 0 or delay < 0:
        return 1e9
    err = 0.0
    for joint, (t, a, final) in data.items():
        ts, qs = simulate(m, joint, x, t_end=0.4, target_deg=final)
        mask = t < 0.4
        best = min(np.mean((measured(ts, qs, t[mask], ph) - a[mask]) ** 2) for ph in np.linspace(0, HOLD_PERIOD, 5, endpoint=False))
        err += best
    return err / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", ".."))
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    data = {}
    for sid, joint in RECORDINGS.items():
        p = os.path.join(args.data_dir, f"rcb4_servo_response_data_id{sid}.csv")
        if os.path.exists(p):
            data[joint] = load_recording(p)
    if not data:
        raise SystemExit(f"no rcb4_servo_response_data_id*.csv in {args.data_dir}")
    m = hanging_model()

    x0 = np.array([6.0, 0.08, 7.5, 0.02])
    print("initial guess  kp=%.2f kv=%.3f vmax=%.2f delay=%.3f  rmse=%.2f deg" % (*x0, np.sqrt(loss(x0, m, data))))
    # coarse grid, then Nelder-Mead
    best = (np.inf, x0)
    for kp in (3, 6, 12, 25, 50):
        for vmax in (4, 6, 8, 11):
            for delay in (0.0, 0.01, 0.02, 0.03):
                x = np.array([kp, 0.08 * np.sqrt(kp / 6), vmax, delay])
                L = loss(x, m, data)
                if L < best[0]:
                    best = (L, x)
    res = minimize(loss, best[1], args=(m, data), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400})
    kp, kv, vmax, delay = res.x
    print("fitted         kp=%.2f kv=%.3f vmax=%.2f rad/s (%.0f deg/s) delay=%.1f ms  rmse=%.2f deg"
          % (kp, kv, vmax, np.rad2deg(vmax), delay * 1e3, np.sqrt(res.fun)))
    for joint, (t, a, final) in data.items():
        ts, qs = simulate(m, joint, res.x, target_deg=final)
        i90 = np.argmax(qs >= 0.9 * final)
        print(f"  {joint}: settles at {final:.1f} deg; sim reaches 90% at {ts[i90]*1e3:.0f} ms, "
              f"overshoot {qs.max()-final:+.1f} deg (real {a.max()-final:+.1f})")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 3.6), sharey=True)
        for ax, (joint, (t, a, final)) in zip(np.atleast_1d(axes), data.items()):
            ts, qs = simulate(m, joint, res.x, t_end=0.5, target_deg=final)
            ts0, qs0 = simulate(m, joint, (6.0 if "_h" in joint or "_k" in joint or "_a" in joint else 3.0, 0.08, 1e3, 0.0),
                                t_end=0.5, target_deg=final)
            ax.plot(t, a, "k.-", lw=1, ms=4, label="real (RCB-4 read)")
            ax.plot(ts, qs, color="#d6604d", lw=2, label="sim, fitted")
            ax.plot(ts0, qs0, color="#4393c3", lw=1.5, ls="--", label="sim, current model")
            ax.axhline(final, color="0.6", lw=0.8, ls=":")
            ax.set_xlim(0, 0.5)
            ax.set_title(joint)
            ax.set_xlabel("time (s)")
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("angle (deg)")
        np.atleast_1d(axes)[0].legend(loc="lower right", fontsize=8)
        fig.suptitle(f"20° step: kp={kp:.1f}, kv={kv:.3f}, vmax={np.rad2deg(vmax):.0f}°/s, delay={delay*1e3:.0f} ms")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
