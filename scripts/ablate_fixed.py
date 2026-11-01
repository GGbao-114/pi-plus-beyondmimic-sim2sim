#!/usr/bin/env python
"""在「已修复的自动生成 MJCF」上重做消融实验。

之前的消融是在坏模型（无 armature）上做的，结论作废。
本脚本用能正常跟踪动作的模型，逐项测试承载/接触相关参数。

用法:
  python ablate_fixed.py --motion_file <npz> --xml_path <pi_auto.xml> \
      --policy_path <onnx> --duration 8
"""
import argparse
import numpy as np
import mujoco
import onnxruntime

SIM_DT = 0.002
CTRL_DEC = 10
ARMATURE_LEG = 0.044277
ARMATURE_ARM = 0.008419

BFS_JOINTS = [
    "l_hip_pitch_joint", "l_shoulder_pitch_joint", "r_hip_pitch_joint", "r_shoulder_pitch_joint",
    "l_hip_roll_joint", "l_shoulder_roll_joint", "r_hip_roll_joint", "r_shoulder_roll_joint",
    "l_thigh_joint", "l_upper_arm_joint", "r_thigh_joint", "r_upper_arm_joint",
    "l_calf_joint", "l_elbow_joint", "r_calf_joint", "r_elbow_joint",
    "l_ankle_pitch_joint", "l_wrist_joint", "r_ankle_pitch_joint", "r_wrist_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
]
ARM_SUF = ("_shoulder_pitch_joint", "_shoulder_roll_joint",
           "_upper_arm_joint", "_elbow_joint", "_wrist_joint")


def dpos_of(n):
    if n.endswith("_hip_pitch_joint"): return -0.25
    if n.endswith("_calf_joint"): return 0.65
    if n.endswith("_ankle_pitch_joint"): return -0.4
    return 0.0


def gains(n):
    return (30.0, 0.6) if n.endswith(ARM_SUF) else (80.0, 1.1)


def scale_of(n):
    return 0.25 * (20.0 if n.endswith(ARM_SUF) else 30.0) / gains(n)[0]


def build(xml, setup):
    m = mujoco.MjModel.from_xml_path(xml)
    m.opt.timestep = SIM_DT
    m.actuator_gainprm[:, 0] = 1.0
    m.actuator_biasprm[:, :] = 0.0

    # 关节名
    jname = {}
    for jid in range(m.njnt):
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if nm:
            jname[nm] = jid

    # armature（必须）
    for nm, jid in jname.items():
        m.dof_armature[m.jnt_dofadr[jid]] = ARMATURE_ARM if nm.endswith(ARM_SUF) else ARMATURE_LEG
        if m.dof_damping[m.jnt_dofadr[jid]] < 0.01:
            m.dof_damping[m.jnt_dofadr[jid]] = 0.05

    # 力矩上限
    for i in range(m.nu):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0])) or ""
        lim = 20.0 if nm.endswith(ARM_SUF) else 30.0
        m.actuator_ctrlrange[i] = [-lim, lim]

    if setup:
        setup(m)
    return m, jname


def run(xml, jp, jv, bp, bq, sess, duration, setup=None, kp_scale=1.0):
    m, jname = build(xml, setup)
    d = mujoco.MjData(m)

    act_joint = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0]))
                 for i in range(m.nu)]
    valid = [i for i in range(m.nu) if act_joint[i] in jname]
    a2b = np.array([BFS_JOINTS.index(act_joint[i]) for i in valid])
    aq = np.array([int(m.jnt_qposadr[jname[act_joint[i]]]) for i in valid])
    ad = np.array([int(m.jnt_dofadr[jname[act_joint[i]]]) for i in valid])
    kp_a = np.array([gains(BFS_JOINTS[b])[0] for b in a2b]) * kp_scale
    kd_a = np.array([gains(BFS_JOINTS[b])[1] for b in a2b])
    qadr = {n: int(m.jnt_qposadr[jname[n]]) for n in BFS_JOINTS}
    dpos_b = np.array([dpos_of(j) for j in BFS_JOINTS])
    sc_b = np.array([scale_of(j) for j in BFS_JOINTS])

    use_sensor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity") >= 0

    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.qpos[0:3] = bp[0, 0]
    q0 = bq[0, 0].copy()
    d.qpos[3:7] = q0 / np.linalg.norm(q0)
    for n in BFS_JOINTS:
        d.qpos[qadr[n]] = jp[0][BFS_JOINTS.index(n)]
    mujoco.mj_forward(m, d)

    target_b = jp[0].copy()
    act_buf = np.zeros(22, dtype=np.float32)
    n_frames = min(len(jp), len(bp))
    n_ctrl = int(duration / (SIM_DT * CTRL_DEC))

    fall_t, min_h, errs, hs, sat = None, 9.9, [], [], 0
    for step in range(n_ctrl * CTRL_DEC):
        tau = np.zeros(m.nu)
        tau[valid] = (target_b[a2b] - d.qpos[aq]) * kp_a - d.qvel[ad] * kd_a
        lim = np.array([m.actuator_ctrlrange[i][1] for i in valid])
        sat += int(np.sum(np.abs(tau[valid]) >= 0.98 * lim))
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        if (step + 1) % CTRL_DEC == 0:
            t = (step + 1) // CTRL_DEC
            fi = min(t, n_frames - 1)
            h = d.qpos[2]
            min_h = min(min_h, h)
            hs.append(h)
            if fall_t is None and h < 0.15:
                fall_t = t * SIM_DT * CTRL_DEC

            q_b = np.array([d.qpos[qadr[n]] for n in BFS_JOINTS])
            errs.append(np.abs(q_b - jp[fi]).mean())

            quat = d.qpos[3:7].copy()
            omega = d.sensor("angular-velocity").data.copy() if use_sensor else d.qvel[3:6].copy()
            qq = bq[fi, 0]
            w1, x1, y1, z1 = quat[0], -quat[1], -quat[2], -quat[3]
            w2, x2, y2, z2 = qq
            qm = np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                           w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
            qm /= np.linalg.norm(qm)
            r, i, j, k = qm
            s2 = 2.0 / (qm*qm).sum()
            mat = np.array([
                [1-s2*(j*j+k*k), s2*(i*j-k*r), s2*(i*k+j*r)],
                [s2*(i*j+k*r), 1-s2*(i*i+k*k), s2*(j*k-i*r)],
                [s2*(i*k-j*r), s2*(j*k+i*r), 1-s2*(i*i+j*j)]])
            obs = np.zeros(119, dtype=np.float32)
            obs[0:44] = np.concatenate((jp[fi], jv[fi]))
            obs[44:50] = mat[:, :2].reshape(6)
            obs[50:53] = omega
            obs[53:75] = q_b - dpos_b
            obs[75:97] = np.array([d.qvel[int(m.jnt_dofadr[jname[n]])] for n in BFS_JOINTS])
            obs[97:119] = act_buf
            out = sess.run(["actions"], {
                "obs": obs.reshape(1, -1),
                "time_step": np.array([[float(fi)]], dtype=np.float32)})[0]
            act_b = np.asarray(out).reshape(-1).astype(np.float32)
            act_buf = act_b.copy()
            target_b = act_b * sc_b + dpos_b

    return dict(fall=fall_t if fall_t is not None else duration, min_h=min_h,
                jerr=float(np.mean(errs)), hs=np.array(hs),
                sat_pct=100.0 * sat / max(len(valid) * n_ctrl, 1))


# ---- 消融设置 ----
def s_solref_soft(m):   m.geom_solref[:, 0] = 0.004
def s_solref_hard(m):   m.geom_solref[:, 0] = 0.001
def s_fric15(m):        m.geom_friction[:, 0] = 1.5
def s_fric25(m):        m.geom_friction[:, 0] = 2.5
def s_dt1(m):           m.opt.timestep = 0.001
def s_margin(m):        m.geom_margin[:] = 0.002
def s_warmstart(m):     m.opt.iterations = 100; m.opt.ls_iterations = 50
def s_combo(m):
    m.geom_solref[:, 0] = 0.004
    m.geom_friction[:, 0] = 1.5
    m.geom_margin[:] = 0.002
    m.opt.iterations = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_file", required=True)
    ap.add_argument("--xml_path", required=True)
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--duration", type=float, default=8.0)
    args = ap.parse_args()

    mo = np.load(args.motion_file)
    jp, jv, bp, bq = mo["joint_pos"], mo["joint_vel"], mo["body_pos_w"], mo["body_quat_w"]
    sess = onnxruntime.InferenceSession(args.policy_path, providers=["CPUExecutionProvider"])

    print("=" * 86)
    print(" 在「已修复模型」上的消融实验")
    print("=" * 86)
    print(f" 时长上限 {args.duration}s    Isaac 参考 9.80s")
    print("=" * 86)
    print(f"{'配置':<38s} {'撑住':>8s} {'最低高度':>10s} {'关节误差':>10s} {'饱和%':>8s}")
    print("-" * 86)

    configs = [
        ("(0) 基线（已修复模型）", None, 1.0),
        ("(1) 接触更硬 solref 0.002", s_solref_hard, 1.0),
        ("(2) 接触更软 solref 0.004", s_solref_soft, 1.0),
        ("(3) 摩擦 1.5", s_fric15, 1.0),
        ("(4) 摩擦 2.5", s_fric25, 1.0),
        ("(5) 步长 0.001 (1000Hz)", s_dt1, 1.0),
        ("(6) 接触 margin 2mm", s_margin, 1.0),
        ("(7) 求解迭代 100", s_warmstart, 1.0),
        ("(8) kp x1.5", None, 1.5),
        ("(9) 组合 (硬接触+摩擦1.5+margin)", s_combo, 1.0),
    ]

    results = []
    for name, setup, kps in configs:
        r = run(args.xml_path, jp, jv, bp, bq, sess, args.duration, setup=setup, kp_scale=kps)
        tag = "撑满!" if r["fall"] >= args.duration - 1e-6 else f"{r['fall']:.2f}s"
        print(f"{name:<38s} {tag:>8s} {r['min_h']:10.4f} {r['jerr']:10.4f} {r['sat_pct']:7.1f}%")
        results.append((name, r["fall"], r))

    print("-" * 86)
    best = max(results, key=lambda x: x[1])
    print(f"\n 最佳: {best[0]}  ->  {best[1]:.2f}s")
    print(f" Isaac 参考: 9.80s")

    # 最佳配置的高度轨迹
    hs = best[2]["hs"]
    print(f"\n 高度轨迹（{best[0]}）:")
    step_i = max(1, len(hs) // 16)
    for i in range(0, len(hs), step_i):
        print(f"   {i*0.02:5.2f}s  {hs[i]:.4f}  {'#'*int(max(0,hs[i])*100)}")
    print("=" * 86)


if __name__ == "__main__":
    main()
