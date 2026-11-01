#!/usr/bin/env python
"""MuJoCo 参数消融实验：逐项修改 MJCF，看机器人能撑多久。

目的: 找出 MuJoCo 与 Isaac 之间导致摔倒的关键差异。
方法: 每次只改一项，跑固定时长，记录"摔倒时间"。

用法:
  MUJOCO_GL=egl python sweep_mjcf.py \
    --motion_file <npz> --xml_path <mjcf> --policy_path <onnx>
"""
import argparse
import numpy as np
import mujoco
import onnx
import onnxruntime

SIM_DT = 0.002
CTRL_DEC = 10

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
    if n.endswith("_hip_pitch_joint"):
        return -0.25
    if n.endswith("_calf_joint"):
        return 0.65
    if n.endswith("_ankle_pitch_joint"):
        return -0.4
    return 0.0


def gains(n):
    return (30.0, 0.6) if n.endswith(ARM_SUF) else (80.0, 1.1)


def build_model(xml_path):
    m = mujoco.MjModel.from_xml_path(xml_path)
    m.opt.timestep = SIM_DT
    return m


def run_once(xml_path, jp, jv, bp, bq, sess, duration, modify=None,
             init_from_motion=True, kp_scale=1.0):
    m = build_model(xml_path)
    d = mujoco.MjData(m)

    if modify:
        modify(m)

    xml_names, qadr, dadr = [], [], []
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        xml_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
        qadr.append(int(m.jnt_qposadr[jid]))
        dadr.append(int(m.jnt_dofadr[jid]))
    qadr, dadr = np.array(qadr), np.array(dadr)

    act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0]))
                 for i in range(m.nu)]
    a2b = np.array([BFS_JOINTS.index(j) for j in act_names])
    x2b = np.array([BFS_JOINTS.index(j) for j in xml_names])

    kp_a = np.array([gains(j)[0] for j in act_names]) * kp_scale
    kd_a = np.array([gains(j)[1] for j in act_names])
    dpos_b = np.array([dpos_of(j) for j in BFS_JOINTS])
    sc_b = np.array([0.25 * (20.0 if j.endswith(ARM_SUF) else 30.0) / gains(j)[0]
                     for j in BFS_JOINTS])

    # 初始状态
    d.qpos[:] = 0.0
    if init_from_motion:
        d.qpos[0:3] = bp[0, 0]
        q0 = bq[0, 0].copy()
        d.qpos[3:7] = q0 / np.linalg.norm(q0)
        for i, nm in enumerate(xml_names):
            d.qpos[qadr[i]] = jp[0][BFS_JOINTS.index(nm)]
    mujoco.mj_forward(m, d)

    target_b = jp[0].copy()
    action_buf = np.zeros(22, dtype=np.float32)
    n_frames = min(len(jp), len(bp))
    n_ctrl = int(duration / (SIM_DT * CTRL_DEC))

    fall_t = None
    min_h = 9.9
    jerrs = []

    for step in range(n_ctrl * CTRL_DEC):
        d.ctrl[:] = (target_b[a2b] - d.qpos[qadr]) * kp_a - d.qvel[dadr] * kd_a
        mujoco.mj_step(m, d)

        if (step + 1) % CTRL_DEC == 0:
            t = (step + 1) // CTRL_DEC
            fi = min(t, n_frames - 1)
            h = d.qpos[2]
            min_h = min(min_h, h)
            if fall_t is None and h < 0.15:
                fall_t = t * SIM_DT * CTRL_DEC

            q_b = d.qpos[qadr][x2b]
            jerrs.append(np.abs(q_b - jp[fi]).mean())

            quat = d.qpos[3:7].copy()
            omega = d.sensor("angular-velocity").data.copy()
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
            obs[75:97] = d.qvel[dadr][x2b]
            obs[97:119] = action_buf
            out = sess.run(["actions"], {
                "obs": obs.reshape(1, -1),
                "time_step": np.array([[float(fi)]], dtype=np.float32)})[0]
            action_b = np.asarray(out).reshape(-1).astype(np.float32)
            action_buf = action_b.copy()
            target_b = action_b * sc_b + dpos_b

    return fall_t if fall_t is not None else duration, min_h, float(np.mean(jerrs))


# ============ 修改函数 ============
def mod_contact_stiff(m):
    m.geom_solref[:, 0] = 0.002      # 默认 0.02 -> 更硬的接触

def mod_friction(m):
    m.geom_friction[:, 0] = 1.5      # 滑动摩擦

def mod_torque(m):
    m.jnt_actfrcrange[:, 0] = -30.0
    m.jnt_actfrcrange[:, 1] = 30.0
    m.actuator_ctrlrange[:, 0] = -30.0
    m.actuator_ctrlrange[:, 1] = 30.0

def mod_nofrictionloss(m):
    m.dof_frictionloss[:] = 0.0

def mod_noimu(m):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
    if sid >= 0:
        m.sensor_cutoff[sid] = 0.0   # 关闭低通滤波

def mod_all(m):
    mod_contact_stiff(m); mod_friction(m); mod_torque(m)
    mod_nofrictionloss(m); mod_noimu(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_file", required=True)
    ap.add_argument("--xml_path", required=True)
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--duration", type=float, default=6.0)
    args = ap.parse_args()

    mo = np.load(args.motion_file)
    jp, jv, bp, bq = mo["joint_pos"], mo["joint_vel"], mo["body_pos_w"], mo["body_quat_w"]
    sess = onnxruntime.InferenceSession(args.policy_path,
                                        providers=["CPUExecutionProvider"])

    print("=" * 78)
    print(" MuJoCo 参数消融实验")
    print("=" * 78)
    print(f" 仿真时长上限 {args.duration}s   (Isaac 里能撑 9.8s)")
    print("=" * 78)
    print(f"{'配置':<34s} {'摔倒时间':>10s} {'最低高度':>10s} {'关节误差':>10s}")
    print("-" * 78)

    configs = [
        ("① 基线（原版设置）", None, True),
        ("② 初始姿态=动作第0帧", None, False),   # 注意: 单独看初始化影响
        ("③ 提高接触刚度 solref 0.02→0.002", mod_contact_stiff, True),
        ("④ 提高摩擦 1.0→1.5", mod_friction, True),
        ("⑤ 力矩上限 20→30 (对齐 Isaac)", mod_torque, True),
        ("⑥ 去掉关节干摩擦 frictionloss", mod_nofrictionloss, True),
        ("⑦ 关闭 IMU 低通滤波", mod_noimu, True),
        ("⑧ 全部修改一起", mod_all, True),
    ]

    results = []
    for name, mod, init_motion in configs:
        ft, mh, je = run_once(args.xml_path, jp, jv, bp, bq, sess,
                              args.duration, modify=mod, init_from_motion=init_motion)
        tag = "撑满" if ft >= args.duration - 1e-6 else f"{ft:.2f}s"
        print(f"{name:<34s} {tag:>10s} {mh:10.4f} {je:10.4f}")
        results.append((name, ft))

    print("-" * 78)
    best = max(results, key=lambda x: x[1])
    print(f"\n 撑最久的配置: {best[0]}  ({best[1]:.2f}s)")
    print(" 参考: Isaac Sim 里约 9.8s")
    print("=" * 78)


if __name__ == "__main__":
    main()
