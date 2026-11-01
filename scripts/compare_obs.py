#!/usr/bin/env python
"""用 Isaac 的真实状态复现到 MuJoCo，对比观测。

用法:
  python compare_obs.py --motion_file <npz> --xml_path <mjcf> \
      --policy_path <onnx> --isaac_state /data/gaoli/logs/isaac_state.npz
"""
import argparse
import numpy as np
import mujoco
import onnxruntime

# MJCF 关节顺序（从 XML 读出，见下）
ARM_SUF = ("_shoulder_pitch_joint", "_shoulder_roll_joint",
           "_upper_arm_joint", "_elbow_joint", "_wrist_joint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_file", required=True)
    ap.add_argument("--xml_path", required=True)
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--isaac_state", default="/data/gaoli/logs/isaac_state.npz")
    args = ap.parse_args()

    S = np.load(args.isaac_state, allow_pickle=True)
    isaac_joint_names = [str(x) for x in S["joint_names"]]
    obs_i = S["obs"]
    ts = int(S["time_step"])
    print("=" * 78)
    print(" Isaac 状态 -> MuJoCo 复现，观测对比")
    print("=" * 78)
    print(f"  Isaac 关节顺序: {isaac_joint_names[:6]} ...")
    print(f"  time_step = {ts}")

    mo = np.load(args.motion_file)
    jp, jv, bp, bq = mo["joint_pos"], mo["joint_vel"], mo["body_pos_w"], mo["body_quat_w"]

    m = mujoco.MjModel.from_xml_path(args.xml_path)
    d = mujoco.MjData(m)

    xml_names, qadr, dadr = [], [], []
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        xml_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
        qadr.append(int(m.jnt_qposadr[jid]))
        dadr.append(int(m.jnt_dofadr[jid]))
    qadr, dadr = np.array(qadr), np.array(dadr)

    # ---- 建立 Isaac关节序 <-> MJCF关节序 的映射（按名字）----
    def get_isaac(name):
        return S["joint_pos"][isaac_joint_names.index(name)]

    def get_isaac_vel(name):
        return S["joint_vel"][isaac_joint_names.index(name)]

    # ---- 设置 MuJoCo 状态 = Isaac 的状态 ----
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.qpos[0:3] = S["root_pos"]
    q = S["root_quat"].copy()
    d.qpos[3:7] = q / np.linalg.norm(q)
    for i, nm in enumerate(xml_names):
        d.qpos[qadr[i]] = get_isaac(nm)
        d.qvel[dadr[i]] = get_isaac_vel(nm)
    # 基座角速度：Isaac 给的是体系，MuJoCo qvel[3:6] 也用体系
    d.qvel[3:6] = S["root_ang_vel_b"]
    mujoco.mj_forward(m, d)

    # ---- 组装观测（按训练/Isaac 顺序）----
    quat = d.qpos[3:7].copy()
    omega = d.sensor("angular-velocity").data.copy()

    # 关节角/速度 -> Isaac 顺序
    q_is = np.array([d.qpos[qadr][xml_names.index(n)] for n in isaac_joint_names])
    dq_is = np.array([d.qvel[dadr][xml_names.index(n)] for n in isaac_joint_names])

    # 默认关节角（用 Isaac 导出的，保持一致）
    dpos_is = S["default_pos"]

    # motion_anchor_ori_b
    qq = bq[ts, 0]
    robot_q = d.qpos[3:7]
    w1, x1, y1, z1 = robot_q[0], -robot_q[1], -robot_q[2], -robot_q[3]
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

    obs_m = np.zeros(119, dtype=np.float32)
    obs_m[0:44] = np.concatenate((jp[ts], jv[ts]))
    obs_m[44:50] = mat[:, :2].reshape(6)
    obs_m[50:53] = omega
    obs_m[53:75] = q_is - dpos_is
    obs_m[75:97] = dq_is
    obs_m[97:119] = 0.0

    # ---- 对比 ----
    print("\n" + "=" * 78)
    print(" 观测逐项对比（Isaac 真实状态 -> 两边计算）")
    print("=" * 78)
    for nm, s, e in [("command", 0, 44), ("motion_ori", 44, 50), ("base_ang_vel", 50, 53),
                     ("joint_pos_rel", 53, 75), ("joint_vel", 75, 97), ("last_action", 97, 119)]:
        dd = np.abs(obs_i[s:e] - obs_m[s:e])
        mark = "OK" if dd.max() < 1e-3 else ("~" if dd.max() < 0.02 else "XX")
        print(f"  {nm:16s} 最大差 {dd.max():10.6f}  [{mark}]")
        if dd.max() > 1e-3:
            for i in np.argsort(-dd)[:4]:
                print(f"       [{s+i:3d}] Isaac={obs_i[s+i]:+.5f}  MuJoCo={obs_m[s+i]:+.5f}  diff={obs_i[s+i]-obs_m[s+i]:+.5f}")

    # action 对比
    sess = onnxruntime.InferenceSession(args.policy_path, providers=["CPUExecutionProvider"])
    a_i = np.asarray(sess.run(["actions"], {
        "obs": obs_i.reshape(1, -1), "time_step": np.array([[float(ts)]], dtype=np.float32)})[0]).reshape(-1)
    a_m = np.asarray(sess.run(["actions"], {
        "obs": obs_m.reshape(1, -1), "time_step": np.array([[float(ts)]], dtype=np.float32)})[0]).reshape(-1)
    df = np.abs(a_i - a_m)
    print(f"\n  action 最大差 {df.max():.6f}  (同一 ONNX，不同观测 -> 不同动作)")

    print("\n" + "=" * 78)
    print(" 关键诊断")
    print("=" * 78)
    print(f"  Isaac 的 base_ang_vel_b : {np.round(S['root_ang_vel_b'],5)}")
    print(f"  MuJoCo gyro             : {np.round(omega,5)}")
    print(f"  两者差                  : {np.abs(S['root_ang_vel_b']-omega).max():.6f}")
    print(f"\n  Isaac 的 joint_pos 前6  : {np.round(S['joint_pos'][:6],5)}")
    print(f"  MuJoCo 映射后  前6      : {np.round(q_is[:6],5)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
