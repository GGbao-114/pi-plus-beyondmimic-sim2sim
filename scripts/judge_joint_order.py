#!/usr/bin/env python
"""判定 npz 的真实关节顺序：CSV顺序 vs BFS顺序。

方法: 用每种假设把 MuJoCo 摆成动作第0帧的姿态，做正运动学，
      然后和 npz 的 body_pos_w 对比。误差小的那个就是正确顺序。

不需要 Isaac Sim。
用法:
  python judge_joint_order.py --motion_file <npz> --xml_path <mjcf>
"""
import argparse
import numpy as np
import mujoco

# ---- 假设 A: CSV 顺序（左腿->右腿->左臂->右臂）----
CSV_JOINTS = [
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint",
    "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint", "r_calf_joint",
    "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_joint",
    "l_elbow_joint", "l_wrist_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_joint",
    "r_elbow_joint", "r_wrist_joint",
]

# ---- 假设 B: BFS 顺序（PhysX articulation 树遍历）----
BFS_JOINTS = [
    "l_hip_pitch_joint", "l_shoulder_pitch_joint", "r_hip_pitch_joint", "r_shoulder_pitch_joint",
    "l_hip_roll_joint", "l_shoulder_roll_joint", "r_hip_roll_joint", "r_shoulder_roll_joint",
    "l_thigh_joint", "l_upper_arm_joint", "r_thigh_joint", "r_upper_arm_joint",
    "l_calf_joint", "l_elbow_joint", "r_calf_joint", "r_elbow_joint",
    "l_ankle_pitch_joint", "l_wrist_joint", "r_ankle_pitch_joint", "r_wrist_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
]

# ---- 13 个被跟踪的身体，在 npz 23 身体列表里的索引（由 find_body_order 得出）----
TRACKED_BODY_NAMES = [
    "base_link", "l_hip_roll_link", "l_calf_link", "l_ankle_roll_link",
    "r_hip_roll_link", "r_calf_link", "r_ankle_roll_link",
    "l_shoulder_roll_link", "l_elbow_link", "l_wrist_link",
    "r_shoulder_roll_link", "r_elbow_link", "r_wrist_link",
]
NPZ_BODY_IDX = [0, 5, 13, 21, 7, 15, 22, 6, 14, 18, 8, 16, 20]


def test_order(m, d, xml_names, qadr, order, jp, bp, bq, frames):
    """用给定关节顺序做 FK，返回平均误差"""
    errs = []
    for fr in frames:
        d.qpos[:] = 0.0
        d.qpos[0:3] = bp[fr, 0]
        q = bq[fr, 0].copy()
        d.qpos[3:7] = q / np.linalg.norm(q)
        # xml 关节 -> 该假设下的值
        for i, nm in enumerate(xml_names):
            d.qpos[qadr[i]] = jp[fr][order.index(nm)]
        mujoco.mj_forward(m, d)

        for k, bidx in enumerate(NPZ_BODY_IDX):
            bn = TRACKED_BODY_NAMES[k]
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
            if bid < 0:
                continue
            errs.append(np.linalg.norm(d.xpos[bid] - bp[fr, bidx]))
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_file", required=True)
    ap.add_argument("--xml_path", required=True)
    args = ap.parse_args()

    mo = np.load(args.motion_file)
    jp = mo["joint_pos"]
    bp = mo["body_pos_w"]
    bq = mo["body_quat_w"]
    print(f"[INFO] npz: joint_pos {jp.shape}, body_pos_w {bp.shape}")

    if bp.shape[1] != 23:
        print(f"[警告] body 数 {bp.shape[1]} != 23，索引映射可能需要调整")

    m = mujoco.MjModel.from_xml_path(args.xml_path)
    d = mujoco.MjData(m)

    xml_names, qadr = [], []
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        xml_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
        qadr.append(int(m.jnt_qposadr[jid]))
    qadr = np.array(qadr)

    frames = [0, 50, 100, 200, 300]

    print("\n" + "=" * 64)
    print(" 关节顺序判定：用 FK 对比动作数据")
    print("=" * 64)

    results = {}
    for name, order in [("CSV顺序 (左腿→右腿→左臂→右臂)", CSV_JOINTS),
                        ("BFS顺序 (PhysX树遍历)", BFS_JOINTS)]:
        e = test_order(m, d, xml_names, qadr, order, jp, bp, bq, frames)
        results[name] = e
        print(f"\n  假设【{name}】")
        print(f"    平均误差 : {e.mean():.4f} m")
        print(f"    最大误差 : {e.max():.4f} m")
        print(f"    中位误差 : {np.median(e):.4f} m")

    best = min(results, key=lambda k: results[k].mean())
    print("\n" + "=" * 64)
    print(f"  判定结果: 【{best}】误差更小")
    print("=" * 64)

    # 逐帧细看
    print("\n  逐帧误差对比:")
    print(f"    {'帧':>5s}  {'CSV':>10s}  {'BFS':>10s}")
    e_csv = results["CSV顺序 (左腿→右腿→左臂→右臂)"]
    e_bfs = results["BFS顺序 (PhysX树遍历)"]
    n = 13  # 每帧13个身体
    for i, fr in enumerate(frames):
        s, t = i * n, (i + 1) * n
        print(f"    {fr:5d}  {e_csv[s:t].mean():10.4f}  {e_bfs[s:t].mean():10.4f}")

    print("\n  参考: 误差 < 0.02 m 表示顺序正确")


if __name__ == "__main__":
    main()
