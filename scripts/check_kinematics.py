#!/usr/bin/env python
"""运动学一致性检查：MJCF 的正运动学 vs 动作数据(URDF生成)是否一致。

如果 MJCF 和 URDF 的关节轴向/连杆偏移不同，同样的关节角会得到不同姿态，
策略在 MuJoCo 里必然失败。这是 sim2sim 最常见也最隐蔽的 gap。

用法:
  python check_kinematics.py --motion_file <npz> --xml_path <mjcf>
"""
import argparse
import numpy as np
import mujoco

TRAIN_JOINTS = [
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint",
    "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint", "r_calf_joint",
    "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_joint",
    "l_elbow_joint", "l_wrist_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_joint",
    "r_elbow_joint", "r_wrist_joint",
]

BODY_NAMES = [
    "base_link", "l_hip_roll_link", "l_calf_link", "l_ankle_roll_link",
    "r_hip_roll_link", "r_calf_link", "r_ankle_roll_link",
    "l_shoulder_roll_link", "l_elbow_link", "l_wrist_link",
    "r_shoulder_roll_link", "r_elbow_link", "r_wrist_link",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_file", required=True)
    ap.add_argument("--xml_path", required=True)
    args = ap.parse_args()

    mo = np.load(args.motion_file)
    jp = mo["joint_pos"]          # (N,22) 训练顺序
    bp = mo["body_pos_w"]         # (N,13,3)
    bq = mo["body_quat_w"]        # (N,13,4)
    print(f"[INFO] 动作 {jp.shape}, body {bp.shape}")

    m = mujoco.MjModel.from_xml_path(args.xml_path)
    d = mujoco.MjData(m)

    # 自省关节信息
    xml_names, qadr = [], []
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        xml_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
        qadr.append(int(m.jnt_qposadr[jid]))
    qadr = np.array(qadr)
    print(f"[INFO] MJCF 关节 {len(xml_names)} 个")

    # 检查关节名匹配
    miss = [j for j in TRAIN_JOINTS if j not in xml_names]
    if miss:
        print(f"[!!] MJCF 缺少: {miss}")

    print("\n=== 关节轴向对比（MJCF）===")
    for i, nm in enumerate(xml_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        ax = m.jnt_axis[jid]
        print(f"  {nm:26s} axis=[{ax[0]:+.2f} {ax[1]:+.2f} {ax[2]:+.2f}] "
              f"range=[{m.jnt_range[jid][0]:+.2f} {m.jnt_range[jid][1]:+.2f}]")

    # 检查身体名
    print("\n=== 身体名检查 ===")
    for bn in BODY_NAMES:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
        st = "OK" if bid >= 0 else "缺失!!"
        print(f"  {bn:26s} {st}")

    # ---- 对前几帧做 FK 对比 ----
    print("\n" + "=" * 68)
    print(" 正运动学对比：MuJoCo FK  vs  动作数据(Isaac/URDF 生成)")
    print("=" * 68)

    for frame in [0, 100, 300]:
        if frame >= len(jp):
            continue

        # 用该帧的关节角 + 基座位姿设置 MuJoCo
        d.qpos[:] = 0.0
        d.qpos[0:3] = bp[frame, 0]
        q = bq[frame, 0].copy()
        d.qpos[3:7] = q / np.linalg.norm(q)
        for i, nm in enumerate(xml_names):
            d.qpos[qadr[i]] = jp[frame][TRAIN_JOINTS.index(nm)]
        mujoco.mj_forward(m, d)

        print(f"\n--- 第 {frame} 帧 ---")
        print(f"  {'body':26s} {'误差(m)':>9s}   MuJoCo位置                动作数据位置")
        errs = []
        for bi, bn in enumerate(BODY_NAMES):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
            if bid < 0:
                continue
            p_mj = d.xpos[bid]
            p_ref = bp[frame, bi]
            e = np.linalg.norm(p_mj - p_ref)
            errs.append(e)
            flag = "  <<<" if e > 0.05 else ""
            print(f"  {bn:26s} {e:9.4f}   "
                  f"[{p_mj[0]:+.3f} {p_mj[1]:+.3f} {p_mj[2]:+.3f}]  "
                  f"[{p_ref[0]:+.3f} {p_ref[1]:+.3f} {p_ref[2]:+.3f}]{flag}")
        errs = np.array(errs)
        print(f"  {'平均误差':26s} {errs.mean():9.4f}   最大 {errs.max():.4f}")

    print("\n" + "=" * 68)
    print(" 结论指引")
    print("=" * 68)
    print("  误差 < 0.02 m  -> MJCF 与 URDF 运动学一致，问题在别处")
    print("  误差 > 0.05 m  -> MJCF 与 URDF 不一致，这是 sim2sim gap 的根因")
    print("=" * 68)


if __name__ == "__main__":
    main()
