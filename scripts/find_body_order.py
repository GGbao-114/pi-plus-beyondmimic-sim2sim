#!/usr/bin/env python
"""找出 npz body_pos_w 的真实身体顺序。

方法: 把 MuJoCo 摆成动作第0帧的姿势(已知正确)，然后对 npz 的每个身体索引，
      在 MuJoCo 的所有身体里找位置最接近的那个 -> 反推真实顺序。

用法:
  python find_body_order.py --motion_file <npz> --xml_path <mjcf>
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_file", required=True)
    ap.add_argument("--xml_path", required=True)
    args = ap.parse_args()

    mo = np.load(args.motion_file)
    print("[npz 内容]")
    for k in mo.files:
        print(f"   {k:20s} {mo[k].shape}")

    jp = mo["joint_pos"]
    bp = mo["body_pos_w"]

    m = mujoco.MjModel.from_xml_path(args.xml_path)
    d = mujoco.MjData(m)

    # 关节自省
    xml_names, qadr = [], []
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        xml_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
        qadr.append(int(m.jnt_qposadr[jid]))
    qadr = np.array(qadr)

    # 所有 body 名
    all_bodies = []
    for bid in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if nm:
            all_bodies.append((bid, nm))

    # 摆到第 0 帧
    d.qpos[:] = 0.0
    d.qpos[0:3] = bp[0, 0]
    q = mo["body_quat_w"][0, 0].copy()
    d.qpos[3:7] = q / np.linalg.norm(q)
    for i, nm in enumerate(xml_names):
        d.qpos[qadr[i]] = jp[0][TRAIN_JOINTS.index(nm)]
    mujoco.mj_forward(m, d)

    # 打印 MuJoCo 所有 body 位置（按 z 排序，便于识别）
    print("\n[MuJoCo 各 body 位置 - 第0帧，按 z 从低到高]")
    rows = []
    for bid, nm in all_bodies:
        p = d.xpos[bid]
        rows.append((p[2], nm, bid, p))
    for z, nm, bid, p in sorted(rows):
        print(f"   z={z:+.3f}  y={p[1]:+.3f}  x={p[0]:+.3f}   {nm}")

    # npz 的 13 个身体，按 z 排序
    print("\n[npz body_pos_w 的 13 个索引 - 第0帧，按 z 从低到高]")
    ref_rows = []
    for i in range(bp.shape[1]):
        p = bp[0, i]
        ref_rows.append((p[2], i, p))
    for z, i, p in sorted(ref_rows):
        print(f"   z={z:+.3f}  y={p[1]:+.3f}  x={p[0]:+.3f}   npz[{i}]")

    # 最近邻匹配
    print("\n[最近邻匹配] npz索引 -> 最接近的 MuJoCo body")
    print(f"   {'npz':>4s}  {'最近body':26s} {'距离(m)':>9s}  {'npz位置':>22s}  {'MuJoCo位置':>22s}")
    for i in range(bp.shape[1]):
        p_ref = bp[0, i]
        best, bd = None, 1e9
        for bid, nm in all_bodies:
            dd = np.linalg.norm(d.xpos[bid] - p_ref)
            if dd < bd:
                bd, best = dd, nm
        ps = f"[{p_ref[0]:+.3f} {p_ref[1]:+.3f} {p_ref[2]:+.3f}]"
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, best)
        pm = d.xpos[bid]
        pms = f"[{pm[0]:+.3f} {pm[1]:+.3f} {pm[2]:+.3f}]"
        flag = "  <== 匹配" if bd < 0.02 else ""
        print(f"   {i:4d}  {best:26s} {bd:9.4f}  {ps}  {pms}{flag}")

    print("\n" + "=" * 62)
    print(" 如果左列 npz 索引与右列 body 名有一致的错位，")
    print(" 说明 npz body 顺序 != flat_env_cfg.py 的 body_names 顺序。")
    print("=" * 62)


if __name__ == "__main__":
    main()
