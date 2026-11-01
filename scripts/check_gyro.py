#!/usr/bin/env python
"""无歧义判定 MuJoCo gyro 传感器的坐标系。

用 mj_objectVelocity 分别取【世界系】和【局部系】角速度，
与 gyro 传感器读数对比 —— 谁匹配就是谁。

用法:
  python check_gyro.py --xml_path <mjcf>
"""
import argparse
import math
import numpy as np
import mujoco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_path", required=True)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml_path)
    print("=" * 78)
    print(" MuJoCo gyro 坐标系判定")
    print("=" * 78)

    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
    print(f"  gyro sensor id = {sid}, cutoff = {m.sensor_cutoff[sid] if sid>=0 else 'N/A'}")

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    print(f"  base body id   = {bid}")

    # 设一个明显非对称的状态：基座 roll 60°, pitch 30°，并给一个非零角速度
    tests = [
        ("roll 60°+pitch 30°, qvel[3:6]=(0.3,0.5,0.8)", 60.0, 30.0, (0.3, 0.5, 0.8)),
        ("roll 0°, qvel[3:6]=(1,0,0)", 0.0, 0.0, (1.0, 0.0, 0.0)),
        ("yaw 90°, qvel[3:6]=(0,0,1)", 0.0, 0.0, (0.0, 0.0, 1.0)),
    ]

    for label, roll, pitch, qv in tests:
        d = mujoco.MjData(m)
        d.qpos[:] = 0.0
        d.qpos[0:3] = [0, 0, 1.0]

        # 组合 roll/pitch
        r = math.radians(roll); p = math.radians(pitch)
        qr = np.array([math.cos(r/2), math.sin(r/2), 0, 0])
        qp = np.array([math.cos(p/2), 0, math.sin(p/2), 0])
        # qp ⊗ qr
        w1,x1,y1,z1 = qp; w2,x2,y2,z2 = qr
        q = np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                      w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
        # 如果是 yaw 测试
        if "yaw" in label:
            y = math.radians(90.0)
            q = np.array([math.cos(y/2), 0, 0, math.sin(y/2)])
        d.qpos[3:7] = q / np.linalg.norm(q)
        d.qvel[:] = 0.0
        d.qvel[3:6] = qv
        mujoco.mj_forward(m, d)

        # 用 mj_objectVelocity 取两种坐标系
        res = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)  # 0 = global
        w_global = res[0:3].copy()
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, res, 1)  # 1 = local
        w_local = res[0:3].copy()

        gyro = d.sensor("angular-velocity").data.copy()

        print(f"\n--- {label} ---")
        print(f"  d.qvel[3:6]              = [{qv[0]:+.4f} {qv[1]:+.4f} {qv[2]:+.4f}]")
        print(f"  mj_objectVelocity global = [{w_global[0]:+.4f} {w_global[1]:+.4f} {w_global[2]:+.4f}]")
        print(f"  mj_objectVelocity local  = [{w_local[0]:+.4f} {w_local[1]:+.4f} {w_local[2]:+.4f}]")
        print(f"  gyro 传感器               = [{gyro[0]:+.4f} {gyro[1]:+.4f} {gyro[2]:+.4f}]")

        eg = np.abs(gyro - w_global).max()
        el = np.abs(gyro - w_local).max()
        if el < 1e-6 and eg > 1e-6:
            verdict = "✅ 局部系（与 Isaac 的 root_ang_vel_b 一致）"
        elif eg < 1e-6 and el > 1e-6:
            verdict = "⚠️ 世界系（与 Isaac 不一致，需转换）"
        elif eg < 1e-6 and el < 1e-6:
            verdict = "— 两者相同（该姿态无区分度）"
        else:
            verdict = "❓ 都不匹配"
        print(f"  |gyro-global|={eg:.6f}   |gyro-local|={el:.6f}")
        print(f"  判定: {verdict}")

    # 顺便：检查 Isaac 用的 root_ang_vel_b 语义
    print("\n" + "=" * 78)
    print(" 结论")
    print("=" * 78)
    print("  Isaac Lab 的 mdp.base_ang_vel = robot.data.root_ang_vel_b")
    print("  即【基座体坐标系下的角速度】。")
    print("  如果上面判定为『局部系』 → 现有 sim2sim 的用法正确，问题在别处。")
    print("  如果判定为『世界系』   → 必须转换: omega_body = R^T @ omega_world")
    print("=" * 78)


if __name__ == "__main__":
    main()
