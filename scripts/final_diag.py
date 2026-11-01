#!/usr/bin/env python
"""最终诊断：转动惯量 + 关节轴 + 静态站立测试。

三件事:
  A. 对比 MJCF 与 URDF 的转动惯量（张量旋转后对比）
  B. 对比关节轴向量
  C. 静态站立测试：保持默认站立姿态，看模型能否撑住

用法:
  MUJOCO_GL=egl python final_diag.py --xml_path <mjcf> --urdf_path <urdf>
"""
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

BFS_JOINTS = [
    "l_hip_pitch_joint", "l_shoulder_pitch_joint", "r_hip_pitch_joint", "r_shoulder_pitch_joint",
    "l_hip_roll_joint", "l_shoulder_roll_joint", "r_hip_roll_joint", "r_shoulder_roll_joint",
    "l_thigh_joint", "l_upper_arm_joint", "r_thigh_joint", "r_upper_arm_joint",
    "l_calf_joint", "l_elbow_joint", "r_calf_joint", "r_elbow_joint",
    "l_ankle_pitch_joint", "l_wrist_joint", "r_ankle_pitch_joint", "r_wrist_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
]


def quat_to_mat(q):
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)]])


def rpy_to_mat(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx


def parse_urdf(path):
    tree = ET.parse(path)
    root = tree.getroot()
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        ine = link.find("inertial")
        if ine is None:
            continue
        mass = float(ine.find("mass").get("value").strip())
        org = ine.find("origin")
        xyz = np.array([float(v) for v in (org.get("xyz", "0 0 0")).split()]) if org is not None else np.zeros(3)
        rpy = [float(v) for v in (org.get("rpy", "0 0 0")).split()] if org is not None else [0,0,0]
        i = ine.find("inertia")
        I = np.array([
            [float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
            [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
            [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))]])
        out[name] = dict(mass=mass, com=xyz, rpy=rpy, I=I)
    # joints
    joints = {}
    for j in root.findall("joint"):
        name = j.get("name")
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None
        org = j.find("origin")
        xyz = np.array([float(v) for v in (org.get("xyz","0 0 0")).split()]) if org is not None else np.zeros(3)
        joints[name] = dict(axis=axis, xyz=xyz, type=j.get("type"))
    return out, joints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_path", required=True)
    ap.add_argument("--urdf_path", required=True)
    args = ap.parse_args()

    urdf_links, urdf_joints = parse_urdf(args.urdf_path)
    m = mujoco.MjModel.from_xml_path(args.xml_path)
    print(f"[INFO] URDF links {len(urdf_links)}, MJCF bodies {m.nbody}")

    # ================= A. 转动惯量 =================
    print("\n" + "=" * 84)
    print(" A. 转动惯量对比（特征值，越小越不敏感于坐标系）")
    print("=" * 84)
    print(f"{'body':<26s} {'URDF特征值':>30s} {'MJCF特征值':>30s}")
    print("-" * 84)

    bad = []
    for bid in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if nm is None or nm not in urdf_links:
            continue
        # MJCF: diaginertia (principal) + iquat
        din = m.body_inertia[bid].copy()
        iq = m.body_iquat[bid].copy()
        R = quat_to_mat(iq)
        I_mj = R @ np.diag(din) @ R.T

        # URDF: full tensor (in inertial frame, rpy=0)
        I_ur = urdf_links[nm]["I"]

        ev_u = np.sort(np.linalg.eigvalsh(I_ur))
        ev_m = np.sort(np.linalg.eigvalsh(I_mj))
        rel = np.abs(ev_u - ev_m).max() / max(np.abs(ev_u).max(), 1e-9)

        flag = ""
        if rel > 0.15:
            flag = "  <<< 差异大"
            bad.append((nm, rel))
        print(f"{nm:<26s} [{ev_u[0]:.6f} {ev_u[1]:.6f} {ev_u[2]:.6f}] "
              f"[{ev_m[0]:.6f} {ev_m[1]:.6f} {ev_m[2]:.6f}]{flag}")

    print("-" * 84)
    if bad:
        print(f"  ⚠️ {len(bad)} 个 link 的转动惯量差异 > 15%:")
        for nm, r in sorted(bad, key=lambda x: -x[1])[:8]:
            print(f"     {nm:<26s} 相对差异 {r*100:.1f}%")
    else:
        print("  ✅ 所有 link 的转动惯量特征值一致（差异 < 15%）")

    # ================= B. 关节轴 =================
    print("\n" + "=" * 84)
    print(" B. 关节轴对比")
    print("=" * 84)
    print(f"{'joint':<26s} {'URDF axis':>22s} {'MJCF axis':>22s}")
    print("-" * 84)
    ax_bad = []
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        ax_m = m.jnt_axis[jid]
        if nm in urdf_joints and urdf_joints[nm]["axis"] is not None:
            ax_u = urdf_joints[nm]["axis"]
            d = np.abs(ax_u - ax_m).max()
            flag = ""
            if d > 0.01:
                flag = "  <<< 轴不同!"
                ax_bad.append(nm)
            print(f"{nm:<26s} [{ax_u[0]:+.2f} {ax_u[1]:+.2f} {ax_u[2]:+.2f}]"
                  f"{'':>6s} [{ax_m[0]:+.2f} {ax_m[1]:+.2f} {ax_m[2]:+.2f}]{flag}")
    print("-" * 84)
    print("  ✅ 关节轴一致" if not ax_bad else f"  ⚠️ {len(ax_bad)} 个关节轴不同: {ax_bad}")

    # ================= C. 静态站立测试 =================
    print("\n" + "=" * 84)
    print(" C. 静态站立测试（保持默认姿态，不用策略）")
    print("=" * 84)
    d = mujoco.MjData(m)
    m.opt.timestep = 0.002

    xml_names, qadr, dadr = [], [], []
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        xml_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
        qadr.append(int(m.jnt_qposadr[jid]))
        dadr.append(int(m.jnt_dofadr[jid]))
    qadr, dadr = np.array(qadr), np.array(dadr)

    act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i,0]))
                 for i in range(m.nu)]

    def dpos_of(n):
        if n.endswith("_hip_pitch_joint"): return -0.25
        if n.endswith("_calf_joint"): return 0.65
        if n.endswith("_ankle_pitch_joint"): return -0.4
        return 0.0

    def gains(n):
        if n.endswith(("_shoulder_pitch_joint","_shoulder_roll_joint","_upper_arm_joint",
                       "_elbow_joint","_wrist_joint")):
            return 30.0, 0.6
        return 80.0, 1.1

    kp_a = np.array([gains(j)[0] for j in act_names])
    kd_a = np.array([gains(j)[1] for j in act_names])
    tgt = np.array([dpos_of(j) for j in act_names])

    # 初始：默认姿态，基座在 0.351
    d.qpos[:] = 0.0
    d.qpos[0:3] = [0, 0, 0.351]
    d.qpos[3:7] = [1, 0, 0, 0]
    for i, nm in enumerate(xml_names):
        d.qpos[qadr[i]] = dpos_of(nm)
    mujoco.mj_forward(m, d)
    print(f"  初始基座高度 {d.qpos[2]:.4f} m, 接触数 {d.ncon}")
    for ci in range(min(d.ncon, 4)):
        c = d.contact[ci]
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]) or "world"
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]) or "world"
        print(f"    接触: {b1} <-> {b2}   dist={c.dist:+.5f}")

    print(f"\n  {'秒':>6s} {'基座高度':>10s} {'基座x':>9s} {'基座y':>9s} {'最大|tau|':>10s}")
    print("  " + "-" * 50)
    t_fall = None
    for step in range(int(6.0 / 0.002)):
        tau = (tgt - d.qpos[qadr]) * kp_a - d.qvel[dadr] * kd_a
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        if step % 250 == 0:
            print(f"  {step*0.002:6.2f} {d.qpos[2]:10.4f} {d.qpos[0]:9.4f} {d.qpos[1]:9.4f} "
                  f"{np.abs(tau).max():10.2f}")
        if t_fall is None and d.qpos[2] < 0.15:
            t_fall = step * 0.002

    print("  " + "-" * 50)
    if t_fall is None:
        print("  ✅ 静态站立成功（6 秒未倒）→ 模型+增益可支撑静态姿态")
    else:
        print(f"  ❌ 静态站立失败，{t_fall:.2f}s 后倒下 → 模型或增益有根本问题")
    print("=" * 84)


if __name__ == "__main__":
    main()
