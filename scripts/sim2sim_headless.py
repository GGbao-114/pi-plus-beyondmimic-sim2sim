#!/usr/bin/env python
"""无头版 sim2sim v3：从 MuJoCo 模型自省关节顺序（不硬编码），并诊断力矩饱和。

用法:
  MUJOCO_GL=egl python sim2sim_headless.py \
    --motion_file <npz> --xml_path <mjcf> --policy_path <onnx> \
    --out_gif <gif> --duration 8 [--leg_torque 30] [--arm_torque 20]
"""
import argparse
import os
import numpy as np
import mujoco
import onnx
import onnxruntime

SIM_DT = 0.002
CTRL_DECIMATION = 10

# 训练顺序 = BFS（PhysX articulation 树遍历）
# 已验证: BFS 误差 0.0002m vs CSV 误差 0.0914m
TRAIN_JOINTS = [
    "l_hip_pitch_joint", "l_shoulder_pitch_joint", "r_hip_pitch_joint", "r_shoulder_pitch_joint",
    "l_hip_roll_joint", "l_shoulder_roll_joint", "r_hip_roll_joint", "r_shoulder_roll_joint",
    "l_thigh_joint", "l_upper_arm_joint", "r_thigh_joint", "r_upper_arm_joint",
    "l_calf_joint", "l_elbow_joint", "r_calf_joint", "r_elbow_joint",
    "l_ankle_pitch_joint", "l_wrist_joint", "r_ankle_pitch_joint", "r_wrist_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
]
ARM_SUFFIX = ("_shoulder_pitch_joint", "_shoulder_roll_joint",
              "_upper_arm_joint", "_elbow_joint", "_wrist_joint")


def is_arm(n):
    return n.endswith(ARM_SUFFIX)


def apply_isaac_force_limits(m, actuator_names):
    """Apply Isaac's 30/20 Nm limits to ctrl, actuator, and joint ranges."""
    for actuator_id, joint_name in enumerate(actuator_names):
        if joint_name not in TRAIN_JOINTS:
            continue
        limit = 20.0 if is_arm(joint_name) else 30.0
        m.actuator_ctrlrange[actuator_id] = [-limit, limit]
        if hasattr(m, "actuator_ctrllimited"):
            m.actuator_ctrllimited[actuator_id] = 1
        if hasattr(m, "actuator_forcerange"):
            m.actuator_forcerange[actuator_id] = [-limit, limit]
        if hasattr(m, "actuator_forcelimited"):
            m.actuator_forcelimited[actuator_id] = 1
        jid = int(m.actuator_trnid[actuator_id, 0])
        if hasattr(m, "jnt_actfrcrange"):
            m.jnt_actfrcrange[jid] = [-limit, limit]
        if hasattr(m, "jnt_actfrclimited"):
            m.jnt_actfrclimited[jid] = 1


def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def mat_from_quat(q):
    r, i, j, k = q
    s = 2.0 / (q * q).sum()
    return np.array([
        [1 - s*(j*j + k*k), s*(i*j - k*r), s*(i*k + j*r)],
        [s*(i*j + k*r), 1 - s*(i*i + k*k), s*(j*k - i*r)],
        [s*(i*k - j*r), s*(j*k + i*r), 1 - s*(i*i + j*j)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_file", required=True)
    ap.add_argument("--xml_path", required=True)
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--out_gif", default="/data/gaoli/logs/sim2sim.gif")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--gif_fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--use_gyro_sensor", action="store_true",
                    help="使用 MJCF gyro sensor；默认使用已验证的 qvel[3:6] 真值")
    # 力矩上限覆盖（用于测试 sim2sim gap）
    ap.add_argument("--leg_torque", type=float, default=None,
                    help="覆盖腿关节力矩上限 (MJCF 默认 20)")
    ap.add_argument("--arm_torque", type=float, default=None,
                    help="覆盖臂关节力矩上限 (MJCF 默认 10)")
    args = ap.parse_args()

    # ================= 动作 =================
    mo = np.load(args.motion_file)
    m_jp, m_jv = mo["joint_pos"], mo["joint_vel"]
    m_bp, m_bq = mo["body_pos_w"], mo["body_quat_w"]
    n_frames = min(len(m_jp), len(m_bp))
    print(f"[INFO] 动作帧数 {n_frames}")

    # ================= ONNX =================
    model = onnx.load(args.policy_path)
    meta = {p.key: p.value for p in model.metadata_props}
    jn = meta["joint_names"].split(",")
    dpos_t = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
    kp_t = np.array([float(x) for x in meta["joint_stiffness"].split(",")])
    kd_t = np.array([float(x) for x in meta["joint_damping"].split(",")])
    sc_t = np.array([float(x) for x in meta["action_scale"].split(",")])

    if jn != TRAIN_JOINTS:
        print("[警告] 元数据顺序 != 标准训练顺序，按名字重排")
        k = [jn.index(j) for j in TRAIN_JOINTS]
        dpos_t, kp_t, kd_t, sc_t = dpos_t[k], kp_t[k], kd_t[k], sc_t[k]
    else:
        print("[OK] 元数据顺序 = 训练顺序")

    # ================= MuJoCo =================
    m = mujoco.MjModel.from_xml_path(args.xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = SIM_DT
    print(f"[INFO] MJCF nq={m.nq} nv={m.nv} nu={m.nu} njnt={m.njnt}")

    # ---- 自省：关节顺序 ----
    xml_names, qadr, dadr = [], [], []
    extra_xml_names = []
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if nm not in TRAIN_JOINTS:
            extra_xml_names.append(nm)
            continue
        xml_names.append(nm)
        qadr.append(int(m.jnt_qposadr[jid]))
        dadr.append(int(m.jnt_dofadr[jid]))
    qadr = np.array(qadr)
    dadr = np.array(dadr)
    print(f"[INFO] MJCF 关节数 {len(xml_names)}")
    if extra_xml_names:
        print(f"[INFO] 跳过额外非训练关节: {extra_xml_names}")

    # ---- 自省：执行器 -> 关节 ----
    act_names = []
    for i in range(m.nu):
        jid = int(m.actuator_trnid[i, 0])
        act_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
    print(f"[INFO] 执行器数 {m.nu}")

    # Match Isaac's effort limits before applying optional experiment overrides.
    apply_isaac_force_limits(m, act_names)
    print("[INFO] Isaac effort limits applied: legs +/-30 Nm, arms +/-20 Nm")

    # ---- 建立映射（全部基于名字，不硬编码）----
    missing = [j for j in TRAIN_JOINTS if j not in xml_names]
    if missing:
        print(f"[错误] MJCF 缺少关节: {missing}")
        return

    # 训练序 -> xml序
    t2x = np.array([xml_names.index(j) for j in TRAIN_JOINTS])
    # xml序 -> 训练序
    x2t = np.array([TRAIN_JOINTS.index(j) for j in xml_names])
    # 执行器序 -> 训练序
    valid = [i for i, name in enumerate(act_names) if name in TRAIN_JOINTS]
    invalid_act_names = [name for name in act_names if name not in TRAIN_JOINTS]
    if invalid_act_names:
        print(f"[INFO] 跳过额外执行器: {invalid_act_names}")
    a2t = np.array([TRAIN_JOINTS.index(act_names[i]) for i in valid])
    joint_addr = {name: (int(m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, name)]),
                         int(m.jnt_dofadr[mujoco.mj_name2id(
                             m, mujoco.mjtObj.mjOBJ_JOINT, name)]))
                  for name in TRAIN_JOINTS}
    act_qadr = np.array([joint_addr[act_names[i]][0] for i in valid])
    act_dadr = np.array([joint_addr[act_names[i]][1] for i in valid])

    # 显示映射，便于核对
    print("\n[映射] 执行器序 -> 关节名 -> 训练序索引")
    for i, nm in enumerate(act_names):
        if i in valid:
            print(f"   ctrl[{i:2d}] {nm:26s} -> train[{a2t[valid.index(i)]:2d}]")
        else:
            print(f"   ctrl[{i:2d}] {nm:26s} -> skipped")

    # ---- 力矩上限覆盖 ----
    if args.leg_torque or args.arm_torque:
        print("\n[覆盖] 修改力矩上限")
        for i, nm in enumerate(act_names):
            if i not in valid:
                continue
            lim = args.arm_torque if is_arm(nm) else args.leg_torque
            if lim is None:
                continue
            m.actuator_ctrlrange[i] = [-lim, lim]
            if hasattr(m, "actuator_forcerange"):
                m.actuator_forcerange[i] = [-lim, lim]
            jid = int(m.actuator_trnid[i, 0])
            if hasattr(m, "jnt_actfrcrange"):
                m.jnt_actfrcrange[jid] = [-lim, lim]
            print(f"   {nm:26s} -> ±{lim}")

    # 打印当前有效上限
    print("\n[力矩上限]")
    for i, nm in enumerate(act_names[:6]):
        print(f"   {nm:26s} ctrl={m.actuator_ctrlrange[i]}")
    print(f"   ... (共 {m.nu})")
    print(f"   {act_names[6]:26s} ctrl={m.actuator_ctrlrange[6]}")

    kp_a = kp_t[a2t]        # 有效执行器序
    kd_a = kd_t[a2t]

    # ---- 初始状态 = 动作第 0 帧 ----
    d.qpos[:] = 0.0
    d.qpos[0:3] = m_bp[0, 0]
    q0 = m_bq[0, 0].copy()
    d.qpos[3:7] = q0 / np.linalg.norm(q0)
    # xml序 -> 训练序取值
    for i, nm in enumerate(xml_names):
        d.qpos[qadr[i]] = m_jp[0][TRAIN_JOINTS.index(nm)]
    mujoco.mj_forward(m, d)
    print(f"\n[INFO] 初始基座高度 {d.qpos[2]:.3f} m")
    print(f"[INFO] 动作第0帧基座高度 {m_bp[0,0,2]:.3f} m")

    sess = onnxruntime.InferenceSession(args.policy_path,
                                        providers=["CPUExecutionProvider"])

    n_ctrl = int(args.duration / (SIM_DT * CTRL_DECIMATION))
    n_sim = n_ctrl * CTRL_DECIMATION

    target_t = dpos_t.copy()
    action_buf = np.zeros(22, dtype=np.float32)

    def infer(fi):
        """Infer before the next control window, matching Isaac's reset loop."""
        nonlocal target_t, action_buf
        quat = d.qpos[3:7].copy()
        if args.use_gyro_sensor and mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity") >= 0:
            omega = d.sensor("angular-velocity").data.copy()
        else:
            omega = d.qvel[3:6].copy()

        q_train = d.qpos[qadr][x2t]
        dq_train = d.qvel[dadr][x2t]
        obs = np.zeros(119, dtype=np.float32)
        o = 0
        obs[o:o+44] = np.concatenate((m_jp[fi], m_jv[fi])); o += 44
        qr = quat_mul(quat_inv(quat), m_bq[fi, 0])
        qr /= np.linalg.norm(qr)
        obs[o:o+6] = mat_from_quat(qr)[:, :2].reshape(6); o += 6
        obs[o:o+3] = omega; o += 3
        obs[o:o+22] = q_train - dpos_t; o += 22
        obs[o:o+22] = dq_train; o += 22
        obs[o:o+22] = action_buf

        out = sess.run(["actions"], {
            "obs": obs.reshape(1, -1),
            "time_step": np.array([[float(fi)]], dtype=np.float32),
        })[0]
        action_buf = np.asarray(out).reshape(-1).astype(np.float32)
        target_t = action_buf * sc_t + dpos_t

    # Isaac computes action_0 from obs_0 before the first physics step.
    infer(0)

    rec_base, rec_ref, rec_jerr, rec_tau = [], [], [], []
    frames = []
    sat_count = 0
    total_ctrl = 0

    renderer = None
    if not args.no_render:
        try:
            renderer = mujoco.Renderer(m, height=args.height, width=args.width)
            print("[INFO] 离屏渲染可用")
        except Exception as e:
            print(f"[警告] 渲染不可用: {str(e)[:80]}")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    cam.distance = 2.6
    cam.azimuth = 130
    cam.elevation = -12

    t = 0
    print(f"[INFO] 开始仿真 {args.duration}s / {n_ctrl} 控制步\n")

    for step in range(n_sim):
        # 控制：目标(训练序) -> 执行器序
        tgt_a = target_t[a2t]
        q_a = d.qpos[act_qadr]
        dq_a = d.qvel[act_dadr]
        tau = np.zeros(m.nu)
        tau[valid] = (tgt_a - q_a) * kp_a - dq_a * kd_a
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        if (step + 1) % CTRL_DECIMATION == 0:
            t += 1
            fi = min(t, n_frames - 1)

            # 力矩饱和统计
            for i in valid:
                lim = m.actuator_ctrlrange[i][1]
                if abs(tau[i]) >= 0.98 * lim:
                    sat_count += 1
            total_ctrl += len(valid)
            rec_tau.append(np.abs(tau).copy())

            # xml序 -> 训练序
            q_train = d.qpos[qadr][x2t]

            rec_base.append(d.qpos[0:3].copy())
            rec_ref.append(m_bp[fi, 0].copy())
            rec_jerr.append(np.abs(q_train - m_jp[fi]).mean())
            infer(fi)

            if renderer is not None and t % 2 == 0:
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())

    # ================= 评估 =================
    rec_base = np.array(rec_base)
    rec_ref = np.array(rec_ref)
    rec_jerr = np.array(rec_jerr)
    rec_tau = np.array(rec_tau)
    off = rec_base[0] - rec_ref[0]
    off[2] = 0.0
    e_xy = np.linalg.norm((rec_base - off) - rec_ref, axis=1)
    e_z = np.abs(rec_base[:, 2] - rec_ref[:, 2])
    min_h = rec_base[:, 2].min()
    sat_pct = 100.0 * sat_count / max(total_ctrl, 1)

    print("=" * 56)
    print(" sim2sim 结果")
    print("=" * 56)
    print(f"  关节跟踪误差   : {rec_jerr.mean():.4f} rad")
    print(f"  水平漂移       : {e_xy.mean():.4f} m  (最大 {e_xy.max():.4f})")
    print(f"  高度误差       : {e_z.mean():.4f} m")
    print(f"  高度           : {rec_base[0,2]:.3f} -> {rec_base[-1,2]:.3f} m")
    print(f"  最低高度       : {min_h:.3f} m")
    print(f"  判定           : {'摔倒 X' if min_h < 0.15 else '站立 OK'}")
    print("-" * 56)
    print(f"  【力矩诊断】")
    print(f"  平均|tau|      : {rec_tau.mean():.2f} N·m   峰值 {rec_tau.max():.2f}")
    leg_idx = [i for i, name in enumerate(act_names) if name in TRAIN_JOINTS and not is_arm(name)]
    arm_idx = [i for i, name in enumerate(act_names) if name in TRAIN_JOINTS and is_arm(name)]
    print(f"  腿平均|tau|    : {rec_tau[:, leg_idx].mean():.2f}  (上限 30)")
    print(f"  臂平均|tau|    : {rec_tau[:, arm_idx].mean():.2f}  (上限 20)")
    print(f"  饱和比例       : {sat_pct:.1f}%   (>20% 说明力矩不够)")
    print("=" * 56)

    stem = os.path.splitext(args.out_gif)[0]
    np.savez(stem + "_data.npz", base=rec_base, ref=rec_ref,
             jerr=rec_jerr, tau=rec_tau)
    print(f"[INFO] 数据: {stem}_data.npz")

    if renderer is not None and frames:
        try:
            from PIL import Image
            pick = max(1, int(25 / args.gif_fps))
            sel = list(range(0, len(frames), pick))
            imgs = [Image.fromarray(frames[i]) for i in sel]
            imgs[0].save(args.out_gif, save_all=True, append_images=imgs[1:],
                         duration=int(1000/args.gif_fps), loop=0)
            print(f"[INFO] GIF: {args.out_gif} ({len(imgs)} 帧)")
        except Exception as e:
            print(f"[警告] GIF 失败: {e}")
    print("\n完成。")


if __name__ == "__main__":
    main()
