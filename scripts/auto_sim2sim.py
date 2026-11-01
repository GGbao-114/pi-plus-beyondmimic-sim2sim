#!/usr/bin/env python
"""URDF -> MJCF -> sim2sim 一条龙，带完整容错。

已处理的坑（都是实际踩到的）:
  1. URDF 里 <material name=""> 空名字 -> MuJoCo 拒绝
  2. 生成的 MJCF 用相对 meshdir，移动后找不到 STL
  3. 生成的 MJCF 可能没有 gyro sensor
  4. actuator 数可能不等于 22（含头部关节等）
  5. actuator 可能是 position 型而非力矩型
  6. 关节顺序与训练不同（按名字映射，顺序无关）

用法:
  MUJOCO_GL=egl python auto_sim2sim.py \
    --urdf <urdf> --motion_file <npz> --policy_path <onnx> \
    --out_xml /data/gaoli/logs/pi_auto.xml --gif /data/gaoli/logs/auto.gif --duration 8
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import numpy as np
import mujoco
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

# 来自 robots/pi_plus.py 与 mjcf/pi_20dof.xml（URDF 里没有，必须补）
ARMATURE_LEG = 0.044277
ARMATURE_ARM = 0.008419


def dpos_of(n):
    if n.endswith("_hip_pitch_joint"): return -0.25
    if n.endswith("_calf_joint"): return 0.65
    if n.endswith("_ankle_pitch_joint"): return -0.4
    return 0.0


def gains(n):
    return (30.0, 0.6) if n.endswith(ARM_SUF) else (80.0, 1.1)


def scale_of(n):
    kp = gains(n)[0]
    return 0.25 * (20.0 if n.endswith(ARM_SUF) else 30.0) / kp


def apply_isaac_force_limits(m, joint_ids, actuator_joint_names):
    """Apply Isaac's effort limits at every MuJoCo limit boundary.

    Updating only ``actuator_ctrlrange`` is insufficient when the generated
    model retains a joint-level ``actuatorfrcrange`` or an actuator force
    range from the URDF conversion.
    """
    for actuator_id, joint_name in enumerate(actuator_joint_names):
        if joint_name not in BFS_JOINTS:
            continue
        limit = 20.0 if joint_name.endswith(ARM_SUF) else 30.0
        m.actuator_ctrlrange[actuator_id] = [-limit, limit]
        if hasattr(m, "actuator_ctrllimited"):
            m.actuator_ctrllimited[actuator_id] = 1

        if hasattr(m, "actuator_forcerange"):
            m.actuator_forcerange[actuator_id] = [-limit, limit]
        if hasattr(m, "actuator_forcelimited"):
            m.actuator_forcelimited[actuator_id] = 1

        joint_id = joint_ids[joint_name]
        if hasattr(m, "jnt_actfrcrange"):
            m.jnt_actfrcrange[joint_id] = [-limit, limit]
        if hasattr(m, "jnt_actfrclimited"):
            m.jnt_actfrclimited[joint_id] = 1


# ---------------- 1. 转换 ----------------
def find_cli():
    for p in ["/home/gaoli/.local/bin/urdf2mjcf",
              os.path.expanduser("~/.local/bin/urdf2mjcf"),
              shutil.which("urdf2mjcf")]:
        if p and os.path.isfile(p):
            return p
    return None


def convert_urdf(urdf, out_xml):
    cli = find_cli()
    print(f"[转换] CLI = {cli}")
    if cli is None:
        # 试试 python -m
        for py in ["/home/gaoli/.local/bin/python", "/opt/anaconda3/bin/python"]:
            if os.path.isfile(py):
                print(f"  尝试 {py} -m urdf2mjcf")
                try:
                    r = subprocess.run([py, "-m", "urdf2mjcf", urdf],
                                       capture_output=True, text=True, timeout=600)
                    print(f"    退出码 {r.returncode}: {(r.stderr or '')[:200]}")
                except Exception as e:
                    print(f"    异常 {e}")
    else:
        # 已知用法: urdf2mjcf <urdf>   -> 生成 <urdf同目录>/<name>.mjcf
        print(f"  执行: {cli} {urdf}")
        try:
            r = subprocess.run([cli, urdf], capture_output=True, text=True, timeout=600)
            print(f"  退出码 {r.returncode}")
            if r.stdout.strip():
                print(f"  stdout: {r.stdout.strip()[:400]}")
            if r.returncode != 0 and r.stderr.strip():
                print(f"  stderr: {r.stderr.strip()[:400]}")
        except Exception as e:
            print(f"  异常: {e}")

    guess = os.path.splitext(urdf)[0] + ".mjcf"
    if os.path.isfile(guess):
        os.makedirs(os.path.dirname(out_xml), exist_ok=True)
        shutil.copy(guess, out_xml)
        print(f"  ✅ 生成 {guess}  ->  {out_xml}")
        return out_xml
    print(f"  ❌ 未找到生成的 {guess}")
    return None


# ---------------- 2. 修复 XML ----------------
def find_mesh_dir(urdf, xml_path):
    cands = []
    if urdf:
        d = os.path.dirname(os.path.abspath(urdf))
        cands += [os.path.join(d, "meshes"), os.path.join(os.path.dirname(d), "meshes")]
    if xml_path:
        d = os.path.dirname(os.path.abspath(xml_path))
        for _ in range(4):
            cands.append(os.path.join(d, "meshes"))
            d = os.path.dirname(d)
            if d in ("/", ""):
                break
    for c in cands:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.STL")):
            return c
    return None


def sanitize(xml_path, mesh_dir):
    with open(xml_path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    orig = txt
    n_mat = len(re.findall(r'<material\b', txt))

    # (1) 删除 material（URDF 的空 name 会导致 MuJoCo 报错）
    txt = re.sub(r'<material\b[^>]*/>', '', txt)
    txt = re.sub(r'<material\b[^>]*>.*?</material>', '', txt, flags=re.S)
    txt = re.sub(r'\bmaterial\s*=\s*"[^"]*"', '', txt)

    # (2) mesh 文件名去掉 meshes/ 前缀，交给 meshdir 解析
    txt = re.sub(r'(<mesh\b[^>]*?\bfile\s*=\s*")[^"]*?meshes/', r'\1', txt)

    # (3) 绝对 meshdir
    if mesh_dir:
        md = os.path.abspath(mesh_dir).replace("\\", "/")
        if re.search(r'\bmeshdir\s*=', txt):
            txt = re.sub(r'\bmeshdir\s*=\s*"[^"]*"', f'meshdir="{md}"', txt, count=1)
        elif re.search(r'<compiler\b', txt):
            txt = re.sub(r'(<compiler\b)', lambda m: m.group(1) + f' meshdir="{md}"', txt, count=1)
        else:
            txt = re.sub(r'(<mujoco\b[^>]*>)',
                         lambda m: m.group(1) + f'\n  <compiler meshdir="{md}"/>', txt, count=1)

    # (4) 补地面（URDF 无 worldbody 地面，urdf2mjcf 也不会生成）
    if 'type="plane"' not in txt and "<worldbody" in txt:
        floor = ('\n    <geom name="floor" type="plane" size="0 0 0.05" '
                 'contype="15" conaffinity="15" rgba="0.4 0.4 0.45 1"/>')
        txt = re.sub(r'(<worldbody\b[^>]*>)', lambda m: m.group(1) + floor, txt, count=1)
        print("  [修复] 已补地面平面")

    if txt != orig:
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"  [修复] 删除 material {n_mat} 个; meshdir -> {mesh_dir}")
    else:
        print("  [修复] 无需改动")
    return xml_path


# ---------------- 3. 仿真 ----------------
def simulate(xml_path, jp, jv, bp, bq, sess, duration, render=False, w=480, h=360,
             use_gyro_sensor=False, align_force_limits=True, initial_policy=True,
             sim_dt=None, control_dec=None, target_rate_limit=False,
             passive_damping_scale=1.0, frictionloss_scale=1.0,
             position_actuator=False, qvel_clamp=False, integrator=None):
    m = mujoco.MjModel.from_xml_path(xml_path)
    sim_dt = SIM_DT if sim_dt is None else sim_dt
    control_dec = CTRL_DEC if control_dec is None else control_dec
    m.opt.timestep = sim_dt
    if integrator is not None:
        m.opt.integrator = getattr(mujoco.mjtIntegrator, integrator)

    # Motor mode is injected for the explicit-PD baseline. A compiled
    # position-actuator XML must retain its native gain/bias tables.
    if not position_actuator:
        m.actuator_gainprm[:, 0] = 1.0
        m.actuator_biasprm[:, :] = 0.0
    m.dof_damping[:] *= passive_damping_scale
    m.dof_frictionloss[:] *= frictionloss_scale
    d = mujoco.MjData(m)

    # --- 关节（只取铰链）---
    jname, jqadr, jdadr = {}, {}, {}
    for jid in range(m.njnt):
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if nm is None:
            continue
        jname[nm] = jid
        jqadr[nm] = int(m.jnt_qposadr[jid])
        jdadr[nm] = int(m.jnt_dofadr[jid])

    missing = [j for j in BFS_JOINTS if j not in jname]
    extra = [j for j in jname if j not in BFS_JOINTS]
    print(f"  铰链关节 {len(jname)} 个, 执行器 {m.nu} 个")
    if missing:
        print(f"  [!] 缺少训练用的关节: {missing}")
        return None
    if extra:
        print(f"  [i] 额外关节(不控制): {extra}")

    # ---- 关键：补 armature（URDF 无此概念，urdf2mjcf 生成为 0）----
    # 没有 armature，手腕等小惯量关节的 PD 自然频率会远超奈奎斯特频率 -> 数值发散
    n_arm = 0
    for jid in range(m.njnt):
        if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if nm is None:
            continue
        a = ARMATURE_ARM if nm.endswith(ARM_SUF) else ARMATURE_LEG
        m.dof_armature[m.jnt_dofadr[jid]] = a
        # 关节阻尼（对齐手写 MJCF 的 0.02~0.05）
        if m.dof_damping[m.jnt_dofadr[jid]] < 0.01:
            m.dof_damping[m.jnt_dofadr[jid]] = 0.05
        n_arm += 1
    print(f"  [armature] {n_arm} 个关节: 腿={ARMATURE_LEG} 臂={ARMATURE_ARM} (URDF 里缺失，已补)")
    # Apply experimental passive terms after the legacy fallback assignment so
    # a zero/scale experiment cannot silently be overwritten.
    m.dof_damping[:] *= passive_damping_scale
    m.dof_frictionloss[:] *= frictionloss_scale

    # ---- 力矩上限对齐 Isaac ----
    act_names_all = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0]))
                     for i in range(m.nu)]
    if align_force_limits:
        apply_isaac_force_limits(m, jname, act_names_all)
    else:
        for i, name in enumerate(act_names_all):
            lim = 20.0 if (name or "").endswith(ARM_SUF) else 30.0
            m.actuator_ctrlrange[i] = [-lim, lim]
    print(f"  [力矩限制] {'all boundaries' if align_force_limits else 'legacy ctrlrange only'}")

    # --- 执行器 -> BFS 索引 ---
    act_joint = []
    for i in range(m.nu):
        jid = int(m.actuator_trnid[i, 0])
        act_joint.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))
    valid = [i for i in range(m.nu) if act_joint[i] in BFS_JOINTS]
    a2b = np.array([BFS_JOINTS.index(act_joint[i]) for i in valid])
    aq = np.array([jqadr[act_joint[i]] for i in valid])
    ad = np.array([jdadr[act_joint[i]] for i in valid])
    kp_a = np.array([gains(BFS_JOINTS[b])[0] for b in a2b])
    kd_a = np.array([gains(BFS_JOINTS[b])[1] for b in a2b])
    if position_actuator and np.all(m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE):
        m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[:, :] = 0.0
        m.actuator_biasprm[:, :] = 0.0
        m.actuator_gainprm[valid, 0] = kp_a
        m.actuator_biasprm[valid, 1] = -kp_a
        m.actuator_biasprm[valid, 2] = -kd_a
    print(f"  受控执行器 {len(valid)}/{m.nu}")

    # --- 传感器兜底 ---
    has_sensor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity") >= 0
    use_sensor = use_gyro_sensor and has_sensor
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    source = "gyro sensor" if use_sensor else "qvel[3:6]"
    suffix = " (sensor unavailable)" if use_gyro_sensor and not has_sensor else ""
    print(f"  gyro source: {source}{suffix}, base body id={bid}")

    dpos_b = np.array([dpos_of(j) for j in BFS_JOINTS])
    sc_b = np.array([scale_of(j) for j in BFS_JOINTS])

    def qpos_to_bfs():
        return np.array([d.qpos[jqadr[n]] for n in BFS_JOINTS])

    def qvel_to_bfs():
        return np.array([d.qvel[jdadr[n]] for n in BFS_JOINTS])

    # --- 初始状态 ---
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.qpos[0:3] = bp[0, 0]
    q0 = bq[0, 0].copy()
    d.qpos[3:7] = q0 / np.linalg.norm(q0)
    for n in BFS_JOINTS:
        d.qpos[jqadr[n]] = jp[0][BFS_JOINTS.index(n)]
    mujoco.mj_forward(m, d)
    print(f"  初始基座高度 {d.qpos[2]:.4f} (npz {bp[0,0,2]:.4f}), 接触数 {d.ncon}")

    target_b = jp[0].copy()
    act_buf = np.zeros(22, dtype=np.float32)
    n_frames = min(len(jp), len(bp))

    renderer, frames = None, []
    if render:
        try:
            renderer = mujoco.Renderer(m, height=h, width=w)
        except Exception as e:
            print(f"  [!] 渲染失败: {str(e)[:70]}")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = bid
    cam.distance = 2.6
    cam.azimuth = 130
    cam.elevation = -12

    fall_t, min_h, errs, hs = None, 9.9, [], []
    control_dt = sim_dt * control_dec
    n_ctrl = int(duration / control_dt)
    if n_ctrl < 1:
        raise ValueError("duration must include at least one control period")
    traces = {key: [] for key in (
        "time", "joint_pos", "joint_vel", "requested_tau", "effective_tau",
        "base_quat", "base_omega", "ncon", "contact_force_norm_max",
    )}
    max_speed = np.zeros(len(BFS_JOINTS))
    control_peak = np.zeros(m.nu)
    effective_peak = np.zeros(m.nu)
    velocity_limits = np.array([
        17.0 if n.endswith(ARM_SUF) or "_ankle_" in n else 8.0
        for n in BFS_JOINTS
    ])
    velocity_exceed_count = np.zeros(len(BFS_JOINTS), dtype=int)
    saturation_count = np.zeros(m.nu, dtype=int)
    effective_params = {
        "timestep": float(m.opt.timestep),
        "control_dec": int(control_dec),
        "position_actuator": bool(position_actuator),
        "target_rate_limit": bool(target_rate_limit),
        "qvel_clamp": bool(qvel_clamp),
        "dof_damping": m.dof_damping.copy().tolist(),
        "dof_frictionloss": m.dof_frictionloss.copy().tolist(),
        "dof_armature": m.dof_armature.copy().tolist(),
        "actuator_gaintype": m.actuator_gaintype.tolist(),
        "actuator_biastype": m.actuator_biastype.tolist(),
        "actuator_gainprm": m.actuator_gainprm[:, :3].copy().tolist(),
        "actuator_biasprm": m.actuator_biasprm[:, :3].copy().tolist(),
        "actuator_ctrlrange": m.actuator_ctrlrange.copy().tolist(),
    }

    def infer(fi):
        """Infer before the next control window, matching Isaac's reset loop."""
        nonlocal target_b, act_buf
        q_b = qpos_to_bfs()
        dq_b = qvel_to_bfs()

        quat = d.qpos[3:7].copy()
        if use_sensor:
            omega = d.sensor("angular-velocity").data.copy()
        else:
            # MuJoCo's free-joint qvel[3:6] is body-frame angular velocity.
            omega = d.qvel[3:6].copy()

        qq = bq[fi, 0]
        w1, x1, y1, z1 = quat[0], -quat[1], -quat[2], -quat[3]
        w2, x2, y2, z2 = qq
        qm = np.array([
            w1*w2-x1*x2-y1*y2-z1*z2,
            w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2,
            w1*z2+x1*y2-y1*x2+z1*w2,
        ])
        qm /= np.linalg.norm(qm)
        r, i, j, k = qm
        s2 = 2.0 / (qm*qm).sum()
        mat = np.array([
            [1-s2*(j*j+k*k), s2*(i*j-k*r), s2*(i*k+j*r)],
            [s2*(i*j+k*r), 1-s2*(i*i+k*k), s2*(j*k-i*r)],
            [s2*(i*k-j*r), s2*(j*k+i*r), 1-s2*(i*i+j*j)],
        ])
        obs = np.zeros(119, dtype=np.float32)
        obs[0:44] = np.concatenate((jp[fi], jv[fi]))
        obs[44:50] = mat[:, :2].reshape(6)
        obs[50:53] = omega
        obs[53:75] = q_b - dpos_b
        obs[75:97] = dq_b
        obs[97:119] = act_buf
        out = sess.run(["actions"], {
            "obs": obs.reshape(1, -1),
            "time_step": np.array([[float(fi)]], dtype=np.float32),
        })[0]
        act_buf = np.asarray(out).reshape(-1).astype(np.float32)
        target_b = act_buf * sc_b + dpos_b

    # Isaac computes action_0 from obs_0 before the first physics step.
    if initial_policy:
        infer(0)

    for step in range(n_ctrl * control_dec):
        if target_rate_limit:
            q_b_now = qpos_to_bfs()
            target_b = np.clip(target_b, q_b_now - velocity_limits * control_dt,
                               q_b_now + velocity_limits * control_dt)
        tau = np.zeros(m.nu)
        tau[valid] = (target_b[a2b] - d.qpos[aq]) * kp_a - d.qvel[ad] * kd_a
        if position_actuator:
            d.ctrl[:] = 0.0
            d.ctrl[valid] = target_b[a2b]
        else:
            d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        if qvel_clamp:
            d.qvel[ad] = np.clip(d.qvel[ad], -velocity_limits[a2b], velocity_limits[a2b])
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise FloatingPointError(f"nonfinite state at physics step {step}")
        speed = np.abs(qvel_to_bfs())
        max_speed = np.maximum(max_speed, speed)
        velocity_exceed_count += speed > velocity_limits
        control_peak = np.maximum(control_peak, np.abs(tau))
        effective_peak[valid] = np.maximum(effective_peak[valid], np.abs(d.qfrc_actuator[ad]))
        saturation_count[valid] += np.abs(tau[valid]) >= 0.98 * m.actuator_ctrlrange[valid, 1]

        if (step + 1) % control_dec == 0:
            t = (step + 1) // control_dec
            fi = min(t, n_frames - 1)
            hgt = d.qpos[2]
            min_h = min(min_h, hgt)
            hs.append(hgt)
            if fall_t is None and hgt < 0.15:
                fall_t = t * control_dt

            q_b = qpos_to_bfs()
            errs.append(np.abs(q_b - jp[fi]).mean())
            traces["time"].append(t * control_dt)
            traces["joint_pos"].append(q_b.copy())
            traces["joint_vel"].append(qvel_to_bfs())
            traces["requested_tau"].append(tau[valid].copy())
            traces["effective_tau"].append(d.qfrc_actuator[ad].copy())
            traces["base_quat"].append(d.qpos[3:7].copy())
            traces["base_omega"].append(d.qvel[3:6].copy())
            traces["ncon"].append(d.ncon)
            force_norms = []
            for ci in range(d.ncon):
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, force)
                force_norms.append(float(np.linalg.norm(force[:3])))
            traces["contact_force_norm_max"].append(max(force_norms, default=0.0))
            infer(fi)

            if renderer is not None and t % 2 == 0:
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())

    if renderer is not None:
        renderer.close()
    elapsed = n_ctrl * control_dt
    prefall = np.array(traces["time"]) < (fall_t if fall_t is not None else elapsed + 1e-9)
    return dict(fall=fall_t if fall_t is not None else elapsed, fell=fall_t is not None,
                min_h=min_h, jerr=float(np.mean(errs)),
                jerr_prefall=float(np.mean(np.array(errs)[prefall])) if prefall.any() else None,
                heights=np.array(hs), frames=frames,
                traces={k: np.asarray(v) for k, v in traces.items()},
                max_joint_speed=max_speed, velocity_limits=velocity_limits,
                velocity_exceed_fraction=velocity_exceed_count / (n_ctrl * control_dec),
                requested_tau_peak=control_peak, effective_tau_peak=effective_peak,
                saturation_fraction=saturation_count / (n_ctrl * control_dec),
                joint_names=BFS_JOINTS, actuator_names=[act_joint[i] for i in valid],
                effective_params=effective_params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--motion_file", required=True)
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--out_xml", default="/data/gaoli/logs/pi_auto.xml")
    ap.add_argument("--gif", default="/data/gaoli/logs/auto_sim2sim.gif")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--use_gyro_sensor", action="store_true",
                    help="使用 MJCF gyro sensor；默认使用已验证的 qvel[3:6] 真值")
    args = ap.parse_args()

    print("=" * 82)
    print(" URDF -> MJCF -> sim2sim 一条龙")
    print("=" * 82)

    if not os.path.isfile(args.out_xml):
        if convert_urdf(os.path.abspath(args.urdf), args.out_xml) is None:
            return
    else:
        print(f"[跳过转换] 已存在 {args.out_xml}")

    mesh_dir = find_mesh_dir(args.urdf, args.out_xml)
    print(f"\n[修复 XML] mesh 目录 = {mesh_dir}")
    sanitize(args.out_xml, mesh_dir)

    mo = np.load(args.motion_file)
    jp, jv, bp, bq = mo["joint_pos"], mo["joint_vel"], mo["body_pos_w"], mo["body_quat_w"]
    sess = onnxruntime.InferenceSession(args.policy_path, providers=["CPUExecutionProvider"])

    print("\n" + "=" * 82)
    print(f" 仿真 {args.duration}s")
    print("=" * 82)
    r = simulate(args.out_xml, jp, jv, bp, bq, sess, args.duration,
                 render=False, use_gyro_sensor=args.use_gyro_sensor)
    if r is None:
        print("[中止] 模型构建失败")
        return

    tag = "撑满!" if r["fall"] >= args.duration - 1e-6 else f"{r['fall']:.2f}s"
    print(f"\n  撑住时间 : {tag}      (Isaac 参考 9.80s)")
    print(f"  最低高度 : {r['min_h']:.4f} m")
    print(f"  关节误差 : {r['jerr']:.4f} rad")
    print("\n  高度轨迹:")
    hh = r["heights"]
    step_i = max(1, len(hh) // 20)
    for i in range(0, len(hh), step_i):
        print(f"    {i*0.02:5.2f}s  {hh[i]:.4f}  {'#'*int(max(0,hh[i])*100)}")

    print("\n[生成 GIF]")
    r2 = simulate(args.out_xml, jp, jv, bp, bq, sess, args.duration,
                  render=True, use_gyro_sensor=args.use_gyro_sensor)
    if r2 and r2["frames"]:
        try:
            from PIL import Image
            fr = r2["frames"]
            sel = list(range(0, len(fr), 2))
            imgs = [Image.fromarray(fr[i]) for i in sel]
            imgs[0].save(args.gif, save_all=True, append_images=imgs[1:],
                         duration=40, loop=0)
            print(f"[OK] {args.gif}  ({len(imgs)} 帧, {len(imgs)*0.04:.1f}s)")
        except Exception as e:
            print(f"[GIF 失败] {e}")
    else:
        print("[GIF] 无渲染帧")
    print("=" * 82)


if __name__ == "__main__":
    main()
