#!/usr/bin/env python
"""给导出的 ONNX 补上 sim2sim 需要的元数据。

【重要】关节顺序 = BFS（PhysX articulation 树遍历），
        已用正运动学验证：BFS 误差 0.0002m vs CSV 误差 0.0914m。

BFS 顺序:
  [0]  l_hip_pitch        [1]  l_shoulder_pitch    [2]  r_hip_pitch       [3]  r_shoulder_pitch
  [4]  l_hip_roll         [5]  l_shoulder_roll     [6]  r_hip_roll        [7]  r_shoulder_roll
  [8]  l_thigh            [9]  l_upper_arm         [10] r_thigh           [11] r_upper_arm
  [12] l_calf             [13] l_elbow             [14] r_calf            [15] r_elbow
  [16] l_ankle_pitch      [17] l_wrist             [18] r_ankle_pitch     [19] r_wrist
  [20] l_ankle_roll       [21] r_ankle_roll

用法:
  python add_metadata.py <onnx文件>
"""
import sys
import os
import onnx

# ===== 训练顺序 = BFS（PhysX 树遍历）=====
TRAIN_JOINTS = [
    "l_hip_pitch_joint", "l_shoulder_pitch_joint", "r_hip_pitch_joint", "r_shoulder_pitch_joint",
    "l_hip_roll_joint", "l_shoulder_roll_joint", "r_hip_roll_joint", "r_shoulder_roll_joint",
    "l_thigh_joint", "l_upper_arm_joint", "r_thigh_joint", "r_upper_arm_joint",
    "l_calf_joint", "l_elbow_joint", "r_calf_joint", "r_elbow_joint",
    "l_ankle_pitch_joint", "l_wrist_joint", "r_ankle_pitch_joint", "r_wrist_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
]

STIFF_LEG, DAMP_LEG, EFFORT_LEG = 80.0, 1.1, 30.0
STIFF_ARM, DAMP_ARM, EFFORT_ARM = 30.0, 0.6, 20.0
ARM_SUFFIX = ("_shoulder_pitch_joint", "_shoulder_roll_joint",
              "_upper_arm_joint", "_elbow_joint", "_wrist_joint")


def default_pos(name):
    if name.endswith("_hip_pitch_joint"):
        return -0.25
    if name.endswith("_calf_joint"):
        return 0.65
    if name.endswith("_ankle_pitch_joint"):
        return -0.4
    return 0.0


def main():
    if len(sys.argv) < 2:
        print("用法: python add_metadata.py <onnx文件>")
        return 1
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"[错误] 不存在: {path}")
        return 1

    print(f"处理: {path}")
    model = onnx.load(path, load_external_data=True)
    print(f"  节点数: {len(model.graph.node)}  原元数据: {len(model.metadata_props)}")

    while len(model.metadata_props) > 0:
        model.metadata_props.pop()

    names, dpos, kps, kds, scales = [], [], [], [], []
    for j in TRAIN_JOINTS:
        arm = j.endswith(ARM_SUFFIX)
        kp = STIFF_ARM if arm else STIFF_LEG
        kd = DAMP_ARM if arm else DAMP_LEG
        ef = EFFORT_ARM if arm else EFFORT_LEG
        names.append(j)
        dpos.append(default_pos(j))
        kps.append(kp)
        kds.append(kd)
        scales.append(0.25 * ef / kp)

    def csv(v, nd=6):
        return ",".join(f"{x:.{nd}f}" for x in v)

    metadata = {
        "joint_names": ",".join(names),
        "default_joint_pos": csv(dpos),
        "joint_stiffness": csv(kps),
        "joint_damping": csv(kds),
        "action_scale": csv(scales),
        "anchor_body_name": "base_link",
        "body_names": "base_link,l_hip_roll_link,l_calf_link,l_ankle_roll_link,"
                      "r_hip_roll_link,r_calf_link,r_ankle_roll_link,"
                      "l_shoulder_roll_link,l_elbow_link,l_wrist_link,"
                      "r_shoulder_roll_link,r_elbow_link,r_wrist_link",
        "command_names": "motion",
        "observation_names": "command,motion_anchor_ori_b,base_ang_vel,"
                             "joint_pos,joint_vel,actions",
        "run_path": "none",
    }
    for k, v in metadata.items():
        e = onnx.StringStringEntryProto()
        e.key, e.value = k, str(v)
        model.metadata_props.append(e)

    onnx.save_model(model, path, save_as_external_data=False)
    data = path + ".data"
    if os.path.isfile(data):
        os.remove(data)

    chk = onnx.load(path)
    print(f"\n[OK] 元数据 {len(chk.metadata_props)} 条, 文件 {os.path.getsize(path):,} 字节")
    print("\n关节顺序（BFS）:")
    for i, n in enumerate(TRAIN_JOINTS):
        print(f"  {i:2d}. {n:26s} kp={kps[i]:5.1f} scale={scales[i]:.5f} dpos={dpos[i]:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
