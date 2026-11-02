#!/usr/bin/env python3
import os, time
import numpy as np, rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener

TARGET_PAIRS = 12

# Ground truth: ArUco marker pose in world (from aruco_world.sdf)
T_wm = np.eye(4)
T_wm[:3, 3] = [0.4, 0.0, 0.25]

# Ground truth: camera in link5 (from URDF: link5_to_camera + camera_optical_joint)
T_ec_gt = np.eye(4)
T_ec_gt[:3, 3] = [0.0, 0.406, 0.39]
T_ec_gt[:3, :3] = Rotation.from_euler("xyz", [0, 0, 3.1415]).as_matrix()


class GTSimCollector(Node):
    def __init__(self):
        super().__init__("gt_sim_collector")
        self._tf_buf = Buffer()
        self._tf_ln = TransformListener(self._tf_buf, self)

    def get_T_be(self):
        try:
            tf = self._tf_buf.lookup_transform(
                "base_link", "link5", rclpy.time.Time(),
                rclpy.duration.Duration(seconds=2.0),
            )
            t = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
            q = tf.transform.rotation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            return T
        except Exception:
            return None

    def compute_T_cm(self, T_be):
        T_w_cam = T_be @ T_ec_gt
        return np.linalg.inv(T_w_cam) @ T_wm


def main():
    rclpy.init()
    node = GTSimCollector()
    time.sleep(2)

    T_be_list, T_cm_list = [], []
    print("=" * 60)
    print(f"  Ground-Truth Sim Collector (target: {TARGET_PAIRS})")
    print("  No camera needed. T_cm computed from known poses.")
    print("  Move arm via joint sliders, type c+Enter to record.")
    print("=" * 60)

    import select, sys
    def read_stdin():
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        if r:
            try:
                return sys.stdin.readline().strip().lower()
            except:
                return "q"
        return ""

    while len(T_be_list) < TARGET_PAIRS and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        cmd = read_stdin()
        if cmd == "q":
            break
        if cmd != "c":
            continue

        T_be = node.get_T_be()
        if T_be is None:
            print("  [SKIP] No FK")
            continue

        T_cm = node.compute_T_cm(T_be)
        T_be_list.append(T_be)
        T_cm_list.append(T_cm)

        print(f"  [{len(T_be_list)}/{TARGET_PAIRS}] "
              f"FK:({T_be[0,3]:.3f},{T_be[1,3]:.3f},{T_be[2,3]:.3f})  "
              f"ArUco:({T_cm[0,3]:.3f},{T_cm[1,3]:.3f},{T_cm[2,3]:.3f})")

    if len(T_be_list) >= 3:
        np.savez("calibration_data.npz", T_be=np.array(T_be_list), T_cm=np.array(T_cm_list))
        print(f"\nSaved {len(T_be_list)} pairs")
    else:
        print(f"\nOnly {len(T_be_list)} pairs. Not saved.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
