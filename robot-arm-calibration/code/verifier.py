import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from handeye_calibration.utils import (
    decompose_transform,
    pose_to_matrix,
    position_error_mm,
    rotation_error_deg,
)
from handeye_calibration.aruco_detect import detect_aruco_marker, rvec_tvec_to_matrix


ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
MARKER_LENGTH = 0.05
TEST_POSES = [
    ([0.14, 0.03, 0.26], [0.0, 0.0, 0.0, 1.0]),
    ([0.16, -0.03, 0.24], [0.0, 0.0, 0.0, 1.0]),
    ([0.12, 0.04, 0.27], [0.0, 0.0, 0.0, 1.0]),
    ([0.15, -0.02, 0.25], [0.03, 0.0, 0.0, 0.999]),
    ([0.13, 0.02, 0.25], [0.0, 0.03, 0.0, 0.999]),
]


class VerifierNode(Node):

    def __init__(self):
        super().__init__("verifier_node")
        self._bridge = CvBridge()
        self._latest_image = None
        self._current_fk_pose = None
        self._camera_matrix = None
        self._dist_coeffs = None
        self._T_ec = None
        self._T_bo_gt = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._image_sub = self.create_subscription(
            Image, "/camera1/image_raw", self._image_callback, qos,
        )
        self._fk_sub = self.create_subscription(
            Pose, "/fk_pose", self._fk_callback, 10,
        )
        self._pose_pub = self.create_publisher(Pose, "/gui_pose", 10)
        self.get_logger().info("Verifier ready.")

    def _image_callback(self, msg):
        try:
            self._latest_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def _fk_callback(self, msg):
        self._current_fk_pose = msg

    def wait_for_image(self, timeout=3.0):
        self._latest_image = None
        start = time.time()
        while self._latest_image is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout:
                return None
        return self._latest_image

    def wait_for_fk(self, timeout=3.0):
        self._current_fk_pose = None
        start = time.time()
        while self._current_fk_pose is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout:
                return None
        return self._current_fk_pose

    def publish_target_pose(self, position, quaternion_xyzw):
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(quaternion_xyzw[0])
        pose.orientation.y = float(quaternion_xyzw[1])
        pose.orientation.z = float(quaternion_xyzw[2])
        pose.orientation.w = float(quaternion_xyzw[3])
        self._pose_pub.publish(pose)

    def detect_aruco(self, image):
        try:
            K = self._camera_matrix if self._camera_matrix is not None else np.eye(3)
            D = self._dist_coeffs if self._dist_coeffs is not None else np.zeros((4, 1))
            rvec, tvec, _ = detect_aruco_marker(image, K, D)
            if rvec is None:
                return None
            return rvec_tvec_to_matrix(rvec, tvec)
        except Exception:
            return None

    def load_calibration(self, intrinsics_path="camera_intrinsics.npz",
                         handeye_path="handeye_transform.npy"):
        if os.path.exists(intrinsics_path):
            data = np.load(intrinsics_path)
            self._camera_matrix = data["K"]
            self._dist_coeffs = data["D"]
            self.get_logger().info(f"Loaded {intrinsics_path}")
        else:
            self.get_logger().error(f"{intrinsics_path} not found")
            return False
        if os.path.exists(handeye_path):
            self._T_ec = np.load(handeye_path)
            self.get_logger().info(f"Loaded {handeye_path}")
        else:
            self.get_logger().error(f"{handeye_path} not found")
            return False
        return True


def verify(node, test_poses=None):
    if test_poses is None:
        test_poses = TEST_POSES
    if node._T_ec is None:
        node.get_logger().error("Hand-eye transform not loaded.")
        return

    pos_errors = []
    ori_errors = []

    for i, (pos, quat) in enumerate(test_poses):
        node.get_logger().info(f"--- Test {i+1}/{len(test_poses)} ---")
        node.publish_target_pose(pos, quat)
        time.sleep(3)

        T_be_pose = node.wait_for_fk()
        if T_be_pose is None:
            continue
        T_be = pose_to_matrix(T_be_pose)

        image = node.wait_for_image()
        if image is None:
            continue

        T_co = node.detect_aruco(image)
        if T_co is None:
            continue

        T_bo_pred = T_be @ node._T_ec @ T_co
        if node._T_bo_gt is None:
            node._T_bo_gt = T_bo_pred
        T_bo_gt = node._T_bo_gt

        R_pred, t_pred = decompose_transform(T_bo_pred)
        R_gt, t_gt = decompose_transform(T_bo_gt)
        pe = position_error_mm(t_pred, t_gt)
        oe = rotation_error_deg(R_pred, R_gt)
        pos_errors.append(pe)
        ori_errors.append(oe)
        node.get_logger().info(f"  pos_err={pe:.2f}mm  ori_err={oe:.3f}deg")

    if not pos_errors:
        return
    mp, mo = np.mean(pos_errors), np.mean(ori_errors)
    print(f"\n  Pos: mean={mp:.2f}mm  {'PASS' if mp<5 else 'FAIL'}")
    print(f"  Ori: mean={mo:.3f}deg {'PASS' if mo<3 else 'FAIL'}")


def main(args=None):
    rclpy.init(args=args)
    node = VerifierNode()
    if not node.load_calibration():
        node.destroy_node()
        rclpy.shutdown()
        return
    verify(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
