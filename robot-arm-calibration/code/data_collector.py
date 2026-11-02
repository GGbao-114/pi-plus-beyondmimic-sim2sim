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

from handeye_calibration.utils import pose_to_matrix, decompose_transform
from handeye_calibration.aruco_detect import detect_aruco_marker, rvec_tvec_to_matrix


ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
MARKER_LENGTH = 0.05
TARGET_PAIRS = 12

COOLDOWN_SEC = 5.0
MIN_TRANSLATION = 0.02
MIN_ROTATION_DEG = 5.0


class DataCollectorNode(Node):

    def __init__(self):
        super().__init__("data_collector_node")
        self._bridge = CvBridge()
        self._latest_image = None
        self._current_fk_pose = None
        self._camera_matrix = None
        self._dist_coeffs = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._image_sub = self.create_subscription(
            Image, "/camera1/image_raw", self._image_callback, qos,
        )
        self._fk_sub = self.create_subscription(
            Pose, "/fk_pose", self._fk_callback, 10,
        )

    def _image_callback(self, msg):
        try:
            self._latest_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def _fk_callback(self, msg):
        self._current_fk_pose = msg

    def get_image(self):
        rclpy.spin_once(self, timeout_sec=0.05)
        return self._latest_image.copy() if self._latest_image is not None else None

    def get_fk(self):
        rclpy.spin_once(self, timeout_sec=0.05)
        return self._current_fk_pose

    def detect_aruco(self, image):
        K = self._camera_matrix if self._camera_matrix is not None else np.eye(3)
        D = self._dist_coeffs if self._dist_coeffs is not None else np.zeros((4, 1))
        rvec, tvec, _ = detect_aruco_marker(image, K, D)
        if rvec is None:
            return None
        return rvec_tvec_to_matrix(rvec, tvec)

    def load_intrinsics(self, path="camera_intrinsics.npz"):
        if not os.path.exists(path):
            self.get_logger().warn(f"{path} not found.")
            return False
        data = np.load(path)
        self._camera_matrix = data["K"]
        self._dist_coeffs = data["D"]
        self.get_logger().info(f"Loaded {path}")
        return True


def is_motion_sufficient(T_new, T_ref):
    if T_ref is None:
        return True
    R_new, t_new = decompose_transform(T_new)
    R_ref, t_ref = decompose_transform(T_ref)
    d = np.linalg.norm(t_new - t_ref)
    R_diff = R_new @ R_ref.T
    cos_a = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos_a)))
    return d >= MIN_TRANSLATION or angle >= MIN_ROTATION_DEG


def collect_data(node):
    T_be_list = []
    T_cm_list = []
    last_T_be = None
    last_record_time = 0

    print("=" * 60)
    print(f"  Smart Hand-Eye Data Collection  (target: {TARGET_PAIRS})")
    print(f"  Cooldown: {COOLDOWN_SEC}s  Min move: {MIN_TRANSLATION*100:.0f}mm / {MIN_ROTATION_DEG:.0f}deg")
    print("  Move arm via joint GUI. Auto-records when conditions met.")
    print("  Window shows live status.  Ctrl+C to stop.")
    print("=" * 60)

    cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
    _dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    _params = cv2.aruco.DetectorParameters()
    _params.minMarkerPerimeterRate = 0.20
    _params.maxMarkerPerimeterRate = 1.00

    while len(T_be_list) < TARGET_PAIRS and rclpy.ok():
        image = node.get_image()
        fk_pose = node.get_fk()

        if image is not None:
            display = image.copy()
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, _dict, parameters=_params)

            aruco_ok = ids is not None and len(ids) > 0 and 8 in ids.flatten()
            fk_ok = fk_pose is not None

            if aruco_ok:
                idx = list(ids.flatten()).index(8)
                cv2.aruco.drawDetectedMarkers(display, corners[idx:idx+1], np.array([[8]]))

            status = "ArUco: FOUND" if aruco_ok else "ArUco: ----"
            fk_str = f"FK:({fk_pose.position.x:.3f},{fk_pose.position.y:.3f},{fk_pose.position.z:.3f})" if fk_ok else "FK: ----"
            cd = max(0, COOLDOWN_SEC - (time.time() - last_record_time))
            cv2.putText(display, f"{status}  {fk_str}  [{len(T_be_list)}/{TARGET_PAIRS}]  cd={cd:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow("Data Collection", display)

        cv2.waitKey(80)

        if not (aruco_ok and fk_ok):
            continue

        now = time.time()
        if now - last_record_time < COOLDOWN_SEC:
            continue

        T_be = pose_to_matrix(fk_pose)
        if not is_motion_sufficient(T_be, last_T_be):
            continue

        T_cm = node.detect_aruco(image)
        if T_cm is None:
            continue

        T_be_list.append(T_be)
        T_cm_list.append(T_cm)
        last_T_be = T_be
        last_record_time = now

        print(f"  [{len(T_be_list)}/{TARGET_PAIRS}] Recorded  "
              f"FK:({T_be[0,3]:.3f},{T_be[1,3]:.3f},{T_be[2,3]:.3f})  "
              f"ArUco:({T_cm[0,3]:.3f},{T_cm[1,3]:.3f},{T_cm[2,3]:.3f})")

    cv2.destroyAllWindows()

    if len(T_be_list) >= 3:
        np.savez("calibration_data.npz", T_be=np.array(T_be_list), T_cm=np.array(T_cm_list))
        print(f"\nSaved {len(T_be_list)} pairs to calibration_data.npz")
    else:
        print(f"\nOnly {len(T_be_list)} pairs. Not saved.")


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    node.load_intrinsics()
    rclpy.spin_once(node, timeout_sec=0.5)
    collect_data(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
