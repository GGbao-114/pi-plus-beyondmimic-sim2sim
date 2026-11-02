import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo


ALL_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class SimDictFinder(Node):
    def __init__(self):
        super().__init__("sim_dict_finder")
        self._bridge = CvBridge()
        self._latest_img = None
        self._camera_matrix = None
        self._dist_coeffs = None
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, "/camera_sensor", self._img_cb, qos)
        self.create_subscription(CameraInfo, "/camera_info", self._info_cb, 10)

    def _img_cb(self, msg):
        try:
            self._latest_img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def _info_cb(self, msg):
        if self._camera_matrix is None:
            self._camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self._dist_coeffs = np.array(msg.d, dtype=np.float64)

    def get_image(self):
        self._latest_img = None
        start = time.time()
        while self._latest_img is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > 3.0:
                return None
        return self._latest_img


def main():
    rclpy.init()
    node = SimDictFinder()
    time.sleep(1)

    print("=" * 50)
    print("  Sim ArUco Dictionary Finder")
    print("  Subscribes to /camera_sensor, tries all dicts.")
    print("=" * 50)

    image = node.get_image()
    if image is None:
        print("ERROR: No image from /camera_sensor.")
        node.destroy_node()
        rclpy.shutdown()
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 1.00

    K = node._camera_matrix if node._camera_matrix is not None else np.eye(3)
    D = node._dist_coeffs if node._dist_coeffs is not None else np.zeros((4, 1))
    if node._camera_matrix is not None:
        print(f"Camera: {int(K[0,0])}x{int(K[1,1])} px, cx={int(K[0,2])}, cy={int(K[1,2])}")

    found = []
    for name, dict_id in ALL_DICTS.items():
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
        if ids is not None and len(ids) > 0:
            found.append((name, ids.ravel().tolist()))
            print(f"  MATCH: {name} -> ID(s): {ids.ravel()}")

    if not found:
        print("\n  No match. Adjust arm for better ArUco view and retry.")
    else:
        print(f"\n  Found {len(found)} match(es).")
        for n, i in found:
            print(f"    Use: {n}, marker ID(s): {i}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
