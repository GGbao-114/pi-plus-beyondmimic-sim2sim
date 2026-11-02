import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


ALL_DICTS = {
    "DICT_4X4_50"  : cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100" : cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250" : cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50"  : cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100" : cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250" : cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50"  : cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100" : cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250" : cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50"  : cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100" : cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250" : cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class DictFinderNode(Node):
    def __init__(self):
        super().__init__("dict_finder")
        self._bridge = CvBridge()
        self._latest = None
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._sub = self.create_subscription(
            Image, "/camera1/image_raw", self._image_callback, qos,
        )

    def _image_callback(self, msg):
        try:
            self._latest = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def get_image(self):
        self._latest = None
        start = time.time()
        while self._latest is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > 3.0:
                return None
        return self._latest


def main():
    rclpy.init()
    node = DictFinderNode()
    time.sleep(1)

    print("=" * 60)
    print("  ArUco Dictionary Finder")
    print("  Trying all dictionaries to find a match...")
    print("=" * 60)

    image = node.get_image()
    if image is None:
        print("ERROR: No image received.")
        node.destroy_node()
        rclpy.shutdown()
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()

    found = []
    for name, dict_id in ALL_DICTS.items():
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
        if ids is not None and len(ids) > 0:
            found.append((name, len(ids), ids.ravel().tolist()))
            print(f"  MATCH: {name} -> ID(s): {ids.ravel()}")

    if not found:
        print("\n  No match in any dictionary.")
        print("  Possible causes:")
        print("    - Marker too blurry/far/small for internal bits decoding")
        print("    - Marker printed wrong (e.g. from online generator)")
        print("    - Check lighting and focus")
        print("  Try aruco_tuner to adjust detection params first.")
    else:
        print(f"\n  Found {len(found)} matching dictionary(s).")
        print("  Use the correct dict name in your code (DICT_4X4_50 etc).")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
