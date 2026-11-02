import time
import traceback

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
MARKER_LENGTH = 0.05


def detect_aruco_marker(image, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.minMarkerPerimeterRate = 0.20
    aruco_params.maxMarkerPerimeterRate = 1.00
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params,
    )
    if ids is None or 8 not in ids.flatten():
        return None, None, None

    idx = list(ids.flatten()).index(8)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners[idx:idx+1], MARKER_LENGTH, camera_matrix, dist_coeffs,
    )
    return rvecs[0], tvecs[0], corners[idx]


def rvec_tvec_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


class ArucoDetectorNode(Node):

    def __init__(self):
        super().__init__("aruco_detector_node")
        self._bridge = CvBridge()
        self._latest_image = None
        self._camera_matrix = None
        self._dist_coeffs = None
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._sub = self.create_subscription(
            Image, "/camera1/image_raw", self._image_callback, qos,
        )
        self.get_logger().info("ArUco detector ready.")

    def _image_callback(self, msg):
        try:
            self._latest_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def wait_for_image(self, timeout=3.0):
        self._latest_image = None
        start = time.time()
        while self._latest_image is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout:
                return None
        return self._latest_image

    def detect_once(self):
        image = self.wait_for_image()
        if image is None:
            self.get_logger().error("No image.")
            return None

        try:
            image = np.ascontiguousarray(image)
            K = self._camera_matrix if self._camera_matrix is not None else np.eye(3)
            D = self._dist_coeffs if self._dist_coeffs is not None else np.zeros((4, 1))

            rvec, tvec, corners = detect_aruco_marker(image, K, D)
            if rvec is None:
                return None

            T_cm = rvec_tvec_to_matrix(rvec, tvec)
            t = tvec.ravel()
            self.get_logger().info(f"Detected. pos=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})")
            return T_cm

        except Exception:
            self.get_logger().error(f"Error: {traceback.format_exc()}")
            return None

    def load_intrinsics(self, path="camera_intrinsics.npz"):
        import os
        if not os.path.exists(path):
            self.get_logger().warn(f"{path} not found.")
            return False
        data = np.load(path)
        self._camera_matrix = data["K"]
        self._dist_coeffs = data["D"]
        return True


def detect_aruco_continuously():
    print("=" * 50)
    print("  ArUco Continuous Detection")
    print("  Window shows live view. Q to quit.")
    print("=" * 50)

    rclpy.init()
    node = ArucoDetectorNode()
    node.load_intrinsics()
    time.sleep(1)

    cv2.namedWindow("ArUco Detection", cv2.WINDOW_NORMAL)
    _aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    _aruco_params = cv2.aruco.DetectorParameters()
    _aruco_params.minMarkerPerimeterRate = 0.20
    _aruco_params.maxMarkerPerimeterRate = 1.00

    last_print = 0
    while rclpy.ok():
        result = node.detect_once()

        display = node._latest_image
        if display is not None:
            display = display.copy()
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, _aruco_dict, parameters=_aruco_params,
            )
            if ids is not None and len(ids) > 0 and 8 in ids.flatten():
                idx = list(ids.flatten()).index(8)
                cv2.aruco.drawDetectedMarkers(display, corners[idx:idx+1], np.array([[8]]))
                cv2.putText(display, "FOUND",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Scanning... Q to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("ArUco Detection", display)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

        if result is not None and time.time() - last_print > 0.5:
            last_print = time.time()
            print(f"Detected  t=({result[0,3]:.3f},{result[1,3]:.3f},{result[2,3]:.3f})")

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


def main(args=None):
    detect_aruco_continuously()


if __name__ == "__main__":
    main()
