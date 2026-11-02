import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL


class ArucoTunerNode(Node):
    def __init__(self):
        super().__init__("aruco_tuner_node")
        self._bridge = CvBridge()
        self._latest = None
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._sub = self.create_subscription(
            Image, "/camera1/image_raw", self._image_callback, qos,
        )
        self._dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self._params = cv2.aruco.DetectorParameters()
        self._params.minMarkerPerimeterRate = 0.20
        self._params.maxMarkerPerimeterRate = 1.00

    def _image_callback(self, msg):
        try:
            self._latest = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def get_image(self):
        rclpy.spin_once(self, timeout_sec=0.05)
        return self._latest.copy() if self._latest is not None else None

    def set_params(self, adaptiveThreshWinSizeMin, adaptiveThreshWinSizeMax,
                   adaptiveThreshConstant, minMarkerPerimeterRate,
                   maxMarkerPerimeterRate, polygonalApproxAccuracyRate,
                   cornerRefinementMethod, cornerRefinementWinSize,
                   cornerRefinementMinAccuracy, minDistanceToBorder,
                   markerBorderBits, errorCorrectionRate):
        self._params.adaptiveThreshWinSizeMin = adaptiveThreshWinSizeMin
        self._params.adaptiveThreshWinSizeMax = adaptiveThreshWinSizeMax
        self._params.adaptiveThreshConstant = adaptiveThreshConstant / 10.0
        self._params.minMarkerPerimeterRate = minMarkerPerimeterRate / 100.0
        self._params.maxMarkerPerimeterRate = maxMarkerPerimeterRate / 100.0
        self._params.polygonalApproxAccuracyRate = polygonalApproxAccuracyRate / 100.0
        self._params.cornerRefinementMethod = cornerRefinementMethod
        self._params.cornerRefinementWinSize = cornerRefinementWinSize
        self._params.cornerRefinementMinAccuracy = cornerRefinementMinAccuracy / 100.0
        self._params.minDistanceToBorder = minDistanceToBorder
        self._params.markerBorderBits = markerBorderBits
        self._params.errorCorrectionRate = errorCorrectionRate / 10.0

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self._dict, parameters=self._params,
        )
        return corners, ids, rejected


def nothing(x):
    pass


def main():
    rclpy.init()
    node = ArucoTunerNode()
    time.sleep(1)

    cv2.namedWindow("ArUco Tuner", cv2.WINDOW_NORMAL)

    cv2.createTrackbar("ThreshMin", "ArUco Tuner", 3, 50, nothing)
    cv2.createTrackbar("ThreshMax", "ArUco Tuner", 23, 50, nothing)
    cv2.createTrackbar("ThreshConst", "ArUco Tuner", 7, 50, nothing)
    cv2.createTrackbar("MinSize%", "ArUco Tuner", 20, 100, nothing)
    cv2.createTrackbar("MaxSize%", "ArUco Tuner", 100, 100, nothing)
    cv2.createTrackbar("ApproxAcc%", "ArUco Tuner", 8, 50, nothing)
    cv2.createTrackbar("CornerRefine", "ArUco Tuner", 2, 3, nothing)
    cv2.createTrackbar("RefineWin", "ArUco Tuner", 5, 20, nothing)
    cv2.createTrackbar("AccRate%", "ArUco Tuner", 3, 50, nothing)
    cv2.createTrackbar("BorderDist", "ArUco Tuner", 0, 10, nothing)
    cv2.createTrackbar("BorderBits", "ArUco Tuner", 1, 5, nothing)
    cv2.createTrackbar("ErrCorrect", "ArUco Tuner", 0, 10, nothing)

    print("=" * 60)
    print("  ArUco Parameter Tuner (DICT_5X5_50)")
    print("  GREEN = detected.  RED = rejected.")
    print("  Key sliders to fix red->green:")
    print("    BorderBits: try 0,1,2   (border cell count)")
    print("    ErrCorrect: raise to allow bit errors")
    print("    ApproxAcc%: raise to tolerate distortion")
    print("    MinSize%: raise to filter small internal patterns")
    print("  Press Q to exit.")
    print("=" * 60)

    while rclpy.ok():
        img = node.get_image()
        if img is None:
            cv2.waitKey(100)
            continue

        ts = cv2.getTrackbarPos("ThreshMin", "ArUco Tuner")
        tl = cv2.getTrackbarPos("ThreshMax", "ArUco Tuner")
        tc = cv2.getTrackbarPos("ThreshConst", "ArUco Tuner")
        ms = cv2.getTrackbarPos("MinSize%", "ArUco Tuner")
        mx = cv2.getTrackbarPos("MaxSize%", "ArUco Tuner")
        pa = cv2.getTrackbarPos("ApproxAcc%", "ArUco Tuner")
        cr = cv2.getTrackbarPos("CornerRefine", "ArUco Tuner")
        rw = cv2.getTrackbarPos("RefineWin", "ArUco Tuner")
        ar = cv2.getTrackbarPos("AccRate%", "ArUco Tuner")
        bd = cv2.getTrackbarPos("BorderDist", "ArUco Tuner")
        bb = cv2.getTrackbarPos("BorderBits", "ArUco Tuner")
        ec = cv2.getTrackbarPos("ErrCorrect", "ArUco Tuner")

        if ts < 3:
            ts = 3
        if tl < ts:
            tl = ts + 1
        if ms < 1:
            ms = 1
        if mx < ms + 1:
            mx = ms + 1
        if rw < 1:
            rw = 1

        node.set_params(ts, tl, tc, ms, mx, pa, cr, rw, ar, bd, bb, ec)
        corners, ids, rejected = node.detect(img)

        display = img.copy()
        cv2.putText(display, f"Detected: {len(ids) if ids is not None else 0}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if ids is not None else (0, 0, 255), 2)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            for c, i in zip(corners, ids):
                cx, cy = int(c[0][:, 0].mean()), int(c[0][:, 1].mean())
                cv2.putText(display, f"ID:{i[0]}", (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if rejected:
            for r in rejected:
                pts = r.reshape(-1, 2).astype(np.int32)
                cv2.polylines(display, [pts], True, (0, 0, 255), 1)

        cv2.imshow("ArUco Tuner", display)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
