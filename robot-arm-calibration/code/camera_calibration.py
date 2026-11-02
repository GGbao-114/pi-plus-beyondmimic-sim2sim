import os
import select
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image


CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.025


class CameraCalibrationNode(Node):

    def __init__(self):
        super().__init__("camera_calibration_node")
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = None
        self._display = None
        self._sub = self.create_subscription(
            Image, "/camera1/image_raw", self._image_callback, 10
        )

    def _image_callback(self, msg):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._latest = img
        except Exception:
            pass

    def get_image(self):
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    def set_display(self, img):
        with self._lock:
            self._display = img.copy() if img is not None else None

    def get_display(self):
        with self._lock:
            return self._display.copy() if self._display is not None else None


def spin_thread(node, executor):
    try:
        executor.spin()
    except Exception:
        pass


def display_thread(node, running):
    cv2.namedWindow("Calibration Capture", cv2.WINDOW_NORMAL)
    while running[0]:
        img = node.get_display()
        if img is not None:
            cv2.imshow("Calibration Capture", img)
        cv2.waitKey(50)
    cv2.destroyAllWindows()


def read_stdin(timeout=0.05):
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if r:
        try:
            return sys.stdin.readline().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
    return ""


def calibrate_fisheye():
    rclpy.init()
    node = CameraCalibrationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    ros_t = threading.Thread(target=spin_thread, args=(node, executor), daemon=True)
    ros_t.start()

    running = [True]
    disp_t = threading.Thread(target=display_thread, args=(node, running), daemon=True)
    disp_t.start()

    time.sleep(2)

    obj_points = []
    img_points = []
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 1, 3), np.float32)
    objp[:, 0, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    print("=" * 60)
    print("  Fisheye Camera Calibration  (9x6 chessboard, 25mm squares)")
    print("=" * 60)
    print("  c + Enter : capture    q + Enter : finish")
    print("  Aim for 15-20 captures. Window shows live preview.")
    print("=" * 60)

    image_shape = None
    sample_count = 0
    frame_count = 0

    while True:
        image = node.get_image()
        if image is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        if image_shape is None:
            image_shape = image.shape[1], image.shape[0]
            print(f"Camera: {image_shape[0]}x{image_shape[1]}, fps ~30")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
        )
        if not ret:
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if frame_count % 30 == 0 and not ret:
            print(f"  [debug] frame {frame_count}: no chessboard found. "
                  f"Try closer/farther, tilt, better light.")

        display = image.copy()
        if ret:
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, ret)
        cv2.putText(
            display, f"Captured: {sample_count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        node.set_display(display)

        cmd = read_stdin()
        if cmd is None:
            break
        if cmd == "q":
            print("Finishing capture...")
            break
        if cmd == "c":
            if not ret:
                print("  [SKIP] No chessboard detected.")
                continue
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            img_points.append(corners_refined)
            obj_points.append(objp)
            sample_count += 1
            print(f"  [{sample_count}] Captured (drift={np.mean(np.abs(corners_refined - corners)):.3f} px)")
        elif cmd:
            print(f"  Unknown '{cmd}'. Type c to capture, q to finish.")

    running[0] = False
    disp_t.join(timeout=2)

    executor.remove_node(node)
    node.destroy_node()
    rclpy.shutdown()

    if sample_count < 5:
        print(f"ERROR: Only {sample_count} images, need at least 5.")
        return

    print(f"\nRunning fisheye calibration on {sample_count} images...")
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )
    try:
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            obj_points, img_points, image_shape,
            K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        )
    except Exception as e:
        print(f"Calibration failed: {e}")
        return

    print(f"RMS: {rms:.4f} px  {'[PASS]' if rms < 0.8 else '[HIGH]'}")
    print(f"K:\n{np.array2string(K, precision=2, suppress_small=True)}")
    print(f"D: {np.array2string(D.ravel(), precision=4, suppress_small=True)}")

    np.savez(os.path.join(os.getcwd(), "camera_intrinsics.npz"),
             K=K, D=D, image_shape=image_shape, rms=rms)
    print("Saved to camera_intrinsics.npz")

    print("\nUndistortion preview — press Esc to close.")
    rclpy.init()
    node2 = CameraCalibrationNode()
    exec2 = MultiThreadedExecutor()
    exec2.add_node(node2)
    t2 = threading.Thread(target=spin_thread, args=(node2, exec2), daemon=True)
    t2.start()
    time.sleep(1)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, image_shape, cv2.CV_16SC2,
    )
    cv2.namedWindow("Original vs Undistorted", cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = node2.get_image()
            if frame is None:
                cv2.waitKey(100)
                continue
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            combined = np.hstack((frame, undistorted))
            cv2.imshow("Original vs Undistorted", combined)
            if cv2.waitKey(50) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    exec2.remove_node(node2)
    node2.destroy_node()
    rclpy.shutdown()


def main(args=None):
    calibrate_fisheye()


if __name__ == "__main__":
    main()
