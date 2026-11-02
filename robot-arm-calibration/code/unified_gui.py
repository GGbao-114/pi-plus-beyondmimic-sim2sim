#!/usr/bin/env python3
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from PIL import Image as PILImage, ImageTk
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState

from handeye_calibration.utils import pose_to_matrix, decompose_transform
from handeye_calibration.aruco_detect import detect_aruco_marker, rvec_tvec_to_matrix


ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
MARKER_LENGTH = 0.05
TARGET = 12
COOLDOWN = 5.0
MIN_TRANS = 0.02
MIN_ROT = 5.0

JOINT_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
JOINT_LIMITS = {
    "joint1": (90, 360), "joint2": (0, 180), "joint3": (180, 360),
    "joint4": (90, 360), "joint5": (-90, 90), "joint6": (0, 360),
}


class UnifiedNode(Node):
    def __init__(self):
        super().__init__("unified_node")
        self._bridge = CvBridge()
        self._latest_img = None
        self._current_fk = None
        self._camera_matrix = None
        self._dist_coeffs = None

        qos_img = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, "/camera1/image_raw", self._img_cb, qos_img)
        self.create_subscription(Pose, "/fk_pose", self._fk_cb, 10)
        self._joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._img_lock = threading.Lock()
        self._fk_lock = threading.Lock()

    def _img_cb(self, msg):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._img_lock:
                self._latest_img = img
        except Exception:
            pass

    def _fk_cb(self, msg):
        with self._fk_lock:
            self._current_fk = msg

    def get_image(self):
        with self._img_lock:
            return self._latest_img.copy() if self._latest_img is not None else None

    def get_fk(self):
        with self._fk_lock:
            return self._current_fk

    def publish_joints(self, angles_deg):
        angles_rad = [a * 3.14159265 / 180.0 for a in angles_deg]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_ORDER
        msg.position = angles_rad
        self._joint_pub.publish(msg)

    def detect_aruco(self, image):
        K = self._camera_matrix if self._camera_matrix is not None else np.eye(3)
        D = self._dist_coeffs if self._dist_coeffs is not None else np.zeros((4, 1))
        rvec, tvec, _ = detect_aruco_marker(image, K, D)
        return rvec_tvec_to_matrix(rvec, tvec) if rvec is not None else None

    def load_intrinsics(self, path="camera_intrinsics.npz"):
        if not os.path.exists(path):
            return False
        d = np.load(path)
        self._camera_matrix = d["K"]
        self._dist_coeffs = d["D"]
        return True


def is_sufficient(T_new, T_ref):
    if T_ref is None:
        return True
    Rn, tn = decompose_transform(T_new)
    Rr, tr = decompose_transform(T_ref)
    d = np.linalg.norm(tn - tr)
    Rd = Rn @ Rr.T
    a = float(np.degrees(np.arccos(np.clip((np.trace(Rd)-1)/2, -1, 1))))
    return d >= MIN_TRANS or a >= MIN_ROT


class App:
    def __init__(self, node):
        self.node = node
        self.T_be_list = []
        self.T_cm_list = []
        self.last_T_be = None
        self.last_time = 0
        self.recording = False

        self.root = tk.Tk()
        self.root.title("Lab 10 — Joint Control + ArUco Collector")
        self.root.geometry("1000x700")

        left = ttk.Frame(self.root, width=400)
        left.pack(side="left", fill="y", padx=5, pady=5)

        ttk.Label(left, text="Joint Control (deg)", font=("", 12, "bold")).pack(pady=5)

        self.sliders = {}
        self.slider_labels = {}
        for name in JOINT_ORDER:
            lo, hi = JOINT_LIMITS[name]
            f = ttk.Frame(left)
            f.pack(fill="x", pady=2)
            ttk.Label(f, text=name, width=7).pack(side="left")
            v = tk.DoubleVar(value=(lo if lo > 0 else 0))
            lb = ttk.Label(f, text="0.0", width=6)
            lb.pack(side="right")
            self.slider_labels[name] = lb
            s = ttk.Scale(f, from_=lo, to=hi, variable=v, orient="horizontal", length=200,
                          command=lambda val, n=name: self._on_slide(n))
            s.pack(side="right", padx=3)
            self.sliders[name] = (s, v)

        ttk.Button(left, text="SEND JOINTS", command=self._send).pack(pady=5)

        self.status_var = tk.StringVar(value="Not recording")
        ttk.Label(left, textvariable=self.status_var, foreground="blue").pack(pady=2)

        right = ttk.Frame(self.root)
        right.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        ttk.Label(right, text="Camera + ArUco Detection", font=("", 12, "bold")).pack(pady=5)
        self.cam_label = ttk.Label(right)
        self.cam_label.pack()

        self.info_var = tk.StringVar()
        ttk.Label(right, textvariable=self.info_var, font=("", 10)).pack(pady=3)

        self.rec_btn = ttk.Button(right, text="Start Recording", command=self._toggle_record)
        self.rec_btn.pack(pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update()

    def _on_slide(self, name):
        v = self.sliders[name][1].get()
        self.slider_labels[name].config(text=f"{v:.1f}")

    def _send(self):
        ang = [self.sliders[n][1].get() for n in JOINT_ORDER]
        self.node.publish_joints(ang)
        self.status_var.set(f"Sent: {', '.join(f'{a:.0f}' for a in ang)}")

    def _toggle_record(self):
        if not self.recording:
            self.recording = True
            self.rec_btn.config(text="Recording... (click to stop)")
            self.status_var.set("Recording")
        else:
            self.recording = False
            self.rec_btn.config(text="Start Recording")
            self.status_var.set("Stopped")

    def _update(self):
        rclpy.spin_once(self.node, timeout_sec=0.01)

        img = self.node.get_image()
        fk = self.node.get_fk()

        if img is not None:
            display = img.copy()
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            _d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
            _p = cv2.aruco.DetectorParameters()
            _p.minMarkerPerimeterRate = 0.20
            _p.maxMarkerPerimeterRate = 1.00
            corners, ids, _ = cv2.aruco.detectMarkers(gray, _d, parameters=_p)
            aruco_ok = ids is not None and len(ids) > 0 and 8 in ids.flatten()
            if aruco_ok:
                idx = list(ids.flatten()).index(8)
                cv2.aruco.drawDetectedMarkers(display, corners[idx:idx+1], np.array([[8]]))

            cv2.putText(display, f"Recorded: {len(self.T_be_list)}/{TARGET}  {'FOUND' if aruco_ok else ''}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            h, w = display.shape[:2]
            scale = min(500 / h, 600 / w, 1.0)
            nw, nh = int(w * scale), int(h * scale)
            display = cv2.resize(display, (nw, nh))
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(display)
            imgtk = ImageTk.PhotoImage(pil)
            self.cam_label.config(image=imgtk)
            self.cam_label.image = imgtk

            fk_str = f"FK:({fk.position.x:.3f},{fk.position.y:.3f},{fk.position.z:.3f})" if fk else "FK: none"
            self.info_var.set(f"{fk_str}  ArUco: {'FOUND' if aruco_ok else 'none'}  [{len(self.T_be_list)}/{TARGET}]")

            if self.recording and aruco_ok and fk:
                now = time.time()
                if now - self.last_time >= COOLDOWN:
                    T_be = pose_to_matrix(fk)
                    if is_sufficient(T_be, self.last_T_be):
                        T_cm = self.node.detect_aruco(img)
                        if T_cm is not None:
                            self.T_be_list.append(T_be)
                            self.T_cm_list.append(T_cm)
                            self.last_T_be = T_be
                            self.last_time = now
                            msg = (f"[{len(self.T_be_list)}/{TARGET}] FK:({T_be[0,3]:.3f},{T_be[1,3]:.3f},{T_be[2,3]:.3f})  "
                                   f"ArUco:({T_cm[0,3]:.3f},{T_cm[1,3]:.3f},{T_cm[2,3]:.3f})")
                            print(msg)
                            self.status_var.set(msg[:60])

        if len(self.T_be_list) >= TARGET:
            self.recording = False
            np.savez("calibration_data.npz",
                     T_be=np.array(self.T_be_list), T_cm=np.array(self.T_cm_list))
            print(f"\nDone! {len(self.T_be_list)} pairs saved to calibration_data.npz")
            print("Closing in 2 seconds...")
            self.status_var.set(f"DONE! {len(self.T_be_list)} pairs saved. Closing...")
            self.rec_btn.config(state="disabled")
            self.root.after(2000, self.root.destroy)

        self.root.after(50, self._update)

    def _on_close(self):
        if len(self.T_be_list) >= 3:
            np.savez("calibration_data.npz",
                     T_be=np.array(self.T_be_list), T_cm=np.array(self.T_cm_list))
            print(f"Saved {len(self.T_be_list)} pairs to calibration_data.npz")
        self.node.destroy_node()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    node = UnifiedNode()
    node.load_intrinsics()
    time.sleep(1)
    app = App(node)
    app.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
