#!/usr/bin/env python3
import threading
import time
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_LIMITS = {
    "joint1": (90, 360),    # deg
    "joint2": (0, 180),
    "joint3": (180, 360),
    "joint4": (90, 360),
    "joint5": (-90, 90),
    "joint6": (0, 360),
}

JOINT_ORDER = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class JointControlNode(Node):
    def __init__(self):
        super().__init__("joint_control_node")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.lock = threading.Lock()
        self.angles = [0.0] * 6
        self.timer = self.create_timer(0.1, self._publish)

    def set_angles(self, angles_deg):
        angles_rad = [a * 3.14159265 / 180.0 for a in angles_deg]
        with self.lock:
            self.angles = angles_rad

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_ORDER
        with self.lock:
            msg.position = list(self.angles)
        self.pub.publish(msg)


class JointGUI:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("Joint Control Panel")
        self.root.geometry("500x400")

        ttk.Label(self.root, text="Joint Angle Control (deg)",
                  font=("", 14, "bold")).pack(pady=10)

        self.sliders = {}
        self.labels = {}

        for i, name in enumerate(JOINT_ORDER):
            lo, hi = JOINT_LIMITS[name]
            frame = ttk.Frame(self.root)
            frame.pack(fill="x", padx=20, pady=3)

            ttk.Label(frame, text=f"{name}", width=8).pack(side="left")
            val_label = ttk.Label(frame, text="0.0", width=8)
            val_label.pack(side="right")
            self.labels[name] = val_label

            var = tk.DoubleVar(value=lo if lo > 0 else 0)
            slider = ttk.Scale(
                frame, from_=lo, to=hi, variable=var,
                orient="horizontal", length=280,
                command=lambda v, n=name: self._on_slide(n),
            )
            slider.pack(side="right", padx=5)
            self.sliders[name] = (slider, var)

        ttk.Button(self.root, text="SEND", command=self._send).pack(pady=10)
        self.status = ttk.Label(self.root, text="", foreground="blue")
        self.status.pack()

        self._poll()

    def _on_slide(self, name):
        val = self.sliders[name][1].get()
        self.labels[name].config(text=f"{val:.1f}")

    def _send(self):
        angles = []
        for name in JOINT_ORDER:
            val = self.sliders[name][1].get()
            angles.append(val)
        self.node.set_angles(angles)
        self.status.config(
            text=f"Sent: {', '.join(f'{a:.1f}' for a in angles)} deg",
            foreground="green",
        )

    def _poll(self):
        rclpy.spin_once(self.node, timeout_sec=0.01)
        self.root.after(30, self._poll)

    def run(self):
        self.root.mainloop()


def spin_node(node):
    rclpy.spin(node)


def main():
    rclpy.init()
    node = JointControlNode()

    gui = JointGUI(node)
    gui.run()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
