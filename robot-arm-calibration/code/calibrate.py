import os
import sys

import rclpy

from handeye_calibration.camera_calibration import calibrate_fisheye
from handeye_calibration.data_collector import DataCollectorNode, collect_data
from handeye_calibration.handeye_solver import solve_and_save
from handeye_calibration.verifier import VerifierNode, verify


def step0_calibrate_intrinsics():
    print("\n" + "=" * 50)
    print("STEP 0: Fisheye Camera Intrinsic Calibration")
    print("=" * 50)
    calibrate_fisheye()


def step1_collect_data():
    print("\n" + "=" * 50)
    print("STEP 2: Collect Hand-Eye Calibration Data")
    print("=" * 50)
    rclpy.init(args=sys.argv)
    node = DataCollectorNode()
    node.load_intrinsics()
    rclpy.spin_once(node, timeout_sec=0.5)
    collect_data(node)
    node.destroy_node()
    rclpy.shutdown()


def step2_solve():
    print("\n" + "=" * 50)
    print("STEP 3: Solve Hand-Eye Transform")
    print("=" * 50)
    if not os.path.exists("calibration_data.npz"):
        print("ERROR: calibration_data.npz not found. Run data collection first.")
        return
    solve_and_save()


def step3_verify():
    print("\n" + "=" * 50)
    print("STEP 4: Verify Calibration")
    print("=" * 50)
    if not os.path.exists("camera_intrinsics.npz") or not os.path.exists(
        "handeye_transform.npy"
    ):
        print("ERROR: calibration files not found.")
        return
    rclpy.init(args=sys.argv)
    node = VerifierNode()
    if not node.load_calibration():
        node.get_logger().error("Failed to load calibration.")
        node.destroy_node()
        rclpy.shutdown()
        return
    rclpy.spin_once(node, timeout_sec=0.5)
    verify(node)
    node.destroy_node()
    rclpy.shutdown()


def main(args=None):
    intrinsics_ok = os.path.exists("camera_intrinsics.npz")
    data_ok = os.path.exists("calibration_data.npz")
    handeye_ok = os.path.exists("handeye_transform.npy")

    print("\n" + "=" * 50)
    print("Lab 10: Hand-Eye Calibration Pipeline")
    print("=" * 50)
    print(f"  camera_intrinsics.npz : {'FOUND' if intrinsics_ok else 'MISSING'}")
    print(f"  calibration_data.npz  : {'FOUND' if data_ok else 'MISSING'}")
    print(f"  handeye_transform.npy : {'FOUND' if handeye_ok else 'MISSING'}")
    print("=" * 50)

    if not intrinsics_ok:
        print("\nCamera intrinsics not found.")
        print("Run Step 0 first in a separate terminal:")
        print("  ros2 run handeye_calibration camera_calibration")
        return
    else:
        print("\nCamera intrinsics found, skipping Step 0.")

    if not data_ok:
        print("\nStarting Step 2 (data collection)...")
        step1_collect_data()
    else:
        print("\nCalibration data found, skipping data collection.")

    if not handeye_ok:
        print("\nRunning hand-eye solver...")
        step2_solve()

    if handeye_ok or os.path.exists("handeye_transform.npy"):
        print("\nRunning verification...")
        step3_verify()

    print("\nLab 10 pipeline complete.")


if __name__ == "__main__":
    main()
