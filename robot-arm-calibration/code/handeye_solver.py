import os

import cv2
import numpy as np

from handeye_calibration.utils import decompose_transform


def solve_handeye(T_be_list, T_cm_list):
    if len(T_be_list) < 3:
        raise ValueError(f"Need at least 3 pairs, got {len(T_be_list)}.")

    R_A, t_A, R_B, t_B = [], [], [], []

    for i in range(1, len(T_be_list)):
        A_i = np.linalg.inv(T_be_list[i - 1]) @ T_be_list[i]
        B_i = T_cm_list[i] @ np.linalg.inv(T_cm_list[i - 1])

        R_A_i, t_A_i = decompose_transform(A_i)
        R_B_i, t_B_i = decompose_transform(B_i)

        R_A.append(R_A_i)
        t_A.append(t_A_i)
        R_B.append(R_B_i)
        t_B.append(t_B_i)

    R_X, t_X = cv2.calibrateHandEye(
        R_A, t_A, R_B, t_B,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    T_ec = np.eye(4)
    T_ec[:3, :3] = R_X
    T_ec[:3, 3] = t_X.ravel()
    return T_ec


def solve_and_save(calibration_data_path="calibration_data.npz"):
    if not os.path.exists(calibration_data_path):
        raise FileNotFoundError(f"{calibration_data_path} not found.")

    data = np.load(calibration_data_path, allow_pickle=True)
    T_be_list = list(data["T_be"])
    T_cm_list = list(data["T_cm"])
    print(f"Loaded {len(T_be_list)} pairs from {calibration_data_path}")

    T_ec = solve_handeye(T_be_list, T_cm_list)

    save_path = os.path.join(os.getcwd(), "handeye_transform.npy")
    np.save(save_path, T_ec)
    print(f"Saved hand-eye transform to {save_path}")
    print("T_ec (camera -> end-effector):")
    print(np.array2string(T_ec, precision=4, suppress_small=True))
    return T_ec


def main():
    solve_and_save()


if __name__ == "__main__":
    main()
