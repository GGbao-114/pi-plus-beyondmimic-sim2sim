import os
import sys

import cv2
import numpy as np


CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.025


def calibrate_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.1f} fps")

    obj_points = []
    img_points = []
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 1, 3), np.float32)
    objp[:, 0, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    image_shape = None
    sample_count = 0
    frame_idx = 0
    skip = max(1, int(fps / 5))

    prev_T = None
    min_translation = 0.02
    min_angle_deg = 5.0

    cv2.namedWindow("Auto Capture - press Q to stop early", cv2.WINDOW_NORMAL)

    while True:
        ret, image = cap.read()
        if not ret:
            break
        frame_idx += 1

        if image_shape is None:
            image_shape = image.shape[1], image.shape[0]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
        )

        display = image.copy()
        if ret_cb:
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, ret_cb)

        cv2.putText(display, f"Frame: {frame_idx}/{total_frames}  Captured: {sample_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Auto Capture - press Q to stop early", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("User stopped early.")
            break

        if frame_idx % skip != 0:
            continue
        if not ret_cb:
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        success, rvec, tvec = cv2.solvePnP(
            objp.reshape(-1, 1, 3), corners_refined,
            np.eye(3), None, flags=cv2.SOLVEPNP_IPPE,
        )
        if not success:
            continue

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()

        if prev_T is not None:
            trans = np.linalg.norm(t - prev_T[:3, 3])
            R_diff = R @ prev_T[:3, :3].T
            cos_a = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cos_a)))
            if trans < min_translation and angle < min_angle_deg:
                continue

        prev_T = np.eye(4)
        prev_T[:3, :3] = R
        prev_T[:3, 3] = t

        img_points.append(corners_refined)
        obj_points.append(objp)
        sample_count += 1
        print(f"  [{sample_count}] frame {frame_idx} captured")

        if sample_count >= 20:
            print("Reached 20 captures.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if sample_count < 5:
        print(f"ERROR: Only {sample_count} images, need at least 5.")
        return

    print(f"\nRunning fisheye calibration on {sample_count} images...")
    K, D = np.zeros((3, 3)), np.zeros((4, 1))
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
             + cv2.fisheye.CALIB_CHECK_COND
             + cv2.fisheye.CALIB_FIX_SKEW)
    try:
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            obj_points, img_points, image_shape, K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        )
    except Exception as e:
        print(f"Calibration failed: {e}")
        return

    print(f"RMS: {rms:.4f} px  {'[PASS]' if rms < 0.8 else '[HIGH]'}")
    print(f"K:\n{np.array2string(K, precision=2, suppress_small=True)}")
    print(f"D: {np.array2string(D.ravel(), precision=4, suppress_small=True)}")
    np.savez("camera_intrinsics.npz", K=K, D=D, image_shape=image_shape, rms=rms)
    print("Saved to camera_intrinsics.npz")

    print("\nShowing undistortion comparison. Press any key to close.")
    cap2 = cv2.VideoCapture(video_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, image_shape, cv2.CV_16SC2,
    )
    cv2.namedWindow("Left=Original  Right=Undistorted", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap2.read()
        if not ret:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        h = min(frame.shape[0], undist.shape[0])
        w = min(frame.shape[1], undist.shape[1])
        combined = np.hstack((frame[:h, :w], undist[:h, :w]))
        cv2.imshow("Left=Original  Right=Undistorted", combined)
        if cv2.waitKey(30) & 0xFF != 0xFF:
            break
    cap2.release()
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 calibrate_from_video.py <video_file>")
        sys.exit(1)
    calibrate_from_video(sys.argv[1])


if __name__ == "__main__":
    main()
