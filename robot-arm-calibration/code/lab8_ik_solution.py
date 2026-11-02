"""Lab 8 inverse-kinematics derivation and verification.

This script uses the Lab 7 classic-DH table in SI units.  It has no ROS or
robot dependency, so it can be run at home.

Examples:
    python lab8_ik_solution.py --pose 1
    python lab8_ik_solution.py --pose 1 --symbolic
    python lab8_ik_solution.py --symbolic

The numerical output distinguishes:
  - effective DH angles: theta_i used in the DH matrices;
  - motor angles: delta_theta_i published by ROS, where
    theta_i = theta_offset_i + delta_theta_i.
"""

import argparse
import math

import numpy as np
import sympy as sp


# Lab 7 dh_params.yaml, in the order [a, alpha, d, theta_offset].
# The YAML values 1.5708 and 3.14159 are intended to be pi/2 and pi.
DH_PARAMS = np.array([
    [0.00, math.pi / 2.0, 0.00, math.pi],
    [0.17, 0.00, 0.00, 0.00],
    [0.07, -math.pi / 2.0, 0.00, math.pi / 2.0],
    [0.00, math.pi / 2.0, 0.11, 0.00],
    [0.00, -math.pi / 2.0, 0.00, 0.00],
    [0.00, 0.00, 0.09, 0.00],
], dtype=float)

# Lab 8 PDF Table 1. Quaternion order is (x, y, z, w).
TARGET_POSES = {
    1: (np.array([0.2480, -0.0040, -0.0490]),
        np.array([0.8260, 0.5440, 0.1130, 0.0980])),
    2: (np.array([0.1460, -0.2565, 0.0050]),
        np.array([0.9080, 0.2010, 0.2590, 0.2610])),
    3: (np.array([-0.0380, 0.1670, 0.2370]),
        np.array([0.0790, 0.3650, 0.3680, 0.8520])),
}


def wrap_to_pi(angle):
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def dh_transform(theta, d, a, alpha):
    """Classic DH: RotZ(theta) * TransZ(d) * TransX(a) * RotX(alpha)."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk_effective(effective_angles, joint_count=6):
    """FK when effective_angles are the theta values inside the DH matrices."""
    T = np.eye(4)
    for i in range(joint_count):
        a, alpha, d, _ = DH_PARAMS[i]
        T = T @ dh_transform(effective_angles[i], d, a, alpha)
    return T


def fk_motor(motor_angles):
    """FK for ROS motor angles delta_theta, including theta offsets."""
    offsets = DH_PARAMS[:, 3]
    return fk_effective(offsets + np.asarray(motor_angles))


def quaternion_to_rotation(quaternion):
    """Convert an (x, y, z, w) quaternion into a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("Quaternion cannot be zero.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def rotation_error_degrees(expected, actual):
    """Smallest angular distance between two rotation matrices, in degrees."""
    relative = expected.T @ actual
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def solve_ik(position, quaternion):
    """Return up to eight IK solutions for this arm's spherical-wrist geometry.

    Each solution is a dictionary containing effective DH angles, ROS motor
    angles, and FK verification errors.  The derivation uses the wrist center
    and the two elbow / two shoulder-side / two wrist-flip branches.
    """
    position = np.asarray(position, dtype=float)
    R06 = quaternion_to_rotation(quaternion)

    a2 = DH_PARAMS[1, 0]
    a3 = DH_PARAMS[2, 0]
    d4 = DH_PARAMS[3, 2]
    d6 = DH_PARAMS[5, 2]
    offsets = DH_PARAMS[:, 3]

    # P_w = P_e - d6 * R_0_6 * z_hat.
    wrist_center = position - d6 * R06[:, 2]
    distance_sq = float(wrist_center @ wrist_center)
    cosine_term = (distance_sq - a2 * a2 - a3 * a3 - d4 * d4) / (2.0 * a2)
    link_radius = math.hypot(a3, d4)
    c3 = cosine_term / link_radius

    if c3 < -1.0 - 1e-9 or c3 > 1.0 + 1e-9:
        raise ValueError(
            "Target is outside the reachable workspace: "
            f"cos(theta3 + beta) = {c3:.6f}."
        )
    c3 = float(np.clip(c3, -1.0, 1.0))
    s3 = math.sqrt(max(0.0, 1.0 - c3 * c3))
    beta = math.atan2(d4, a3)
    theta3_candidates = [
        math.atan2(s3, c3) - beta,
        math.atan2(-s3, c3) - beta,
    ]

    base_angle = math.atan2(wrist_center[1], wrist_center[0])
    radial_distance = math.hypot(wrist_center[0], wrist_center[1])
    solutions = []

    for theta3 in theta3_candidates:
        # In the radial-Z plane:
        # r = c0*cos(theta2) - c1*sin(theta2)
        # z = c1*cos(theta2) + c0*sin(theta2)
        c0 = a2 + a3 * math.cos(theta3) - d4 * math.sin(theta3)
        c1 = a3 * math.sin(theta3) + d4 * math.cos(theta3)
        phase = math.atan2(c1, c0)

        # theta1 + pi changes the signed radial coordinate from r to -r.
        for theta1, signed_r in ((base_angle, radial_distance),
                                 (base_angle + math.pi, -radial_distance)):
            theta2 = math.atan2(wrist_center[2], signed_r) - phase
            first_three = np.array([theta1, theta2, theta3])
            R03 = fk_effective(first_three, joint_count=3)[:3, :3]
            R36 = R03.T @ R06

            # R36 = R34(theta4) * R45(theta5) * R56(theta6).
            s5_abs = math.hypot(R36[0, 2], R36[1, 2])
            if s5_abs < 1e-9:
                # Wrist singularity: theta4 and theta6 cannot be separated.
                # Choose theta4 = 0 as one valid representative solution.
                theta4 = 0.0
                theta5 = math.atan2(s5_abs, R36[2, 2])
                theta6 = math.atan2(R36[1, 0], R36[0, 0])
                wrist_candidates = [(theta4, theta5, theta6)]
            else:
                wrist_candidates = [
                    # With alpha4 = +pi/2 and alpha5 = -pi/2:
                    # R36(0,2) = -cos(theta4) sin(theta5),
                    # R36(1,2) = -sin(theta4) sin(theta5).
                    (math.atan2(-R36[1, 2], -R36[0, 2]),
                     math.atan2(s5_abs, R36[2, 2]),
                     math.atan2(-R36[2, 1], R36[2, 0])),
                    (math.atan2(R36[1, 2], R36[0, 2]),
                     math.atan2(-s5_abs, R36[2, 2]),
                     math.atan2(R36[2, 1], -R36[2, 0])),
                ]

            for wrist in wrist_candidates:
                effective = np.array([theta1, theta2, theta3, *wrist])
                motor = np.array([wrap_to_pi(angle) for angle in effective - offsets])
                T_check = fk_motor(motor)
                position_error_m = float(np.linalg.norm(T_check[:3, 3] - position))
                orientation_error_deg = rotation_error_degrees(R06, T_check[:3, :3])
                solutions.append({
                    "effective": effective,
                    "motor": motor,
                    "position_error_m": position_error_m,
                    "orientation_error_deg": orientation_error_deg,
                })

    return wrist_center, solutions


def symbolic_derivation():
    """Print the Task 1-5 symbolic expressions for the current DH structure."""
    t1, t2, t3, t4, t5, t6 = sp.symbols(
        "theta1 theta2 theta3 theta4 theta5 theta6", real=True
    )
    a2, a3, d4 = sp.symbols("a2 a3 d4", positive=True, real=True)

    def dh_sym(theta, d, a, alpha):
        ct, st = sp.cos(theta), sp.sin(theta)
        ca, sa = sp.cos(alpha), sp.sin(alpha)
        return sp.Matrix([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ])

    T01 = dh_sym(t1, 0, 0, sp.pi / 2)
    T12 = dh_sym(t2, 0, a2, 0)
    T23 = dh_sym(t3, 0, a3, -sp.pi / 2)
    T34 = dh_sym(t4, d4, 0, sp.pi / 2)
    T45 = dh_sym(t5, 0, 0, -sp.pi / 2)
    T56 = dh_sym(t6, 0, 0, 0)
    T05 = sp.trigsimp(T01 * T12 * T23 * T34 * T45)
    R36 = sp.trigsimp((T34 * T45 * T56)[:3, :3])

    pw = [sp.trigsimp(sp.expand_trig(T05[i, 3])) for i in range(3)]
    dist_sq = sp.trigsimp(sp.simplify(sum(component ** 2 for component in pw)))
    proj_sq = sp.trigsimp(sp.simplify(pw[0] ** 2 + pw[1] ** 2))

    print("\n=== SymPy derivation: Tasks 1-5 ===")
    print("Angles below are effective DH angles, before theta_offset is removed.")
    print("\nT_0^5 =")
    sp.pprint(T05)
    print("\nP_w = [Pwx, Pwy, Pwz]^T =")
    sp.pprint(sp.Matrix(pw))
    print("\n||P_w||^2 =")
    sp.pprint(dist_sq)
    print("\nTask 2, define:")
    print("C = (||Pw||^2 - a2^2 - a3^2 - d4^2) / (2*a2*sqrt(a3^2 + d4^2))")
    print("beta = atan2(d4, a3)")
    print("theta3 = atan2(+/-sqrt(1 - C^2), C) - beta")
    print("\nPwx^2 + Pwy^2 =")
    sp.pprint(proj_sq)
    print("\nTask 3, let r = +/-sqrt(Pwx^2 + Pwy^2):")
    print("c0 = a2 + a3*cos(theta3) - d4*sin(theta3)")
    print("c1 = a3*sin(theta3) + d4*cos(theta3)")
    print("theta2 = atan2(Pwz, r) - atan2(c1, c0)")
    print("\nTask 4: theta1 = atan2(Pwy, Pwx), with theta1 + pi as the second branch.")
    print("\nTask 5: R_3^6 = (R_0^3)^T R_0^6. For this wrist:")
    sp.pprint(R36)
    print("theta4 = atan2(-R36(1,2), -R36(0,2))")
    print("theta5 = atan2(+sqrt(R36(0,2)^2 + R36(1,2)^2), R36(2,2))")
    print("theta6 = atan2(-R36(2,1), R36(2,0))")
    print("The wrist-flip branch uses the opposite sign of theta5.")
    print("Task 6: two theta1 branches x two theta3 branches x two wrist flips = up to 8 solutions.")


def print_solutions(pose_id):
    position, quaternion = TARGET_POSES[pose_id]
    quaternion_norm = float(np.linalg.norm(quaternion))
    print(f"\n=== Pose {pose_id} numerical IK ===")
    print("Target position [m]:", np.array2string(position, precision=6))
    print("Target quaternion [x, y, z, w]:", np.array2string(quaternion, precision=6))
    print(f"Quaternion norm before normalization: {quaternion_norm:.9f}")

    wrist_center, solutions = solve_ik(position, quaternion)
    print("Wrist center Pw [m]:", np.array2string(wrist_center, precision=6))
    print(f"Reachable IK candidates: {len(solutions)}")

    for index, solution in enumerate(solutions, start=1):
        effective_deg = np.degrees(solution["effective"])
        motor_deg = np.degrees(solution["motor"])
        print(f"\nCandidate {index}")
        print("  effective DH angles [deg]:", np.array2string(effective_deg, precision=3))
        print("  ROS motor angles [deg]:   ", np.array2string(motor_deg, precision=3))
        print(f"  FK position error: {solution['position_error_m'] * 1000.0:.6f} mm")
        print(f"  FK orientation error: {solution['orientation_error_deg']:.9f} deg")


def print_report_notes():
    """Print concise text required by the C++ walkthrough and failure-condition tasks."""
    print("\n=== Report notes ===")
    print("C++ walkthrough, Step 0:")
    print("arm6 = [0, 0, d6]^T and j5 = v - q * arm6 implement "
          "Pw = Pe - d6 * R06 * z_hat. This removes the final d6 offset, "
          "so joints 1-3 can solve the wrist-center position independently.")
    print("\nC++ failure note for the report (26 words):")
    print("When theta5 is 0 or pi, R36(0,2) and R36(1,2) are both zero. "
          "The wrist atan2 expressions become singular, so theta4 and theta6 cannot be uniquely separated.")


def main():
    parser = argparse.ArgumentParser(description="Lab 8 SymPy derivation and numerical IK.")
    parser.add_argument("--pose", choices=("1", "2", "3", "all"), default="all",
                        help="Pose from Lab 8 Table 1, or all three poses (default: all).")
    parser.add_argument("--symbolic", action="store_true",
                        help="Also print the SymPy derivation required by Tasks 1-5.")
    args = parser.parse_args()

    if args.symbolic:
        symbolic_derivation()
    pose_ids = sorted(TARGET_POSES) if args.pose == "all" else [int(args.pose)]
    for pose_id in pose_ids:
        print_solutions(pose_id)
    print_report_notes()


if __name__ == "__main__":
    main()
