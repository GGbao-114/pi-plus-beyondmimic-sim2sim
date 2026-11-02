import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_matrix(pose):
    position = np.array([
        pose.position.x,
        pose.position.y,
        pose.position.z,
    ])
    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w
    rotation = Rotation.from_quat([qx, qy, qz, qw])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = position
    return matrix


def matrix_to_pose(matrix):
    from geometry_msgs.msg import Pose
    pose = Pose()
    pose.position.x = float(matrix[0, 3])
    pose.position.y = float(matrix[1, 3])
    pose.position.z = float(matrix[2, 3])
    rotation = Rotation.from_matrix(matrix[:3, :3])
    quat = rotation.as_quat()
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose


def build_transform(rotation, translation):
    matrix = np.eye(4)
    if isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
        matrix[:3, :3] = rotation
    elif isinstance(rotation, np.ndarray) and rotation.shape == (3,):
        matrix[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    elif isinstance(rotation, Rotation):
        matrix[:3, :3] = rotation.as_matrix()
    elif isinstance(rotation, (list, tuple)) and len(rotation) == 4:
        matrix[:3, :3] = Rotation.from_quat(rotation).as_matrix()
    else:
        raise ValueError(f"Unrecognised rotation type: {type(rotation)}")
    matrix[:3, 3] = np.asarray(translation, dtype=float).ravel()
    return matrix


def decompose_transform(matrix):
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return rotation, translation


def rotation_error_deg(R_pred, R_gt):
    R_diff = R_pred @ R_gt.T
    trace_val = np.trace(R_diff)
    cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def position_error_mm(t_pred, t_gt):
    return float(np.linalg.norm(np.asarray(t_pred) - np.asarray(t_gt)) * 1000.0)


def rodrigues_to_matrix(rvec, tvec):
    return build_transform(
        Rotation.from_rotvec(rvec.ravel()).as_matrix(),
        tvec.ravel(),
    )


def quat_xyzw_to_wxyz(x, y, z, w):
    return np.array([w, x, y, z])
