import numpy as np


def create_projection_matrix(
    width: int,
    height: int,
    fov: float,
) -> np.ndarray:
    """
    Reference:
    """
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    proj_mat = np.identity(3)  # 3x3 one

    proj_mat[0, 0] = focal
    proj_mat[1, 1] = focal
    proj_mat[0, 2] = width / 2.0
    proj_mat[1, 2] = height / 2.0

    return proj_mat


def project_3d_point_on_scene_with_depth(
    points: np.ndarray,
    proj_mat: np.ndarray,
    camera_mat: np.ndarray,
) -> np.ndarray:
    points_camera = _convert_from_world_to_camera(points, camera_mat)
    points_cv2 = _convert_from_ue4_to_cv2(points_camera)
    points_scene = _convert_from_camera_to_scene(points_cv2, proj_mat)

    return points_scene


def _convert_from_ue4_to_cv2(points: np.ndarray) -> np.ndarray:
    # UE4 -> OpenCV corrdinate: (x, y, z, 1) -> (y, -z, x)
    points_cv2 = np.concatenate(
        (
            points[:, 1].reshape(-1, 1),  # y
            -points[:, 2].reshape(-1, 1),  # -z
            points[:, 0].reshape(-1, 1),  # x
        ),
        axis=1,
    )

    return points_cv2


def _convert_from_world_to_camera(
    points: np.ndarray,
    camera_mat: np.ndarray,
) -> np.ndarray:
    point_4d_to_camera = points @ camera_mat.T  # (*, 4) x (4, 4)
    return point_4d_to_camera[:, :3]  # remove 1s from homogenous coordinate


def _convert_from_camera_to_scene(
    points: np.ndarray,
    proj_mat: np.ndarray,
) -> np.ndarray:
    point_scene = points @ proj_mat.T  # (*, 3) x (3, 3)
    # Normalization
    point_scene[:, 0] /= point_scene[:, 2]
    point_scene[:, 1] /= point_scene[:, 2]

    return point_scene
