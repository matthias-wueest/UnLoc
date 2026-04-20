# Parts of this file are adapted from F3Loc (Chen et al., CVPR 2024; MIT License).
# https://github.com/felix-ch/f3loc

"""
Coordinate transformation helpers for converting between world coordinates
and floorplan (map) pixel coordinates via an affine transform.

The affine transform is fitted from known correspondences between trajectory
positions (in meters) and floorplan pixel positions.
"""

import numpy as np
import cv2
import torch

# ---------------------------------------------------------------------------
# Affine transform fitting
# ---------------------------------------------------------------------------

def find_affine_transform(pts_src: np.ndarray, pts_dst: np.ndarray) -> np.ndarray:
    """
    Fit a 2-D affine transformation (least-squares) mapping source points
    to destination points.

    Parameters
    ----------
    pts_src : np.ndarray, shape (N, 2)
        Source points (e.g. world / trajectory coordinates).
    pts_dst : np.ndarray, shape (N, 2)
        Destination points (e.g. floorplan pixel coordinates).

    Returns
    -------
    affine_matrix : np.ndarray, shape (2, 3)
        The affine matrix  [a b tx; d e ty]  such that
        dst ≈ affine_matrix @ [src_x, src_y, 1]^T.
    """
    A = []
    B = []
    for (x, y), (x_prime, y_prime) in zip(pts_src, pts_dst):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.append(x_prime)
        B.append(y_prime)

    A = np.array(A)
    B = np.array(B)

    affine_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    affine_matrix = affine_params.reshape(2, 3)
    return affine_matrix


# ---------------------------------------------------------------------------
# Forward / inverse affine application
# ---------------------------------------------------------------------------

def apply_affine_transformation(point: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    """Apply a 2×3 affine matrix to a 2-D point (homogeneous multiplication)."""
    point_augmented = np.append(point, 1)
    return np.dot(affine_matrix, point_augmented)


def apply_inverse_affine_transformation(point: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    """Apply the inverse of a 2×3 affine matrix to a 2-D point."""
    affine_3x3 = np.vstack([affine_matrix, [0, 0, 1]])
    affine_inv = np.linalg.inv(affine_3x3)[:2, :]
    point_augmented = np.append(point, 1)
    return np.dot(affine_inv, point_augmented)


# ---------------------------------------------------------------------------
# Rotation helpers (extract rotation from the affine's linear part)
# ---------------------------------------------------------------------------

def apply_rotation(angle: float, rotation_matrix: np.ndarray) -> float:
    """Rotate an angle by the 2×2 linear part of the affine matrix."""
    vec = np.array([np.cos(angle), np.sin(angle)])
    transformed_vec = np.dot(rotation_matrix, vec)
    return np.arctan2(transformed_vec[1], transformed_vec[0])


def apply_inverse_rotation(angle: float, rotation_matrix: np.ndarray) -> float:
    """Inverse-rotate an angle using the transpose of the 2×2 rotation."""
    rotation_matrix_inv = rotation_matrix.T
    vec = np.array([np.cos(angle), np.sin(angle)])
    transformed_vec = np.dot(rotation_matrix_inv, vec)
    return np.arctan2(transformed_vec[1], transformed_vec[0])


# ---------------------------------------------------------------------------
# World ↔ Map coordinate conversions
# ---------------------------------------------------------------------------

def world_to_map_hge_complete(position_world, orientation_world_rad,
                              affine_matrix):
    """Transform world coordinates to (uncropped) floorplan map coordinates."""
    position_map = apply_affine_transformation(position_world, affine_matrix)

    orientation_map_rad0 = np.pi / 2
    orientation_map_rad = apply_rotation(orientation_world_rad, affine_matrix[:, :2])
    orientation_map_rad = orientation_map_rad - orientation_map_rad0

    return position_map, orientation_map_rad

def map_to_world_hge_complete(position_map, orientation_map_rad,
                              affine_matrix):
    """Inverse of `world_to_map_hge_complete` (uncropped variant)."""
    position_world = apply_inverse_affine_transformation(position_map, affine_matrix)

    orientation_map_rad_adj = orientation_map_rad + np.pi / 2
    orientation_world_rad = apply_inverse_rotation(orientation_map_rad_adj, affine_matrix[:, :2])

    return position_world, orientation_world_rad



# ---------------------------------------------------------------------------
# Relative pose computation
# ---------------------------------------------------------------------------

def get_rel_pose(ref_pose, src_pose):
    """
    Compute the relative pose of ``src_pose`` w.r.t. ``ref_pose``.

    Supports both single poses (dim=1) and batched poses (dim=2/3).

    Parameters
    ----------
    ref_pose : torch.Tensor
        (3,) or (N, 3) — reference pose [x, y, theta].
    src_pose : torch.Tensor
        (3,) or (N, L, 3) — source pose(s).

    Returns
    -------
    rel_pose : torch.Tensor
        Same shape as ``src_pose``. The relative translation is expressed
        in the reference frame, and the relative angle is wrapped to (-π, π].
    """
    if ref_pose.dim() == 1 and src_pose.dim() == 1:
        rel_pose = src_pose - ref_pose
        cr = torch.cos(ref_pose[-1])
        sr = torch.sin(ref_pose[-1])
        rel_x = cr * rel_pose[0] + sr * rel_pose[1]
        rel_y = -sr * rel_pose[0] + cr * rel_pose[1]
        rel_pose[0] = rel_x
        rel_pose[1] = rel_y
        rel_pose[-1] = (rel_pose[-1] + torch.pi) % (torch.pi * 2) - torch.pi
    else:
        rel_pose = src_pose - ref_pose.unsqueeze(1)
        cr = torch.cos(ref_pose[:, -1]).unsqueeze(-1)
        sr = torch.sin(ref_pose[:, -1]).unsqueeze(-1)
        rel_x = cr * rel_pose[:, :, 0] + sr * rel_pose[:, :, 1]
        rel_y = -sr * rel_pose[:, :, 0] + cr * rel_pose[:, :, 1]
        rel_pose[:, :, 0] = rel_x
        rel_pose[:, :, 1] = rel_y
        rel_pose[:, :, -1] = (
            (rel_pose[:, :, -1] + torch.pi) % (torch.pi * 2) - torch.pi
        )

    return rel_pose


# ---------------------------------------------------------------------------
# Image alignment utility for gravity correction using camera roll/pitch.
# ---------------------------------------------------------------------------
def gravity_align(
    img: np.ndarray,
    r: float,
    p: float,
    K: np.ndarray = np.array([[240, 0, 320], [0, 240, 240], [0, 0, 1]]).astype(np.float32),
    mode: int = 0,
) -> np.ndarray:
    """
    Warp an image to compensate for camera roll and pitch (gravity alignment).

    Parameters
    ----------
    img : np.ndarray
        Input image (H, W) or (H, W, C).
    r : float
        Roll angle [rad].
    p : float
        Pitch angle [rad].
    K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.
    mode : int
        Interpolation mode: 0 = bilinear, 1 = nearest.

    Returns
    -------
    aligned_img : np.ndarray
        Gravity-aligned image, same shape as input.
    """
    # From camera to gravity: roll → pitch → yaw
    # (pitch axis of robot and camera is opposite)
    p = -p
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)

    R_x = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])   # pitch
    R_z = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])    # roll

    R_cg = R_z @ R_x
    R_gc = R_cg.T

    h, w = img.shape[:2]
    persp_M = K @ R_gc @ np.linalg.inv(K)

    interp = cv2.INTER_NEAREST if mode == 1 else cv2.INTER_LINEAR
    aligned_img = cv2.warpPerspective(img, persp_M, (w, h), flags=interp)
    return aligned_img


def ray_cast(occ, pos, ang, dist_max=500):
    """
    Cast ray in the occupancy map
    Input:
        pos: in image coordinate, in pixel, [h, w]
        ang: ray shooting angle, in radian
    Output:
        dist: in pixels
    """
    h = occ.shape[0]
    w = occ.shape[1]
    occ = 255 - occ
    # determine the first corner
    c = np.cos(ang)
    s = np.sin(ang)

    if c == 1:
        # go right
        hit = False
        current_pos = pos.copy()
        while not hit:
            current_pos[1] += 1
            if current_pos[1] >= w:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist
    elif s == 1:
        # go up
        hit = False
        current_pos = pos.copy()
        while not hit:
            current_pos[0] += 1
            if current_pos[0] >= h:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist
    elif c == -1:
        # go left
        hit = False
        current_pos = pos.copy()
        while not hit:
            current_pos[1] -= 1
            if current_pos[1] < 0:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist
    elif s == -1:
        # go down
        hit = False
        current_pos = pos.copy()
        while not hit:
            current_pos[0] -= 1
            if current_pos[0] < 0:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist

    if c > 0 and s > 0:
        corner = np.array([np.floor(pos[0] + 1), np.floor(pos[1] + 1)])
        # go up and right
        hit = False
        current_pos = pos.copy()
        while not hit:
            dw = corner[1] - current_pos[1]
            dh = corner[0] - current_pos[0]
            corner_ang = dh / dw
            if np.tan(ang) > corner_ang:
                # increment upwards
                current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                corner[0] += 1
            elif np.tan(ang) < corner_ang:
                # increment right
                current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                corner[1] += 1
            else:
                # increment both upwards and right
                current_pos = corner.copy()
                corner[0] += 1
                corner[1] += 1
            if current_pos[0] >= h or current_pos[1] >= w:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True

        dist = np.linalg.norm(current_pos - pos, 2)
        return dist

    elif c < 0 and s > 0:
        corner = np.array([np.floor(pos[0] + 1), np.ceil(pos[1] - 1)])
        # go up and left
        hit = False
        current_pos = pos.copy()
        while not hit:
            dw = corner[1] - current_pos[1]
            dh = corner[0] - current_pos[0]
            corner_ang = dh / dw
            if np.tan(ang) < corner_ang:
                # increment upwards
                current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                corner[0] += 1
            elif np.tan(ang) > corner_ang:
                # increment left
                current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                corner[1] -= 1
            else:
                # increment both upwards and left
                current_pos = corner.copy()
                corner[0] += 1
                corner[1] -= 1
            if current_pos[0] >= h or current_pos[1] < 0:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True

        dist = np.linalg.norm(current_pos - pos, 2)
        return dist

    elif c < 0 and s < 0:
        corner = np.array([np.ceil(pos[0] - 1), np.ceil(pos[1] - 1)])
        # go down and left
        hit = False
        current_pos = pos.copy()
        while not hit:
            dw = corner[1] - current_pos[1]
            dh = corner[0] - current_pos[0]
            corner_ang = dh / dw
            if np.tan(ang) > corner_ang:
                # increment downwards
                current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                corner[0] -= 1
            elif np.tan(ang) < corner_ang:
                # increment left
                current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                corner[1] -= 1
            else:
                # increment both downwards and left
                current_pos = corner.copy()
                corner[0] -= 1
                corner[1] -= 1
            if current_pos[0] < 0 or current_pos[1] < 0:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True

        dist = np.linalg.norm(current_pos - pos, 2)
        return dist
    elif c > 0 and s < 0:
        corner = np.array([np.ceil(pos[0] - 1), np.floor(pos[1] + 1)])
        # go down and right
        hit = False
        current_pos = pos.copy()
        while not hit:
            dw = corner[1] - current_pos[1]
            dh = corner[0] - current_pos[0]
            corner_ang = dh / dw
            if np.tan(ang) < corner_ang:
                # increment downwards
                current_pos = np.array([corner[0], current_pos[1] + dh / np.tan(ang)])
                corner[0] -= 1
            elif np.tan(ang) > corner_ang:
                # increment right
                current_pos = np.array([current_pos[0] + dw * np.tan(ang), corner[1]])
                corner[1] += 1
            else:
                # increment both downwards and right
                current_pos = corner.copy()
                corner[0] -= 1
                corner[1] += 1
            if current_pos[0] < 0 or current_pos[1] >= w:
                return dist_max
            if occ[int(current_pos[0]), int(current_pos[1])] > 0:
                hit = True
        dist = np.linalg.norm(current_pos - pos, 2)
        return dist