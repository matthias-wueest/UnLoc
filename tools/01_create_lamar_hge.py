"""
Prepare the customized LaMAR HGE dataset from the raw LaMAR dataset.

This script:
  1. Loads the HGE floorplan and computes an affine transform from world
     coordinates to floorplan pixel coordinates.
  2. For each iOS recording session, extracts indoor-only poses, converts
     them to the floorplan frame, computes ground-truth floorplan depths
     via raycasting, and writes the result as a self-contained dataset
     folder (poses, Euler angles, images, map, depth file).

Required inputs:

  --session-dir: Path to the LaMAR HGE session directory, which contains
      ``trajectories.txt`` and the ``raw_data/`` folder with iOS recording
      sessions. Obtain the LaMAR dataset from https://lamar.ethz.ch and point
      this to ``HGE/HGE/sessions/map``.

  --floorplans-dir: Directory containing the preprocessed floorplan image
      (``map_HGE.png``). The floorplan must be prepared from the
      official building plan as a binary occupancy grid: room numbers, stairs,
      and doors connecting two corridors that are typically open should be
      removed, while all other doors are represented as solid walls.

See Appendix A.1.2 of the UnLoc paper for details.
"""

import os
import shutil
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import griddata

from utils.geometry import (
    apply_affine_transformation,
    apply_rotation,
    find_affine_transform,
    ray_cast,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Paths
FLOORPLAN_FILENAME = "map_HGE.png"

# Camera parameters
FOV_DEG = 48.5
FOCAL_LENGTH_PX = 1596    # focal length (width) in pixels
IMAGE_WIDTH_PX = 1440
NUM_RAYS = 40
DEPTH_COLUMNS = 90        # number of columns in the output depth file

# Affine calibration correspondences (floorplan ↔ world)
# Four building entrance doorways were identified in both the floorplan image
# (pixel coordinates) and the LaMAR ground-truth trajectories (metre
# coordinates, recorded when the camera crosses each threshold).  A 2×3
# affine transform is fitted from these four point pairs to map world
# positions to floorplan pixels.  See Appendix A.1.2 for details.
FLOORPLAN_CORRESPONDENCES = np.array(
    [[1516, 490.5], [1532, 2162], [350, 1515], [391, 135]]
)
TRAJECTORY_CORRESPONDENCES = np.array(
    [[-12.600, 10.103], [77.981, 9.882], [42.641, -54.348], [-33.099, -51.229]]
)

# Timestamp intervals labelling indoor / outdoor segments
INDOOR_INTERVALS = [
    (10283474528, 10627820197),
    (3302315254, 3373150184),
    (3656323308, 3849919589),
    (3918622077, 3960020150),
    (78222688174, 78428312296),
    (1543118709, 1589664520),
    (2711923398, 2997528776),
    (3659211240, 3790719011),
    (3927574368, 3961675755),
    (176666751867, 176945375158),
    (179572739726, 179744144649),
    (11200002113, 11308653298),
    (11413422848, 11458735800),
    (130076928497, 130169603527),
    (130258047052, 130333096575),
    (18044116869, 18220819706),
    (19050491968, 19235407759),
    (19297512805, 19344291501),
    (19832338000, 19950450833),
    (150988929516, 151153005932),
    (151254926780, 151298457171),
]

OUTDOOR_INTERVALS = [
    (2377234416, 2613861694),
    (43545496077, 43574965980),
    (9271751048, 9633470578),
    (3373749916, 3476720414),
    (3639114393, 3655723578),
    (3850519320, 3918022346),
    (3960669859, 3976479425),
    (78155268579, 78222288354),
    (3862736754, 3926974637),
    (176611626918, 176666152140),
    (179420826024, 179572323250),
    (11309253029, 11412823118),
    (130170203258, 130257447322),
    (19235824236, 19296913078),
    (151153405753, 151254327050),
]


# ---------------------------------------------------------------------------
# Helper functions – geometry & coordinate transforms
# ---------------------------------------------------------------------------

def quaternion_to_euler(quaternions):
    """Convert quaternions [w, x, y, z] → Euler angles [roll, pitch, yaw].

    Parameters
    ----------
    quaternions : ndarray, shape (N, 4)

    Returns
    -------
    ndarray, shape (N, 3)
    """
    qw, qx, qy, qz = quaternions.T

    # Roll (x-axis)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx ** 2 + qy ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * np.pi / 2,
                     np.arcsin(sinp))

    # Yaw (z-axis)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy ** 2 + qz ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack((roll, pitch, yaw), axis=-1)


# ---------------------------------------------------------------------------
# Helper functions – location labelling
# ---------------------------------------------------------------------------

def get_location_label(timestamp, indoor_intervals, outdoor_intervals):
    """Return 1 (indoor), 2 (outdoor), or 0 (unlabelled)."""
    for start, end in indoor_intervals:
        if start <= timestamp <= end:
            return 1
    for start, end in outdoor_intervals:
        if start <= timestamp <= end:
            return 2
    return 0


# ---------------------------------------------------------------------------
# Helper functions – depth conversion
# ---------------------------------------------------------------------------

def ray_lengths_to_depth(ray_lengths_m, fov_deg=48.54,
                         focal_px=FOCAL_LENGTH_PX, width_px=IMAGE_WIDTH_PX):
    """Convert per-ray lengths (in metres) to a per-pixel depth vector.

    Parameters
    ----------
    ray_lengths_m : ndarray, shape (V,)
        Lengths of V uniformly-spaced rays across the FoV.

    Returns
    -------
    ndarray, shape (width_px,)
        Depth for every pixel column.
    """
    V = ray_lengths_m.shape[0]
    dv = fov_deg / (V - 1)
    f_over_w = focal_px / width_px

    angles = (np.arange(V) - np.arange(V).mean()) * np.deg2rad(dv)
    w_rays = np.tan(angles) * width_px * f_over_w + (width_px - 1) / 2
    d_rays = ray_lengths_m * np.cos(angles)

    return griddata(w_rays, d_rays, np.arange(width_px), method="linear")


def resample_depth(depth, num_columns):
    """Resample a 1-D depth vector to *num_columns* evenly-spaced samples."""
    w_old = np.arange(depth.shape[0])
    w_new = np.linspace(w_old[0], w_old[-1], num_columns)
    return griddata(w_old, depth, w_new, method="linear")


def rename_images(folder_path):
    """Rename .jpg files in *folder_path* to 00000-0.jpg, 00001-0.jpg, …"""
    files = sorted(
        (f for f in os.listdir(folder_path)
         if f.endswith(".jpg") and os.path.isfile(os.path.join(folder_path, f))),
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    for i, filename in enumerate(files):
        old = os.path.join(folder_path, filename)
        new = os.path.join(folder_path, f"{i:05d}-0.jpg")
        os.rename(old, new)


# ---------------------------------------------------------------------------
# Session-level helpers
# ---------------------------------------------------------------------------

def load_session_poses(trajectory_file, session_name):
    """Read all poses for *session_name* from the trajectories file.

    Returns
    -------
    ndarray, shape (N, 8)
        Columns: timestamp, tx, ty, tz, qw, qx, qy, qz
    """
    rows = []
    with open(trajectory_file, "r") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split(",")
            if not cols[1].strip().startswith(session_name):
                continue
            rows.append([
                int(cols[0].strip()),
                float(cols[6].strip()),  # tx
                float(cols[7].strip()),  # ty
                float(cols[8].strip()),  # tz
                float(cols[2].strip()),  # qw
                float(cols[3].strip()),  # qx
                float(cols[4].strip()),  # qy
                float(cols[5].strip()),  # qz
            ])
    return np.array(rows)


def world_to_map_poses(poses_world, affine_matrix):
    """Convert world poses to floorplan-map poses.

    Parameters
    ----------
    poses_world : ndarray, shape (N, 4)
        Columns: timestamp, x_world, y_world, yaw_world
    affine_matrix : ndarray, shape (2, 3)

    Returns
    -------
    ndarray, shape (N, 4)
        Columns: timestamp, x_map, y_map, yaw_map
    """
    positions = poses_world[:, 1:3]
    positions_map = np.array(
        [apply_affine_transformation(pt, affine_matrix) for pt in positions]
    )

    rotation = affine_matrix[:, :2]
    yaw_offset = np.pi / 2
    yaws_map = np.array(
        [apply_rotation(a, rotation) for a in poses_world[:, 3]]
    ) - yaw_offset

    return np.column_stack([poses_world[:, 0], positions_map, yaws_map])


def filter_indoor_in_bounds(poses_map, poses_raw, floorplan_shape):
    """Return a boolean mask selecting indoor poses inside the floorplan."""
    in_bounds = (
        (poses_map[:, 1] >= 0)
        & (poses_map[:, 1] <= floorplan_shape[1])
        & (poses_map[:, 2] >= 0)
        & (poses_map[:, 2] <= floorplan_shape[0])
    )
    is_indoor = np.array([
        get_location_label(int(ts), INDOOR_INTERVALS, OUTDOOR_INTERVALS) == 1
        for ts in poses_raw[:, 0]
    ])
    return in_bounds & is_indoor


def compute_ray_depths(poses_map, floorplan_occ, fov_deg, num_rays,
                       pixel_per_meter, ax=None, highlight_idx=24):
    """Raycast from every pose and return depths in metres.

    Parameters
    ----------
    poses_map : ndarray, shape (N, 4)
        Columns: timestamp, x_map, y_map, yaw_map
    floorplan_occ : ndarray
        Occupancy grid (pixel values 0–255).
    highlight_idx : int
        Pose index whose rays are drawn on *ax* (0-based).

    Returns
    -------
    ndarray, shape (N, num_rays)
        Ray lengths converted to metres.
    """
    fov_rad = np.deg2rad(fov_deg)
    n_poses = poses_map.shape[0]
    ray_lengths = np.zeros((n_poses, num_rays))

    for idx in range(n_poses):
        print(f"\r  Raycasting frame {idx + 1}/{n_poses}", end="", flush=True)
        pos = np.array([
            int(poses_map[idx, 2]),
            int(poses_map[idx, 1]),
        ])
        ang0 = poses_map[idx, 3]
        angles = np.linspace(ang0 - fov_rad / 2, ang0 + fov_rad / 2, num_rays)

        for j, ang in enumerate(angles):
            rl = ray_cast(floorplan_occ, pos, ang)
            ray_lengths[idx, j] = rl

            if ax is not None and idx == highlight_idx:
                ax.quiver(
                    int(poses_map[idx, 1]), int(poses_map[idx, 2]),
                    rl * np.cos(ang), rl * np.sin(ang),
                    width=0.002, color="g",
                    angles="xy", scale_units="xy", scale=1,
                )

    print()  # newline after progress indicator
    return ray_lengths / pixel_per_meter


def write_session_dataset(session_name, target_dir, poses_world_filtered,
                          euler_angles_filtered, depth_file, floorplan_path,
                          raw_images_dir):
    """Write all artefacts for one session into *target_dir/session_name*."""
    session_dir = os.path.join(target_dir, session_name)
    rgb_dir = os.path.join(session_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    # Poses (x, y, yaw in world frame)
    np.savetxt(
        os.path.join(session_dir, "poses.txt"),
        poses_world_filtered[:, 1:4], delimiter=" ", fmt="%.8f",
    )

    # Euler angles
    np.savetxt(
        os.path.join(session_dir, "euler_angles.txt"),
        euler_angles_filtered, delimiter=" ", fmt="%.8f",
    )

    # Copy images
    source_img_dir = os.path.join(raw_images_dir, session_name, "images")
    for ts in poses_world_filtered[:, 0].astype(np.int64):
        fname = f"{ts}.jpg"
        shutil.copy(
            os.path.join(source_img_dir, fname),
            os.path.join(rgb_dir, fname),
        )

    # Copy floorplan
    shutil.copy(floorplan_path, os.path.join(session_dir, "map.png"))

    # Depth file
    np.savetxt(
        os.path.join(session_dir, "depth90.txt"),
        depth_file, delimiter=" ", fmt="%.8f",
    )

    # Rename images to sequential IDs
    rename_images(rgb_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare the customized LaMAR HGE dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python 01_create_lamar_hge.py \\\n"
            "      --floorplans-dir /path/to/floorplan_dir \\\n"
            "      --session-dir /path/to/LaMAR/HGE/HGE/sessions/map \\\n"
            "      --target-dir ./data/LaMAR_HGE/lamar_hge"
        ),
    )
    parser.add_argument(
        "--floorplans-dir", required=True,
        help="Directory containing the floorplan image",
    )
    parser.add_argument(
        "--session-dir", required=True,
        help="LaMAR session directory (contains trajectories.txt and raw_data/)",
    )
    parser.add_argument(
        "--target-dir", required=True,
        help="Output directory for the prepared dataset",
    )
    args = parser.parse_args()

    trajectory_file = os.path.join(args.session_dir, "trajectories.txt")
    raw_data_dir = os.path.join(args.session_dir, "raw_data")
    target_dir = args.target_dir

    plt.ion()

    # Load floorplan
    floorplan_path = os.path.join(args.floorplans_dir, FLOORPLAN_FILENAME)
    floorplan = mpimg.imread(floorplan_path)
    print(f"Loaded floorplan: {floorplan_path} ({floorplan.shape[1]}x{floorplan.shape[0]} px)")

    # Compute affine transform (world → floorplan pixels)
    affine_matrix = find_affine_transform(
        TRAJECTORY_CORRESPONDENCES, FLOORPLAN_CORRESPONDENCES
    )
    pixel_per_meter = (affine_matrix[0, 1] + affine_matrix[1, 0]) / 2

    # Discover iOS sessions
    ios_sessions = sorted(
        d for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d)) and d.startswith("ios")
    )
    n_sessions = len(ios_sessions) - 1  # skip index 0
    print(f"Found {len(ios_sessions)} iOS sessions, processing {n_sessions}\n")

    # Process each session (skip index 0, matching original behaviour)
    for session_idx in range(1, len(ios_sessions)):
        session = ios_sessions[session_idx]
        print(f"[{session_idx}/{n_sessions}] {session}")

        # Load raw poses
        poses_raw = load_session_poses(trajectory_file, session)

        # Compute world-frame pose vectors
        timestamps = poses_raw[:, 0].astype(np.int64).reshape(-1, 1)
        positions_world = poses_raw[:, 1:3]
        quaternions = poses_raw[:, 4:8]

        yaw = np.arctan2(
            2 * (quaternions[:, 0] * quaternions[:, 3]
                 + quaternions[:, 1] * quaternions[:, 2]),
            1 - 2 * (quaternions[:, 2] ** 2 + quaternions[:, 3] ** 2),
        )
        euler_angles = quaternion_to_euler(quaternions)

        poses_world = np.column_stack([
            timestamps, positions_world, yaw.reshape(-1, 1)
        ])

        # Convert to map frame
        poses_map = world_to_map_poses(poses_world, affine_matrix)

        # Filter: indoor & within floorplan bounds
        mask = filter_indoor_in_bounds(poses_map, poses_raw, floorplan.shape)
        poses_world_filtered = poses_world[mask]
        poses_map_filtered = poses_map[mask]
        euler_angles_filtered = euler_angles[mask]
        print(f"  {poses_raw.shape[0]} raw poses -> {poses_world_filtered.shape[0]} indoor poses")

        # Visualise trajectory on floorplan
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(floorplan, cmap="gray")
        ax.grid()
        ax.plot(
            poses_map_filtered[:, 1].astype(int),
            poses_map_filtered[:, 2].astype(int),
            marker="o", color="r", linestyle="None", markersize=1,
            label="Trajectory",
        )
        plt.show(block=False)

        # Raycast to get floorplan depths
        occ = floorplan * 255
        highlight_idx = np.random.randint(0, poses_map_filtered.shape[0])
        ray_depths_m = compute_ray_depths(
            poses_map_filtered, occ,
            fov_deg=FOV_DEG, num_rays=NUM_RAYS,
            pixel_per_meter=pixel_per_meter,
            ax=ax, highlight_idx=highlight_idx,
        )

        # Convert ray depths → per-pixel depth → resampled depth file
        n_poses = ray_depths_m.shape[0]
        depth_file = np.zeros((n_poses, DEPTH_COLUMNS))
        for i in range(n_poses):
            d_full = ray_lengths_to_depth(ray_depths_m[i])
            depth_file[i] = resample_depth(d_full, DEPTH_COLUMNS)

        # Write dataset
        write_session_dataset(
            session_name=session,
            target_dir=target_dir,
            poses_world_filtered=poses_world_filtered,
            euler_angles_filtered=euler_angles_filtered,
            depth_file=depth_file,
            floorplan_path=floorplan_path,
            raw_images_dir=raw_data_dir,
        )
        print(f"  Written to {os.path.join(target_dir, session)}\n")

    print(f"Done — {n_sessions} sessions written to {target_dir}")
    plt.ioff()
    plt.show()
    input("Press Enter to close the figures and exit...")


if __name__ == "__main__":
    main()