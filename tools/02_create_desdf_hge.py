# Adapted from F3Loc (Chen et al., CVPR 2024; MIT License).
# https://github.com/felix-ch/f3loc

"""
Generate the directional ESDF (DeSDF) for the LaMAR HGE floorplan and
distribute it to each test session folder.

Given the pre-edited floorplan image, this script:
  1. Raycasts the floorplan at every (row, col, orientation) grid cell to
     produce a DeSDF volume of shape (H', W', orn_slice).
  2. Copies the resulting ``desdf.npy`` into every test session folder
     listed in ``split.yaml`` (mirroring the source folder structure).

The DeSDF is shared across all sessions of the HGE building because the
floorplan is fixed; per-session copies are provided for compatibility with
the loader in ``evaluate.py``.

Required inputs:
  - The pre-edited HGE floorplan image (``map_HGE.png``).
  - A ``split.yaml`` file listing the test sessions (used to determine
    the output subfolder names).
"""

import argparse
import multiprocessing
import os
import time

import matplotlib.image as mpimg
import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.geometry import ray_cast


# ---------------------------------------------------------------------------
# DeSDF generation
# ---------------------------------------------------------------------------

def raycast_desdf_parallel(
    occ,
    orn_slice=36,
    max_dist=10,
    original_resolution=0.01,
    resolution=0.1,
):
    """Raycast an occupancy map at all (row, col, orientation) cells in parallel.

    Parallelises across orientation bins with joblib (one worker per bin).

    Parameters
    ----------
    occ : np.ndarray, (H, W)
        Occupancy map (0 = free, 255 = occupied).
    orn_slice : int
        Number of equiangular orientation bins.
    max_dist : float
        Maximum raycasting distance [m].
    original_resolution : float
        Resolution of ``occ`` [m/pixel].
    resolution : float
        Output DeSDF resolution [m/cell].

    Returns
    -------
    desdf : np.ndarray, (H', W', orn_slice)
        Directional ESDF in metres.
    """
    ratio = resolution / original_resolution
    h, w = occ.shape
    desdf_shape = (int(h // ratio), int(w // ratio), orn_slice)
    desdf = np.zeros(desdf_shape)

    def process_orientation(o):
        """Raycast the whole grid at a single orientation bin ``o``."""
        theta = o / orn_slice * 2 * np.pi
        local_desdf = np.zeros((desdf.shape[0], desdf.shape[1]))

        with tqdm(
            total=desdf.shape[0],
            desc=f"Orientation {o}/{orn_slice}",
            position=o + 1, leave=False,
        ) as pbar:
            for row in range(desdf.shape[0]):
                for col in range(desdf.shape[1]):
                    pos = np.array([row, col]) * ratio
                    local_desdf[row, col] = ray_cast(
                        occ, pos, theta, max_dist / original_resolution,
                    )
                pbar.update(1)

        return local_desdf

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(process_orientation)(o) for o in range(orn_slice)
    )

    for o, result in enumerate(results):
        desdf[:, :, o] = result

    return desdf * original_resolution


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate the DeSDF for the LaMAR HGE floorplan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python tools/02_create_desdf_hge.py \\\n"
            "      --floorplan path/to/map_HGE.png \\\n"
            "      --split-yaml data/LaMAR_HGE/lamar_hge/split.yaml \\\n"
            "      --output-dir data/LaMAR_HGE/desdf"
        ),
    )
    parser.add_argument(
        "--floorplan", required=True,
        help="Path to the pre-edited HGE floorplan image (e.g. map_HGE.png).",
    )
    parser.add_argument(
        "--split-yaml", required=True,
        help="Path to split.yaml listing the test sessions.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where per-session desdf.npy files will be written.",
    )
    args = parser.parse_args()

    # --- Fixed (paper) settings ---
    pixel_per_meter = 18.315046895211292
    original_resolution = 1 / pixel_per_meter
    resolution = 0.1
    orn_slice = 72
    max_dist = 10
    desdf_filename = "desdf.npy"

    # --- Generate desdf once ---
    print(f"Loading floorplan from: {args.floorplan}")
    floorplan = mpimg.imread(args.floorplan)
    occ = floorplan * 255

    print("Generating desdf...")
    t0 = time.time()
    desdf = raycast_desdf_parallel(
        occ,
        orn_slice=orn_slice,
        max_dist=max_dist,
        original_resolution=original_resolution,
        resolution=resolution,
    )
    print(f"Desdf generated in {time.time() - t0:.1f}s, shape={desdf.shape}")

    # Save a local backup
    np.save(desdf_filename, {"l": 0, "t": 0, "desdf": desdf})
    print(f"Saved local backup: {desdf_filename}")

    # --- Load test split ---
    with open(args.split_yaml, "r") as f:
        split = yaml.safe_load(f)
    test_folders = split.get("test", [])
    print(f"Found {len(test_folders)} test folders in {args.split_yaml}")

    # --- Copy desdf.npy into each test folder (mirroring source structure) ---
    for folder_name in test_folders:
        dst_folder = os.path.join(args.output_dir, folder_name)
        os.makedirs(dst_folder, exist_ok=True)
        dst_file = os.path.join(dst_folder, desdf_filename)
        np.save(dst_file, {"l": 0, "t": 0, "desdf": desdf})
        print(f"  [OK] Wrote {dst_file}")

    print("Done.")


if __name__ == "__main__":
    main()