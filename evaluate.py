# Adapted from F3Loc (Chen et al., CVPR 2024; MIT License).
# https://github.com/felix-ch/f3loc

"""
Evaluation script for floorplan localization using UnLoc.

Runs the forward UnLoc filter on test trajectories and computes the metrics
reported in the paper.

Supports two dataset variants:
  - gibson_t                 (Gibson synthetic dataset)
  - lamar_hge                (LaMAR HGE, full floorplan)

Usage:
    python evaluate.py --dataset_path /path/to/Gibson_Floorplan_Localization_Dataset --dataset gibson_t
    python evaluate.py --dataset_path /path/to/LaMAR_HGE --dataset lamar_hge
"""

import argparse
import os
import random
import time
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*__array_wrap__.*",
    category=DeprecationWarning,
)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml

from modules.depth_net_pl import UnLocDepthModule
from utils.data_utils import (
    GibsonTrajectoryDataset,
    LaMARHGETrajectoryDataset,
)
from utils.localization_utils import (
    get_ray_from_depth_uncertainty,
    localize_noflip_uncertainty,
    localize_uncertainty,
    transit,
)
from utils.geometry import (
    find_affine_transform,
    get_rel_pose,
    world_to_map_hge_complete,
    map_to_world_hge_complete,
)
from utils.postprocessing import postprocess_trajectory

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UnLoc.")
    # --- Required paths ---
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dataset", choices=["gibson_t", "lamar_hge"], required=True)

    # --- Filter / evaluation settings ---
    parser.add_argument("--traj_len", type=int, default=100)
    parser.add_argument("--dt", type=int, default=1)
    parser.add_argument("--desdf_resolution", type=float, default=0.1)
    parser.add_argument("--orn_slice", type=int, default=None,
                        help="Default: 36 for Gibson, 72 for LaMAR HGE.")

    # --- Toggles & output ---
    parser.add_argument("--no-postprocess", dest="postprocess", action="store_false", default=True,
                        help="Disable SE(2) pose refinement (on by default).")
    parser.add_argument("--no-save_results", dest="save_results", action="store_false", default=True,
                        help="Disable saving results to results.npz (on by default).")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save likelihood/posterior plots per timestep.")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.orn_slice is None:
        args.orn_slice = 36 if args.dataset == "gibson_t" else 72
    return args


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def is_hge(dataset: str) -> bool:
    return dataset == "lamar_hge"

def is_gibson(dataset: str) -> bool:
    return dataset == "gibson_t"


# ---------------------------------------------------------------------------
# Coordinate transforms & calibration
# ---------------------------------------------------------------------------

# Calibration correspondences for mapping world coordinates (meters) to
# HGE floorplan pixels. Four door thresholds spanning the floor area
# were manually identified in both the floorplan image and the LaMAR
# trajectories; an affine transform is fit from these pairs.
# See paper Sec. A.1.2.
FLOORPLAN_CORRESPONDENCES = np.array(
    [[1516, 490.5], [1532, 2162], [350, 1515], [391, 135]]
)
TRAJECTORY_CORRESPONDENCES = np.array(
    [[-12.600, 10.103], [77.981, 9.882], [42.641, -54.348], [-33.099, -51.229]]
)

# Gibson uses a fixed resolution of 0.01 m/pixel
GIBSON_RESOLUTION = 0.01


def compute_calibration():
    """Compute the affine transform from world to floorplan pixel coords (HGE only)."""
    affine_matrix = find_affine_transform(
        TRAJECTORY_CORRESPONDENCES, FLOORPLAN_CORRESPONDENCES,
    )
    pixel_per_meter = (affine_matrix[0, 1] + affine_matrix[1, 0]) / 2
    return affine_matrix, pixel_per_meter


def world_to_map_gibson(pose, h, w):
    """Convert Gibson world-frame pose to map pixel coords."""
    x = pose[0] / GIBSON_RESOLUTION + w / 2
    y = pose[1] / GIBSON_RESOLUTION + h / 2
    return x, y, pose[2]


def load_ground_truth_poses(dataset, dataset_dir, scene_names, traj_len,
                            affine_matrix=None):
    """Load GT poses for every scene and convert from world to map coords."""
    maps, gt_poses = {}, {}
    for scene in tqdm.tqdm(scene_names, desc="Loading poses & maps"):
        occ = cv2.imread(os.path.join(dataset_dir, scene, "map.png"))[:, :, 0]
        maps[scene] = occ
        h, w = occ.shape[:2]

        with open(os.path.join(dataset_dir, scene, "poses.txt"), "r") as f:
            poses_txt = [line.strip() for line in f.readlines()]

        n_poses = len(poses_txt)
        n_poses -= n_poses % traj_len  # trim to whole trajectories

        poses = np.zeros([0, 3], dtype=np.float32)
        for state_id in range(n_poses):
            pose = np.array(poses_txt[state_id].split(" "), dtype=np.float32)

            if dataset == "lamar_hge":
                (x, y), th = world_to_map_hge_complete(
                    position_world=pose[:2],
                    orientation_world_rad=pose[2],
                    affine_matrix=affine_matrix,
                )
            else:  # gibson_t
                x, y, th = world_to_map_gibson(pose, h, w)

            poses = np.concatenate(
                (poses, np.array([[x, y, th]], dtype=np.float32)), axis=0
            )

        gt_poses[scene] = poses
    return maps, gt_poses


# ---------------------------------------------------------------------------
# Build the test dataset
# ---------------------------------------------------------------------------

def build_test_dataset(args, dataset_dir, depth_dir, split):
    """Instantiate the appropriate TrajDataset for the chosen configuration."""
    depth_suffix = "depth90" if is_hge(args.dataset) else "depth40"

    common_kwargs = dict(
        L=args.traj_len,
        depth_dir=depth_dir,
        depth_suffix=depth_suffix,
        add_rp=False,
        roll=0,
        pitch=0,
        without_depth=False,
    )

    if is_hge(args.dataset):
        return LaMARHGETrajectoryDataset(dataset_dir, split["test"], **common_kwargs)
    return GibsonTrajectoryDataset(dataset_dir, split["test"], **common_kwargs)


# ---------------------------------------------------------------------------
# Load depth network
# ---------------------------------------------------------------------------

def load_depth_network(args, device):
    """Load the DepthAnything uncertainty depth network checkpoint."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        net = UnLocDepthModule.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path, strict=False,
        )
    return net.to(device)


# ---------------------------------------------------------------------------
# Load DeSDF volumes
# ---------------------------------------------------------------------------

def load_desdfs(desdf_path, scene_names, dataset):
    """Load DeSDF volumes — one shared volume for HGE, per-scene for Gibson."""
    if is_hge(dataset):
        desdf = np.load(
            os.path.join(desdf_path, scene_names[0], "desdf.npy"),
            allow_pickle=True,
        ).item()
        return {"__shared__": desdf}

    return {
        scene: np.load(
            os.path.join(desdf_path, scene, "desdf.npy"), allow_pickle=True
        ).item()
        for scene in tqdm.tqdm(scene_names, desc="Loading DeSDF")
    }


def get_desdf(desdfs, scene, dataset):
    if is_hge(dataset):
        return desdfs["__shared__"]
    return desdfs[scene]


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def get_original_resolution(dataset, pixel_per_meter=None):
    """Return the original map resolution [m/pixel]."""
    return 1.0 / pixel_per_meter if is_hge(dataset) else GIBSON_RESOLUTION


def desdf_to_map(pose, desdf, resolution_ratio, dataset):
    """Convert a DeSDF-grid pose [grid_x, grid_y, theta] to map-pixel coords."""
    pose_map = pose.copy()
    scale = resolution_ratio if is_hge(dataset) else 10
    pose_map[0] = pose[0] * scale + desdf["l"]
    pose_map[1] = pose[1] * scale + desdf["t"]
    return pose_map


def map_to_desdf(poses_map, desdf, resolution_ratio, dataset):
    """Convert map-pixel poses to DeSDF grid coords."""
    gt_pose_desdf = poses_map.copy()
    scale = resolution_ratio if is_hge(dataset) else 10
    gt_pose_desdf[:, 0] = (gt_pose_desdf[:, 0] - desdf["l"]) / scale
    gt_pose_desdf[:, 1] = (gt_pose_desdf[:, 1] - desdf["t"]) / scale
    return gt_pose_desdf


# ---------------------------------------------------------------------------
# Depth-to-ray conversion
# ---------------------------------------------------------------------------

def predict_depth_and_rays(d_net, imgs, masks, t, args, device):
    """Run depth network and convert predictions to rays with uncertainty."""
    if is_hge(args.dataset):
        input_img = imgs[:, t, :, :]
        input_mask = masks[:, t, :, :] if masks is not None else None
    else:
        input_img = imgs[:, t, :, :, :] if imgs.dim() == 5 else imgs[:, t, :, :]
        input_mask = None

    loc, scale, _, _ = d_net.encoder(input_img, input_mask)
    pred_depths = loc.squeeze(0).detach().cpu().numpy()
    pred_scales = scale.squeeze(0).detach().cpu().numpy()

    fov_desdf = DATASET_FOV_DESDF[args.dataset]
    if fov_desdf is not None:
        dv = 360 / args.orn_slice
        V = fov_desdf / dv
        pred_rays, scales_rays = get_ray_from_depth_uncertainty(
            pred_depths, pred_scales, V=V, dv=dv, F_W=1596 / 1440,
        )
    else:
        pred_rays, scales_rays = get_ray_from_depth_uncertainty(
            pred_depths, pred_scales,
        )

    return (
        pred_depths, pred_scales,
        torch.tensor(pred_rays, device=device),
        torch.tensor(scales_rays, device=device),
    )


# ---------------------------------------------------------------------------
# Observation likelihood
# ---------------------------------------------------------------------------

def compute_likelihood(desdf_tensor, pred_rays, scales_rays, prior_device, args):
    """Compute observation likelihood using uncertainty-aware matching."""
    if is_hge(args.dataset):
        return localize_noflip_uncertainty(
            desdf_tensor.to(prior_device),
            pred_rays.to(prior_device),
            scales_rays.to(prior_device),
            return_np=False,
            orn_slice=args.orn_slice,
        )
    return localize_uncertainty(
        desdf_tensor.to(prior_device),
        pred_rays.to(prior_device),
        scales_rays.to(prior_device),
        return_np=False,
        orn_slice=args.orn_slice,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

SUCCESS_THRESHOLDS_M = [10.0, 5.0, 2.0, 1.0, 0.5, 0.3, 0.2]
RECALL_THRESHOLDS_M  = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def compute_pose_error(dataset, pose_in_map, gt_pose_map, affine_matrix):
    """Return (position_error_m, orientation_error_deg) between prediction and GT."""
    if dataset == "lamar_hge":
        pred_pos, pred_orn_rad = map_to_world_hge_complete(
            pose_in_map[:2], pose_in_map[2], affine_matrix,
        )
        gt_pos, gt_orn_rad = map_to_world_hge_complete(
            gt_pose_map[:2], gt_pose_map[2], affine_matrix,
        )
    else:  # gibson_t
        pred_pos = pose_in_map[:2] * GIBSON_RESOLUTION
        gt_pos = gt_pose_map[:2] * GIBSON_RESOLUTION
        pred_orn_rad = pose_in_map[2]
        gt_orn_rad = gt_pose_map[2]

    pos_err = np.linalg.norm(pred_pos - gt_pos)
    d_deg = (np.degrees(pred_orn_rad) - np.degrees(gt_orn_rad)) % 360
    orn_err = min(d_deg, 360 - d_deg)
    return pos_err, orn_err


def compute_success_flags(last_errors):
    """Return a list of bools: whether *all* last-10 errors < threshold."""
    return [bool(np.all(last_errors < th)) for th in SUCCESS_THRESHOLDS_M]


def compute_recalls(position_errors, orientation_errors):
    """Recall at various distance thresholds + joint 1m/30deg recall."""
    n = position_errors.shape[0]
    recalls = [np.sum(position_errors < th) / n for th in RECALL_THRESHOLDS_M]
    recall_1m_30deg = np.sum(
        np.logical_and(position_errors < 1.0, orientation_errors < 30)
    ) / n
    # Order: 0.1m, 0.5m, 1m, 1m+30deg, 2m, 5m, 10m
    return [recalls[0], recalls[1], recalls[2], recall_1m_30deg,
            recalls[3], recalls[4], recalls[5]]


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------

def save_filter_plots(evol_path, data_idx, t,
                      likelihood_2d, likelihood_pred,
                      posterior_2d, pose, gt_pose_desdf_t,
                      dataset):
    """Save likelihood and posterior heatmaps with predicted/GT pose arrows."""
    s = 0.5
    fig = plt.figure(figsize=(16, 8))

    origin_kw = {} if is_hge(dataset) else {"origin": "lower"}
    quiver_kwargs = dict(
        color="blue", width=s * 0.2, scale_units="inches", units="inches",
        scale=1, headwidth=3, headlength=3, headaxislength=3, minlength=s * 0.1,
    )
    gt_quiver_kwargs = dict(quiver_kwargs, color="green")

    for ax_idx, (grid, title, pred) in enumerate([
        (likelihood_2d, f"{t} likelihood", likelihood_pred),
        (posterior_2d,  f"{t} posterior",  pose),
    ], start=1):
        ax = fig.add_subplot(1, 2, ax_idx)
        ax.imshow(grid, cmap="coolwarm", **origin_kw)
        ax.set_title(title)
        ax.axis("off")
        ax.quiver(pred[0], pred[1],
                  s * np.cos(pred[2]), s * np.sin(pred[2]), **quiver_kwargs)
        ax.quiver(gt_pose_desdf_t[0], gt_pose_desdf_t[1],
                  s * np.cos(gt_pose_desdf_t[2]), s * np.sin(gt_pose_desdf_t[2]),
                  **gt_quiver_kwargs)

    out_dir = os.path.join(evol_path, str(data_idx))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{t}.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fixed hyperparameters (paper settings, not exposed via CLI)
# ---------------------------------------------------------------------------

# Motion model parameters per dataset. These reproduce the paper results
# and are not intended to be tuned per-run.
MOTION_SIGMAS = {
    "gibson_t":  dict(sig_x=0.1, sig_y=0.1, sig_o=0.1, tsize=7,  rsize=7),
    "lamar_hge": dict(sig_x=0.1, sig_y=0.1, sig_o=0.1, tsize=71, rsize=71),
}

# Field-of-view (in degrees) used for DeSDF ray matching.
# None means use the function's default (Gibson).
DATASET_FOV_DESDF = {
    "gibson_t": None,
    "lamar_hge": 49,
}

# Post-processing (SE(2) trajectory refinement)
POSTPROC_T_REFINE = 10
POSTPROC_ITERS = 100
POSTPROC_LR_XY = 1e-3
POSTPROC_LR_THETA = 1e-5


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"======= USING DEVICE: {device} =======")

    # --- Paths ---
    if is_hge(args.dataset):
        dataset_dir = os.path.join(args.dataset_path, args.dataset)
    else:
        dataset_dir = os.path.join(args.dataset_path, args.dataset)

    depth_dir = dataset_dir
    desdf_path = os.path.join(args.dataset_path, "desdf")
    output_dir = os.path.join(
        args.output_dir, args.dataset,
        f"res{args.desdf_resolution}_orn{args.orn_slice}"
        f"_T{args.traj_len}_dt{args.dt}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # --- Calibration (HGE only) ---
    affine_matrix = None
    if is_hge(args.dataset):
        affine_matrix, pixel_per_meter = compute_calibration()
        original_resolution = get_original_resolution(args.dataset, pixel_per_meter)
        print(f"Affine transformation matrix:\n{affine_matrix}")
        print(f"pixel_per_meter: {pixel_per_meter}")
    else:
        original_resolution = GIBSON_RESOLUTION

    resolution_ratio = args.desdf_resolution / original_resolution

    # --- Dataset ---
    traj_l = args.traj_len
    with open(os.path.join(dataset_dir, "split.yaml"), "r") as f:
        split = yaml.safe_load(f)

    test_set = build_test_dataset(args, dataset_dir, depth_dir, split)
    print(f"Test set size: {len(test_set)}")

    # --- Depth network ---
    d_net = load_depth_network(args, device)

    # --- DeSDF ---
    print("Loading DeSDF ...")
    desdfs = load_desdfs(desdf_path, test_set.scene_names, args.dataset)

    # --- Ground-truth poses ---
    print("Loading poses and maps ...")
    _, gt_poses = load_ground_truth_poses(
        args.dataset, dataset_dir, test_set.scene_names, traj_l, affine_matrix,
    )

    # ------------------------------------------------------------------
    # Accumulators for metrics
    # ------------------------------------------------------------------
    # Successful and all sequences. Computed from per-sequence arrays below.
    success_flags_per_seq   = []  # (N, len(SUCCESS_THRESHOLDS_M)) bool
    rmse_per_seq            = []  # (N,) float
    # Recall @ thresholds + depth quality.
    recalls_per_seq         = []  # (N, 7) float
    depth_mae_per_seq       = []  # (N,) float, mean |d_pred - d_gt| per frame
    depth_cos_sim_per_seq   = []  # (N,) float, mean cosine similarity per frame
    # Post-processed metrics.
    refined_success_flags_per_seq = []
    refined_rmse_per_seq          = []

    # Timing
    matching_time = 0.0
    feature_extraction_time = 0.0
    iteration_time = 0.0
    n_iter = 0
    post_processing_time = 0.0
    n_post_processing = 0

    # Motion model parameters for this dataset (paper settings, fixed).
    motion = MOTION_SIGMAS[args.dataset]

    # ------------------------------------------------------------------
    # Loop over test trajectories
    # ------------------------------------------------------------------
    for data_idx in tqdm.tqdm(range(len(test_set)), desc="Evaluating"):
        data = test_set[data_idx]

        scene_idx = np.sum(data_idx >= np.array(test_set.scene_start_idx)) - 1
        scene = test_set.scene_names[scene_idx]
        idx_within_scene = data_idx - test_set.scene_start_idx[scene_idx]

        desdf = get_desdf(desdfs, scene, args.dataset)
        poses_map = gt_poses[scene][
            idx_within_scene * traj_l : idx_within_scene * traj_l + traj_l, :
        ]
        gt_pose_desdf = map_to_desdf(poses_map, desdf, resolution_ratio, args.dataset)

        imgs = torch.tensor(data["imgs"], device=device).unsqueeze(0)
        masks = None
        if "masks" in data:
            masks = torch.tensor(data["masks"], device=device).unsqueeze(0)

        poses_tensor = None
        if "poses" in data and is_gibson(args.dataset):
            poses_tensor = torch.from_numpy(np.stack(data["poses"])).to(device).unsqueeze(0)

        prior = torch.tensor(
            np.ones_like(desdf["desdf"]) / desdf["desdf"].size, device=device
        ).to(torch.float32)

        pred_poses_map = []

        # Per-frame buffers we need inside this sequence (aggregated before
        # moving to the next sequence — not stored long-term).
        obs_pos_errs = []
        obs_orn_errs = []
        depth_l1s   = []
        depth_coss  = []

        # For post-processing
        pose_ls = []
        transition_ls = []
        pred_depths_ls = []
        pred_uncert_ls = []

        desdf_tensor = torch.tensor(desdf["desdf"])

        # ----------------------------------------------------------
        # Forward Bayes filter over timesteps
        # ----------------------------------------------------------
        for t in range(traj_l):

            # === Timed core pipeline ===
            torch.cuda.synchronize()
            t_iter_start = time.time()
            t_feat_start = time.time()

            pred_depths, pred_scales, pred_rays, scales_rays = (
                predict_depth_and_rays(d_net, imgs, masks, t, args, device)
            )
            t_feat_end = time.time()

            t_match_start = time.time()
            likelihood, likelihood_2d, likelihood_orn, likelihood_pred = (
                compute_likelihood(
                    desdf_tensor, pred_rays, scales_rays, prior.device, args,
                )
            )
            torch.cuda.synchronize()
            t_match_end = time.time()

            # Bayesian update
            if t % args.dt == 0:
                posterior = prior * likelihood.to(prior.device)
            else:
                posterior = prior
            posterior = posterior / (posterior.sum() + 1e-10)

            # MAP estimate
            posterior_2d, orientations = torch.max(posterior, dim=2)
            pose_y, pose_x = torch.where(posterior_2d == posterior_2d.max())
            if pose_y.shape[0] > 1:
                pose_y = pose_y[0].unsqueeze(0)
                pose_x = pose_x[0].unsqueeze(0)
            orn = orientations[pose_y, pose_x] / args.orn_slice * 2 * torch.pi
            pose = torch.cat((pose_x, pose_y, orn)).detach().cpu().numpy()

            pose_in_map = desdf_to_map(pose, desdf, resolution_ratio, args.dataset)
            pred_poses_map.append(pose_in_map)
            pose_ls.append(pose)
            pred_depths_ls.append(pred_depths)
            pred_uncert_ls.append(pred_scales)

            # Transition (motion model)
            if t < traj_l - 1:
                if is_hge(args.dataset):
                    curr = torch.from_numpy(gt_pose_desdf[t, :])
                    nxt  = torch.from_numpy(gt_pose_desdf[t + 1, :])
                    current_pose = torch.tensor([
                        curr[0] * args.desdf_resolution,
                        curr[1] * args.desdf_resolution,
                        curr[2],
                    ], device=curr.device)
                    next_pose = torch.tensor([
                        nxt[0] * args.desdf_resolution,
                        nxt[1] * args.desdf_resolution,
                        nxt[2],
                    ], device=nxt.device)
                else:
                    current_pose = poses_tensor[0, t, :]
                    next_pose = poses_tensor[0, t + 1, :]

                transition = get_rel_pose(current_pose, next_pose)
                transition_ls.append(transition.cpu().numpy())

                if is_hge(args.dataset):
                    prior = transit(
                        posterior, transition,
                        sig_o=motion["sig_o"], sig_x=motion["sig_x"], sig_y=motion["sig_y"],
                        tsize=motion["tsize"], rsize=motion["rsize"],
                        resolution=args.desdf_resolution,
                    )
                else:
                    prior = transit(
                        posterior, transition,
                        sig_o=motion["sig_o"], sig_x=motion["sig_x"], sig_y=motion["sig_y"],
                        tsize=motion["tsize"], rsize=motion["rsize"],
                    )

            torch.cuda.synchronize()
            t_iter_end = time.time()
            # === End timed section ===

            feature_extraction_time += t_feat_end - t_feat_start
            matching_time += t_match_end - t_match_start
            iteration_time += t_iter_end - t_iter_start
            n_iter += 1

            # --- Per-frame metrics ---
            gt_depths_frame = np.array(data["gt_depth"])[t, :]
            depth_l1s.append(
                F.l1_loss(torch.tensor(pred_depths),
                          torch.tensor(gt_depths_frame)).item()
            )
            depth_coss.append(
                F.cosine_similarity(torch.tensor(pred_depths),
                                    torch.tensor(gt_depths_frame),
                                    dim=-1).mean().item()
            )

            # Observation-only pose (pre-filter MAP of likelihood) — used to
            # compute single-frame recall in.
            obs_pose_in_map = desdf_to_map(
                np.array([likelihood_pred[0], likelihood_pred[1], likelihood_pred[2]]),
                desdf, resolution_ratio, args.dataset,
            )
            obs_pos_err, obs_orn_err = compute_pose_error(
                args.dataset, obs_pose_in_map, poses_map[t], affine_matrix,
            )
            obs_pos_errs.append(obs_pos_err)
            obs_orn_errs.append(obs_orn_err)

            # Optional plots
            if args.save_plots:
                save_filter_plots(
                    output_dir, data_idx, t,
                    likelihood_2d, likelihood_pred,
                    posterior_2d.detach().cpu().numpy(), pose,
                    gt_pose_desdf[t], args.dataset,
                )

        # ----------------------------------------------------------
        # Per-sequence aggregation
        # ----------------------------------------------------------
        pred_poses_map = np.stack(pred_poses_map)

        last_errors = (
            ((pred_poses_map[-10:, :2] - poses_map[-10:, :2]) ** 2).sum(axis=1)
            ** 0.5
        ) * original_resolution
        rmse = (
            ((pred_poses_map[-10:, :2] - poses_map[-10:, :2]) ** 2)
            .sum(axis=1).mean()
        ) ** 0.5 * original_resolution

        success_flags_per_seq.append(compute_success_flags(last_errors))
        rmse_per_seq.append(rmse)
        recalls_per_seq.append(
            compute_recalls(np.array(obs_pos_errs), np.array(obs_orn_errs))
        )
        depth_mae_per_seq.append(float(np.mean(depth_l1s)))
        depth_cos_sim_per_seq.append(float(np.mean(depth_coss)))

        # ----------------------------------------------------------
        # Post-processing
        # ----------------------------------------------------------
        if args.postprocess:
            pp_start = time.time()
            refined_poses = postprocess_trajectory(
                pose_ls=pose_ls,
                pred_depths_ls=pred_depths_ls,
                pred_uncert_ls=pred_uncert_ls,
                transition_ls=transition_ls,
                desdf=torch.tensor(desdf["desdf"]).to(device),
                desdf_resolution=args.desdf_resolution,
                orn_slice=args.orn_slice,
                T_refine=POSTPROC_T_REFINE,
                iters=POSTPROC_ITERS,
                lr_xy=POSTPROC_LR_XY,
                lr_theta=POSTPROC_LR_THETA,
            )
            post_processing_time += time.time() - pp_start
            n_post_processing += 1

            refined_poses_map = refined_poses.cpu().numpy().copy()
            scale = resolution_ratio if is_hge(args.dataset) else 10
            refined_poses_map[:, 0] = refined_poses_map[:, 0] * scale + desdf["l"]
            refined_poses_map[:, 1] = refined_poses_map[:, 1] * scale + desdf["t"]

            n_refined = refined_poses_map.shape[0]
            n_eval = min(10, n_refined)
            gt_tail = poses_map[-n_refined:]
            refined_last_errors = (
                ((refined_poses_map[-n_eval:, :2] - gt_tail[-n_eval:, :2]) ** 2)
                .sum(axis=1) ** 0.5
            ) * original_resolution
            refined_rmse = (
                ((refined_poses_map[-n_eval:, :2] - gt_tail[-n_eval:, :2]) ** 2)
                .sum(axis=1).mean()
            ) ** 0.5 * original_resolution

            refined_success_flags_per_seq.append(
                compute_success_flags(refined_last_errors)
            )
            refined_rmse_per_seq.append(refined_rmse)

    # ------------------------------------------------------------------
    # Convert accumulators to arrays
    # ------------------------------------------------------------------
    N = len(test_set)
    success_flags         = np.asarray(success_flags_per_seq, dtype=bool)    # (N, T)
    rmses                 = np.asarray(rmse_per_seq, dtype=np.float32)       # (N,)
    recalls               = np.asarray(recalls_per_seq, dtype=np.float32)    # (N, 7)
    depth_maes            = np.asarray(depth_mae_per_seq, dtype=np.float32)  # (N,)
    depth_cos_sims        = np.asarray(depth_cos_sim_per_seq, dtype=np.float32)

    refined_success_flags = (
        np.asarray(refined_success_flags_per_seq, dtype=bool)
        if refined_success_flags_per_seq else np.empty((0, len(SUCCESS_THRESHOLDS_M)), dtype=bool)
    )
    refined_rmses = (
        np.asarray(refined_rmse_per_seq, dtype=np.float32)
        if refined_rmse_per_seq else np.empty(0, dtype=np.float32)
    )

    # ------------------------------------------------------------------
    # Print summary (paper numbers)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Forward filter:")
    for i, th in enumerate(SUCCESS_THRESHOLDS_M):
        sr = success_flags[:, i].mean() if N else 0.0
        print(f"    Success @ {th:4.1f} m : {sr:.4f}")
    succ_1m = success_flags[:, SUCCESS_THRESHOLDS_M.index(1.0)]
    if succ_1m.any():
        print(f"    RMSE (succeeded @ 1m)  : {rmses[succ_1m].mean():.4f}")
    print(f"    RMSE (all)             : {rmses.mean():.4f}")

    print("=" * 60)
    print("  Single-frame depth & recall:")
    print(f"    Mean depth MAE         : {depth_maes.mean():.4f}")
    print(f"    Mean depth cos. sim.   : {depth_cos_sims.mean():.4f}")
    mean_recalls = recalls.mean(axis=0)
    for th, r in zip(["0.1m", "0.5m", "1m", "1m+30°", "2m", "5m", "10m"], mean_recalls):
        print(f"    Recall @ {th:<7}     : {r:.4f}")

    if n_iter > 0:
        print("=" * 60)
        print("  Runtime per frame:")
        print(f"    Feature extraction     : {feature_extraction_time / n_iter:.4f} s")
        print(f"    Matching               : {matching_time / n_iter:.4f} s")
        print(f"    Total iteration        : {iteration_time / n_iter:.4f} s")

    if args.postprocess and refined_rmses.size > 0:
        print("=" * 60)
        print("  Post-processed:")
        for i, th in enumerate(SUCCESS_THRESHOLDS_M):
            sr = refined_success_flags[:, i].mean()
            print(f"    Refined success @ {th:4.1f} m : {sr:.4f}")
        refined_succ_1m = refined_success_flags[:, SUCCESS_THRESHOLDS_M.index(1.0)]
        if refined_succ_1m.any():
            print(f"    Refined RMSE (succeeded @ 1m) : {refined_rmses[refined_succ_1m].mean():.4f}")
        print(f"    Refined RMSE (all)            : {refined_rmses.mean():.4f}")
        if n_post_processing > 0:
            print(f"    Avg post-processing time      : {post_processing_time / n_post_processing:.4f} s")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.save_results:
        filepath = os.path.join(output_dir, "results.npz")
        np.savez(
            filepath,
            # --- config ---
            dataset=args.dataset,
            checkpoint=args.checkpoint_path,
            traj_len=args.traj_len,
            seed=args.seed,
            success_thresholds_m=np.asarray(SUCCESS_THRESHOLDS_M),
            recall_thresholds=np.asarray(
                ["0.1m", "0.5m", "1m", "1m+30deg", "2m", "5m", "10m"]
            ),
            # --- per-sequence arrays ---
            success_flags=success_flags,          # (N, len(success_thresholds_m))
            rmses=rmses,                          # (N,)
            recalls=recalls,                      # (N, 7)
            depth_maes=depth_maes,                # (N,)
            depth_cos_sims=depth_cos_sims,        # (N,)
            refined_success_flags=refined_success_flags,
            refined_rmses=refined_rmses,
            # --- timing (scalar seconds/frame) ---
            mean_feature_extraction_s=feature_extraction_time / max(n_iter, 1),
            mean_matching_s=matching_time / max(n_iter, 1),
            mean_iteration_s=iteration_time / max(n_iter, 1),
            mean_postproc_s=(
                post_processing_time / n_post_processing if n_post_processing else 0.0
            ),
        )
        print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    main()