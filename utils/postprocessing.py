"""
Post-processing for Bayesian filter localization.

Refines the last T_refine poses of a trajectory by optimising a global
SE(2) rigid-body correction (Δx, Δy, Δθ) that minimises the Laplace
depth-matching loss against the DeSDF floor plan.

The pipeline:
  1. Initialise poses by chaining the histogram-filter argmax at the last
     timestep backwards through the GT-derived relative transitions.
  2. Run gradient-based optimisation (Adam) of the global correction,
     using differentiable trilinear interpolation into the padded DeSDF.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SE(2) helpers (batched)
# ---------------------------------------------------------------------------

def pose_vec_to_mat(pose: torch.Tensor) -> torch.Tensor:
    """
    Convert pose vectors to 3×3 homogeneous SE(2) matrices.

    Parameters
    ----------
    pose : (..., 3)  –  [x, y, θ]

    Returns
    -------
    M : (..., 3, 3)
    """
    squeezed = pose.dim() == 1
    if squeezed:
        pose = pose.unsqueeze(0)

    x, y, th = pose[..., 0], pose[..., 1], pose[..., 2]
    c, s = torch.cos(th), torch.sin(th)

    M = torch.zeros(*pose.shape[:-1], 3, 3,
                     dtype=pose.dtype, device=pose.device)
    M[..., 0, 0] = c
    M[..., 0, 1] = -s
    M[..., 0, 2] = x
    M[..., 1, 0] = s
    M[..., 1, 1] = c
    M[..., 1, 2] = y
    M[..., 2, 2] = 1.0

    return M.squeeze(0) if squeezed else M


def mat_to_pose_vec(M: torch.Tensor) -> torch.Tensor:
    """
    Extract [x, y, θ] from a 3×3 homogeneous SE(2) matrix.
    θ is wrapped to [-π, π].

    Parameters
    ----------
    M : (..., 3, 3)

    Returns
    -------
    pose : (..., 3)
    """
    x = M[..., 0, 2]
    y = M[..., 1, 2]
    th = torch.atan2(M[..., 1, 0], M[..., 0, 0])
    th = (th + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.stack([x, y, th], dim=-1)


# ---------------------------------------------------------------------------
# Differentiable depth renderer
# ---------------------------------------------------------------------------

class DifferentiableDepthRenderer:
    """
    Look up depths from a padded DeSDF using differentiable trilinear
    interpolation (bilinear in x/y, linear in orientation).

    The DeSDF is padded once along the orientation axis so that slicing
    never wraps.
    """

    def __init__(self, desdf: torch.Tensor, resolution: float, V: int):
        """
        Parameters
        ----------
        desdf : (H, W, O)  – DeSDF tensor on the target device.
        resolution : float  – spatial resolution [m/cell].
        V : int  – number of ray directions to extract per pose.
        """
        self.resolution = resolution
        self.H, self.W, self.O = desdf.shape
        self.device = desdf.device
        self.V = V

        # Pad orientation axis once
        pad_front = V // 2
        pad_back = V - pad_front
        self.desdf_pad = F.pad(desdf, (pad_front, pad_back), mode="circular")

        # Base orientation offset indices  (V,)
        self._orn_base = torch.arange(V, device=self.device, dtype=torch.long)

    # ---- batch interface ---------------------------------------------------

    def render_batch(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Render depths for multiple poses simultaneously.

        Parameters
        ----------
        poses : (T, 3)  –  [x, y, θ] in metres / radians.

        Returns
        -------
        depths : (T, V)
        """
        T = poses.shape[0]
        x, y, theta = poses[:, 0], poses[:, 1], poses[:, 2]

        # Orientation interpolation weights  (T,)
        theta_idx = (theta % (2 * torch.pi)) / (2 * torch.pi) * self.O
        idx0 = torch.floor(theta_idx).long()
        w_orn = (theta_idx - idx0.float()).unsqueeze(1)        # (T, 1)

        # Orientation index grids  (T, V)
        orn_lo = (self._orn_base.unsqueeze(0) + idx0.unsqueeze(1)).clamp(
            0, self.desdf_pad.shape[2] - 1)
        orn_hi = (orn_lo + 1).clamp(0, self.desdf_pad.shape[2] - 1)

        # Spatial interpolation coordinates  (T,)
        fx = x / self.resolution
        fy = y / self.resolution
        x0 = torch.floor(fx).long().clamp(0, self.W - 1)
        x1 = (x0 + 1).clamp(0, self.W - 1)
        y0 = torch.floor(fy).long().clamp(0, self.H - 1)
        y1 = (y0 + 1).clamp(0, self.H - 1)
        wx = (fx - x0.float()).unsqueeze(1)                     # (T, 1)
        wy = (fy - y0.float()).unsqueeze(1)                     # (T, 1)

        # Expand spatial indices to (T, V) for advanced indexing
        y0e = y0.unsqueeze(1).expand(T, self.V)
        y1e = y1.unsqueeze(1).expand(T, self.V)
        x0e = x0.unsqueeze(1).expand(T, self.V)
        x1e = x1.unsqueeze(1).expand(T, self.V)

        def _sample(yi, xi):
            lo = self.desdf_pad[yi, xi, orn_lo]
            hi = self.desdf_pad[yi, xi, orn_hi]
            return (1 - w_orn) * lo + w_orn * hi

        d00 = _sample(y0e, x0e)
        d01 = _sample(y0e, x1e)
        d10 = _sample(y1e, x0e)
        d11 = _sample(y1e, x1e)

        depths = ((1 - wx) * (1 - wy) * d00
                  + wx * (1 - wy) * d01
                  + (1 - wx) * wy * d10
                  + wx * wy * d11)
        return depths  # (T, V)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def _apply_rigid_transform(
    poses: torch.Tensor,
    delta_xy: torch.Tensor,
    delta_th: torch.Tensor,
) -> torch.Tensor:
    """Apply a global SE(2) correction to a batch of poses.

    Parameters
    ----------
    poses : (T, 3)
    delta_xy : (2,)  –  translation correction.
    delta_th : (1,)  –  rotation correction.

    Returns
    -------
    transformed : (T, 3)
    """
    dx, dy = delta_xy
    dtheta = delta_th[0]
    c, s = torch.cos(dtheta), torch.sin(dtheta)

    R = torch.stack([torch.stack([c, -s]),
                     torch.stack([s,  c])])          # (2, 2)
    xy = poses[:, :2] @ R.T + torch.stack([dx, dy])
    theta = poses[:, 2] + dtheta
    return torch.cat([xy, theta.unsqueeze(-1)], dim=-1)


def refine_sequence(
    init_poses: torch.Tensor,
    pred_depths: torch.Tensor,
    pred_uncertainties: torch.Tensor,
    desdf: torch.Tensor,
    resolution: float = 0.1,
    orn_slice: int = 72,
    iters: int = 100,
    lr_xy: float = 1e-3,
    lr_theta: float = 1e-5,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Refine a sequence of poses by optimising a single global SE(2)
    correction that minimises depth-matching loss (Laplace likelihood).

    Parameters
    ----------
    init_poses : (T, 3)  –  initial [x, y, θ] in metres / radians.
    pred_depths : (T, V)  –  predicted depth per ray.
    pred_uncertainties : (T, V)  –  Laplace scale per ray.
    desdf : (H, W, O)  –  floor-plan DeSDF tensor.
    resolution : float  –  DeSDF spatial resolution [m/cell].
    orn_slice : int  –  number of orientation bins.
    iters : int  –  optimisation steps.
    lr_xy, lr_theta : float  –  learning rates.
    verbose : bool  –  print loss every 20 iterations.

    Returns
    -------
    refined : (T, 3)  –  refined poses (detached).
    """
    torch.manual_seed(42)

    device = init_poses.device
    V = pred_depths.shape[1]

    renderer = DifferentiableDepthRenderer(desdf, resolution, V)

    delta_xy = torch.zeros(2, requires_grad=True, device=device)
    delta_th = torch.zeros(1, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([
        {"params": [delta_xy], "lr": lr_xy},
        {"params": [delta_th], "lr": lr_theta},
    ])

    b_clamped = torch.clamp(pred_uncertainties, min=1e-6)

    for it in range(iters):
        optimizer.zero_grad()
        poses_t = _apply_rigid_transform(init_poses, delta_xy, delta_th)
        d_rendered = renderer.render_batch(poses_t)
        loss = torch.sum(torch.abs(pred_depths - d_rendered) / b_clamped)
        loss.backward()
        optimizer.step()

        if verbose and it % 20 == 0:
            delta_np = torch.cat([delta_xy, delta_th]).detach().cpu().numpy()
            print(f"  [Iter {it:3d}] loss={loss.item():.4f}  Δ={delta_np}")

    refined = _apply_rigid_transform(init_poses, delta_xy, delta_th).detach()
    if verbose:
        delta_np = torch.cat([delta_xy, delta_th]).detach().cpu().numpy()
        print(f"  Final delta: {delta_np}")
    return refined


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def postprocess_trajectory(
    pose_ls: list,
    pred_depths_ls: list,
    pred_uncert_ls: list,
    transition_ls: list,
    desdf: torch.Tensor,
    desdf_resolution: float = 0.1,
    orn_slice: int = 72,
    T_refine: int = 10,
    iters: int = 100,
    lr_xy: float = 1e-3,
    lr_theta: float = 1e-5,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Refine the last ``T_refine`` poses of a trajectory.

    1. Take the histogram-filter argmax at the last timestep (in DeSDF
       grid indices + radians) and walk backwards through the relative
       transitions to build an initial pose chain (in metres).
    2. Run :func:`refine_sequence` to optimise a global SE(2) correction.
    3. Convert the refined poses back to DeSDF grid indices.

    Parameters
    ----------
    pose_ls : list of (3,) np.ndarray
        Per-timestep argmax poses from the Bayes filter, each
        ``[grid_x, grid_y, θ_rad]`` in DeSDF cell indices.
    pred_depths_ls : list of (V,) np.ndarray
        Predicted depth vectors per timestep.
    pred_uncert_ls : list of (V,) np.ndarray
        Predicted Laplace scale vectors per timestep.
    transition_ls : list of (3,) np.ndarray
        Relative transitions ``get_rel_pose(current, next)`` for
        timesteps 0..T-2  (in metres / radians).
    desdf : (H, W, O) torch.Tensor
        DeSDF on the target device.
    desdf_resolution : float
    orn_slice : int
    T_refine : int
        Number of trailing timesteps to refine.
    iters, lr_xy, lr_theta : optimiser settings.
    verbose : bool

    Returns
    -------
    refined_poses : (T_refine, 3) torch.Tensor
        Refined poses in *DeSDF grid indices* (x, y as float, θ as
        orientation-bin index), clamped to valid ranges.
    """
    device = desdf.device
    H, W, O = desdf.shape
    traj_len = len(pose_ls)
    start_idx = max(0, traj_len - T_refine)
    last_idx = traj_len - 1

    # --- 1. Build initial pose chain in metres ---
    last_pose = pose_ls[last_idx]
    s_next = torch.tensor([
        last_pose[0] * desdf_resolution,
        last_pose[1] * desdf_resolution,
        last_pose[2],
    ], dtype=torch.float32, device=device)

    init_list = [s_next]

    for t in range(last_idx - 1, start_idx - 1, -1):
        trans = torch.tensor(
            transition_ls[t], dtype=torch.float32, device=device
        )
        T_rel = pose_vec_to_mat(trans)
        T_rel_inv = torch.inverse(T_rel)
        T_next = pose_vec_to_mat(init_list[0])
        T_prev = T_next @ T_rel_inv
        init_list.insert(0, mat_to_pose_vec(T_prev))

    init_poses = torch.stack(init_list, dim=0)  # (T_refine, 3)

    # --- 2. Prepare depth / uncertainty tensors ---
    depths_t = torch.stack([
        torch.as_tensor(d, dtype=torch.float32, device=device)
        for d in pred_depths_ls[start_idx:traj_len]
    ], dim=0)
    uncert_t = torch.stack([
        torch.as_tensor(u, dtype=torch.float32, device=device)
        for u in pred_uncert_ls[start_idx:traj_len]
    ], dim=0)

    # --- 3. Optimise ---
    refined = refine_sequence(
        init_poses, depths_t, uncert_t, desdf,
        resolution=desdf_resolution,
        orn_slice=orn_slice,
        iters=iters,
        lr_xy=lr_xy,
        lr_theta=lr_theta,
        verbose=verbose,
    )

    # --- 4. Convert back to DeSDF grid indices ---
    refined[:, 0] = torch.clamp(
        torch.round(refined[:, 0] / desdf_resolution).long(), 0, W - 1
    ).float()
    refined[:, 1] = torch.clamp(
        torch.round(refined[:, 1] / desdf_resolution).long(), 0, H - 1
    ).float()
    refined[:, 2] = (
        torch.round(
            (refined[:, 2] % (2 * torch.pi)) / (2 * torch.pi) * orn_slice
        ).long() % O
    ).float()

    return refined
