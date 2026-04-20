# Adapted from F3Loc (Chen et al., CVPR 2024; MIT License).
# https://github.com/felix-ch/f3loc

"""
Localization utilities for UnLoc filtering on a directional ESDF (DeSDF).

Core functions:
    - localize_noflip_uncertainty: compute observation likelihood using
      Laplace-scaled ray matching against the DeSDF (HGE datasets).
    - localize_uncertainty: same, but with ray-flip for Gibson convention.
    - get_ray_from_depth_uncertainty: convert 1-D depth + uncertainty
      predictions into angular ray representations.
    - transit: propagate the belief (probability volume) forward in time
      using a Gaussian motion model.
    - get_filters: build translational + rotational convolution kernels
      for the motion model.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------------------------
# Observation model — shared helpers
# ---------------------------------------------------------------------------------------------

def _extract_map_estimate(prob_dist, orientations, orn_slice):
    """Extract the MAP (x, y, theta) from a max-pooled probability map."""
    prob_dist_cpu = prob_dist.cpu().numpy()
    pred_y, pred_x = np.unravel_index(
        np.argmax(prob_dist_cpu), prob_dist_cpu.shape
    )
    device = prob_dist.device
    pred_x = torch.tensor(pred_x, device=device)
    pred_y = torch.tensor(pred_y, device=device)
    orn = orientations[pred_y, pred_x]
    orn = orn / orn_slice * 2 * torch.pi
    pred = torch.cat((pred_x.unsqueeze(0), pred_y.unsqueeze(0), orn.unsqueeze(0)))
    return pred


def _format_outputs(prob_vol, prob_dist, orientations, pred, return_np):
    """Return results as numpy arrays or CPU float32 tensors."""
    if return_np:
        return (
            prob_vol.detach().cpu().numpy(),
            prob_dist.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
        )
    else:
        return (
            prob_vol.to(torch.float32).detach().cpu(),
            prob_dist.to(torch.float32).detach().cpu(),
            orientations.to(torch.float32).detach().cpu(),
            pred.to(torch.float32).detach().cpu(),
        )


# ---------------------------------------------------------------------------------------------
# Observation model — HGE (no flip)
# ---------------------------------------------------------------------------------------------

def localize_noflip_uncertainty(
    desdf: torch.Tensor,
    rays: torch.Tensor,
    scales_rays: torch.Tensor,
    orn_slice: int = 36,
    return_np: bool = True,
    lambd: float = 40,
) -> tuple[torch.Tensor, ...]:
    """
    Compute the observation likelihood over the DeSDF using Laplace-scaled
    ray matching (HGE convention — rays are NOT flipped).

    The likelihood at each (x, y, orientation) cell is proportional to
        exp( -sum_v clamp(|desdf_ray_v - pred_ray_v| / scale_v, max=5)
             / V * 11 / lambd )

    Parameters
    ----------
    desdf : torch.Tensor, shape (H, W, O)
        Directional ESDF. O orientation bins, counter-clockwise.
    rays : torch.Tensor, shape (V,)
        Predicted depth rays (left to right, no flip).
    scales_rays : torch.Tensor, shape (V,)
        Predicted uncertainty (Laplace scale) per ray.
    orn_slice : int
        Number of orientation bins in the DeSDF.
    return_np : bool
        If True, return numpy arrays; otherwise return CPU float32 tensors.
    lambd : float
        Temperature for the exponential likelihood.

    Returns
    -------
    prob_vol : (H, W, O) — full probability volume (normalized).
    prob_dist : (H, W) — max-pooled probability over orientation.
    orientations : (H, W) — orientation index with highest likelihood per cell.
    pred : (3,) — [x, y, theta_rad] of the MAP estimate.
    """
    O = desdf.shape[2]
    V = rays.shape[0]

    rays = rays.reshape((1, 1, -1))
    scales_rays = scales_rays.reshape((1, 1, -1))

    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    # Laplace-scaled, clamped L1 cost
    prob_vol = torch.stack(
        [
            -torch.sum(
                torch.clamp(
                    torch.abs(
                        (pad_desdf[:, :, i : i + V] - rays) / scales_rays
                    ),
                    max=5,
                ),
                dim=2,
            )
            / V
            * 11
            for i in range(O)
        ],
        dim=2,
    )

    del pad_desdf
    prob_vol = torch.exp(prob_vol / lambd)
    prob_vol = prob_vol / torch.sum(prob_vol)

    prob_dist, orientations = torch.max(prob_vol, dim=2)
    pred = _extract_map_estimate(prob_dist, orientations, orn_slice)

    return _format_outputs(prob_vol, prob_dist, orientations, pred, return_np)


# ---------------------------------------------------------------------------------------------
# Observation model — Gibson (flip)
# ---------------------------------------------------------------------------------------------

def localize_uncertainty(
    desdf: torch.Tensor,
    rays: torch.Tensor,
    scales_rays: torch.Tensor,
    orn_slice: int = 36,
    return_np: bool = True,
    lambd: float = 40,
) -> tuple[torch.Tensor, ...]:
    """
    Compute the observation likelihood over the DeSDF using Laplace-scaled
    ray matching (Gibson convention — rays ARE flipped to match the DeSDF's
    counter-clockwise orientation convention).

    The likelihood at each (x, y, orientation) cell is proportional to
        exp( -||( desdf_rays - pred_rays ) / scales||_1  /  lambd )

    Parameters
    ----------
    desdf : torch.Tensor, shape (H, W, O)
        Directional ESDF. O orientation bins, counter-clockwise.
    rays : torch.Tensor, shape (V,)
        Predicted depth rays (left to right, clockwise — will be flipped).
    scales_rays : torch.Tensor, shape (V,)
        Predicted uncertainty (Laplace scale) per ray (flipped alongside rays).
    orn_slice : int
        Number of orientation bins in the DeSDF.
    return_np : bool
        If True, return numpy arrays; otherwise return CPU float32 tensors.
    lambd : float
        Temperature for the exponential likelihood.

    Returns
    -------
    prob_vol : (H, W, O) — full probability volume (normalized).
    prob_dist : (H, W) — max-pooled probability over orientation.
    orientations : (H, W) — orientation index with highest likelihood per cell.
    pred : (3,) — [x, y, theta_rad] of the MAP estimate.
    """
    # Flip rays to match the DeSDF's counter-clockwise convention
    rays = torch.flip(rays, [0])
    scales_rays = torch.flip(scales_rays, [0])

    O = desdf.shape[2]
    V = rays.shape[0]

    rays = rays.reshape((1, 1, -1))
    scales_rays = scales_rays.reshape((1, 1, -1))

    pad_front = V // 2
    pad_back = V - pad_front
    pad_desdf = F.pad(desdf, [pad_front, pad_back], mode="circular")

    # Laplace-scaled L1 cost (no clamping, matching Gibson original)
    prob_vol = torch.stack(
        [
            -torch.norm(
                (pad_desdf[:, :, i : i + V] - rays) / scales_rays,
                p=1.0,
                dim=2,
            )
            for i in range(O)
        ],
        dim=2,
    )

    prob_vol = torch.exp(prob_vol / lambd)
    prob_vol = prob_vol / torch.sum(prob_vol)

    prob_dist, orientations = torch.max(prob_vol, dim=2)
    pred = _extract_map_estimate(prob_dist, orientations, orn_slice)

    return _format_outputs(prob_vol, prob_dist, orientations, pred, return_np)


# ---------------------------------------------------------------------------------------------
# Depth → ray conversion
# ---------------------------------------------------------------------------------------------

def get_ray_from_depth_uncertainty(
    d: np.ndarray,
    scales: np.ndarray,
    V: int = 11,
    dv: float = 10,
    a0: float = None,
    F_W: float = 3 / 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1-D depth profile and its per-pixel uncertainty into a set of
    angular rays (and corresponding scale values) by interpolation.

    Parameters
    ----------
    d : np.ndarray, shape (W,)
        Predicted metric depths across the image width.
    scales : np.ndarray, shape (W,)
        Predicted Laplace scale (uncertainty) per pixel.
    V : int
        Number of output rays.
    dv : float
        Angular spacing between neighbouring rays [degrees].
    a0 : float or None
        Camera principal-point x-coordinate. If None, assumed to be at the
        image center.
    F_W : float
        Ratio of focal length to image width.

    Returns
    -------
    rays : np.ndarray, shape (V,)
        Depth along each ray direction (corrected for off-axis angle).
    scales_rays : np.ndarray, shape (V,)
        Interpolated uncertainty per ray.
    """
    W = d.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180 * np.pi

    if a0 is None:
        w = np.tan(angles) * W * F_W + (W - 1) / 2
    else:
        w = np.tan(angles) * W * F_W + a0

    interp_d = griddata(np.arange(W).reshape(-1, 1), d, w, method="linear")
    rays = interp_d / np.cos(angles)
    scales_rays = griddata(
        np.arange(W).reshape(-1, 1), scales, w, method="linear"
    )

    return rays, scales_rays


# ---------------------------------------------------------------------------------------------
# Motion model (transition / propagation)
# ---------------------------------------------------------------------------------------------

def transit(
    prob_vol: torch.Tensor,
    transition: torch.Tensor,
    sig_o: float = 0.1,
    sig_x: float = 0.05,
    sig_y: float = 0.05,
    tsize: int = 5,
    rsize: int = 5,
    resolution: float = 0.1,
) -> torch.Tensor:
    """
    Propagate the belief through the motion model using Gaussian convolution.

    The translational component is a 2-D convolution (one per orientation
    bin, grouped), and the rotational component is a 1-D circular convolution.

    Parameters
    ----------
    prob_vol : torch.Tensor, shape (H, W, O)
        Current belief (probability volume).
    transition : torch.Tensor, shape (3,)
        Ego-motion [dx, dy, dtheta] in the local frame (metric units + rad).
    sig_o : float
        Std-dev for rotation [rad].
    sig_x, sig_y : float
        Std-dev for x/y translation [m].
    tsize : int
        Translational kernel size (pixels).
    rsize : int
        Rotational kernel size (orientation bins).
    resolution : float
        Grid resolution [m/pixel].

    Returns
    -------
    prob_vol : torch.Tensor, shape (H, W, O)
        Propagated (predicted) belief, normalized.
    """
    H, W, O = list(prob_vol.shape)

    filters_trans, filter_rot = get_filters(
        transition, O,
        sig_o=sig_o, sig_x=sig_x, sig_y=sig_y,
        tsize=tsize, rsize=rsize, resolution=resolution,
    )

    device = prob_vol.device
    filters_trans = filters_trans.to(device)
    filter_rot = filter_rot.to(device)

    # --- Translational convolution (grouped, one kernel per orientation) ---
    prob_vol = prob_vol.permute((2, 0, 1))  # (O, H, W)
    prob_vol = F.conv2d(
        prob_vol,
        weight=filters_trans.unsqueeze(1).flip([-2, -1]),
        bias=None,
        groups=O,
        padding="same",
    )  # (O, H, W)

    # --- Rotational convolution (circular 1-D) ---
    prob_vol = prob_vol.permute((1, 2, 0))  # (H, W, O)
    prob_vol = prob_vol.reshape((H * W, 1, O))
    prob_vol = F.pad(
        prob_vol,
        pad=[int((rsize - 1) / 2), int((rsize - 1) / 2)],
        mode="circular",
    )
    prob_vol = F.conv1d(
        prob_vol,
        weight=filter_rot.flip(dims=[-1]).unsqueeze(0).unsqueeze(0),
        bias=None,
    )

    prob_vol = prob_vol.reshape([H, W, O])

    # Normalize
    epsilon = 1e-7
    prob_vol = prob_vol / (prob_vol.sum() + epsilon)

    return prob_vol


def get_filters(
    transition: torch.Tensor,
    O: int = 36,
    sig_o: float = 0.1,
    sig_x: float = 0.05,
    sig_y: float = 0.05,
    tsize: int = 5,
    rsize: int = 5,
    resolution: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build Gaussian translational and rotational kernels for the motion model.

    For each of the O orientations, the translational filter center is
    rotated according to the ego-motion direction at that orientation.

    Parameters
    ----------
    transition : torch.Tensor, shape (3,)
        Ego-motion [dx, dy, dtheta].
    O : int
        Number of orientation bins.
    sig_o, sig_x, sig_y : float
        Standard deviations for rotation and translation.
    tsize, rsize : int
        Kernel sizes for translation and rotation.
    resolution : float
        Grid resolution [m/pixel].

    Returns
    -------
    filters_trans : torch.Tensor, shape (O, tsize, tsize)
    filter_rot : torch.Tensor, shape (rsize,)
    """
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
        indexing="ij",
    )
    grid_x = grid_x * resolution
    grid_y = grid_y * resolution

    # Rotational center (same for all orientations)
    center_o = transition[-1]

    # Translational centers depend on orientation
    orns = (
        torch.arange(0, O, dtype=torch.float32, device=transition.device)
        / O * 2 * torch.pi
    )
    c_th = torch.cos(orns).reshape((O, 1, 1))
    s_th = torch.sin(orns).reshape((O, 1, 1))
    center_x = transition[0] * c_th - transition[1] * s_th
    center_y = transition[0] * s_th + transition[1] * c_th

    # Gaussian translational filters
    epsilon = 1e-10
    filters_trans = torch.exp(
        -((grid_x - center_x) ** 2) / (sig_x ** 2)
        - ((grid_y - center_y) ** 2) / (sig_y ** 2)
    )
    filters_trans = filters_trans / (
        filters_trans.sum(-1).sum(-1).reshape((O, 1, 1)) + epsilon
    )

    # Gaussian rotational filter
    grid_o = (
        torch.arange(-(rsize - 1) / 2, (rsize + 1) / 2, 1, device=transition.device)
        / O * 2 * torch.pi
    )
    filter_rot = torch.exp(-((grid_o - center_o) ** 2) / (sig_o ** 2))

    return filters_trans, filter_rot
