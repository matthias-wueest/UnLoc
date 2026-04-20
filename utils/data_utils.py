# Adapted from F3Loc (Chen et al., CVPR 2024; MIT License).
# https://github.com/felix-ch/f3loc

"""
Dataset classes for floorplan localization.

Provides PyTorch ``Dataset`` wrappers for two dataset variants (Gibson,
LaMAR HGE) in two granularities:
  - Trajectory datasets: return L consecutive frames per sample (used by
    ``evaluate.py`` for sequential localization).
  - Frame datasets: return a single frame per sample (used by ``train.py``
    for training the depth prediction head).
"""

import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.geometry import gravity_align


# ===========================================================================
# Trajectory datasets (evaluation — sequences of L frames per sample)
# ===========================================================================

class GibsonTrajectoryDataset(Dataset):
    """
    Trajectory dataset for the Gibson setting with DepthAnything-V2
    compatible preprocessing.

    Each sample is a trajectory of L consecutive frames.  Images are
    normalised with ImageNet statistics and returned in (L, 3, H, W)
    layout — the DepthAnything encoder is applied externally (in the
    evaluation loop or model), not inside this dataset.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root for the active scene split.
    scene_names : list of str
        Scene identifiers belonging to this split.
    L : int
        Trajectory length (number of frames per sample).
    depth_dir : str or None
        Directory containing depth files. Defaults to ``dataset_dir``.
    depth_suffix : str
        Filename stem for the depth text file (e.g. ``"depth40"``).
    add_rp : bool
        If True, apply random virtual roll/pitch augmentation.
    roll, pitch : float
        Maximum roll/pitch perturbation (radians) when ``add_rp=True``.
    without_depth : bool
        If True, skip loading ground-truth depths.
    """

    def __init__(
        self,
        dataset_dir: str,
        scene_names: list[str],
        L: int,
        depth_dir: str | None = None,
        depth_suffix: str = "depth40",
        add_rp: bool = False,
        roll: float = 0,
        pitch: float = 0,
        without_depth: bool = False,
    ) -> None:

        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.without_depth = without_depth

        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []

        self._load_metadata()
        self.N = self.scene_start_idx[-1]

    def __len__(self):
        return self.N

    # ------------------------------------------------------------------------------------
    # Internal: load depths and poses for all scenes
    # ------------------------------------------------------------------------------------

    def _load_metadata(self):
        self.scene_start_idx.append(0)
        start_idx = 0

        for scene in self.scene_names:
            # --- Depths ---
            if not self.without_depth:
                depth_base = self.depth_dir if self.depth_dir else self.dataset_dir
                depth_file = os.path.join(
                    depth_base, scene, self.depth_suffix + ".txt"
                )
                with open(depth_file, "r") as f:
                    depths_txt = [line.strip() for line in f.readlines()]

            # --- Poses ---
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            # Trim to whole trajectories of length L
            traj_len = len(poses_txt)
            traj_len -= traj_len % self.L

            scene_depths = []
            scene_poses = []
            for state_id in range(traj_len):
                if not self.without_depth:
                    depth = np.array(
                        depths_txt[state_id].split(" "), dtype=np.float32
                    )
                    scene_depths.append(depth)

                pose = np.array(
                    poses_txt[state_id].split(" "), dtype=np.float32
                )
                scene_poses.append(pose)

            start_idx += traj_len // self.L
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)

    # ------------------------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        """
        Returns
        -------
        data_dict : dict
            imgs : np.ndarray, (L, 3, H, W) float32 — normalised (and
                   optionally masked if ``add_rp=True``).
            masks : np.ndarray, (L, H, W) uint8 — only present when
                    ``add_rp=True``.
            poses : list of np.ndarray — raw 3-DoF poses per frame.
            gt_depth : list of np.ndarray — (only if ``without_depth=False``).
        """
        # Identify scene and local index
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        sl = slice(idx_within_scene * self.L,
                   idx_within_scene * self.L + self.L)

        data_dict = {}
        if not self.without_depth:
            data_dict["gt_depth"] = self.gt_depth[scene_idx][sl]

        data_dict["poses"] = self.gt_pose[scene_idx][sl]

        # --- Load images ---
        imgs = []
        for l in range(self.L):
            image_path = os.path.join(
                self.dataset_dir, scene_name, "rgb",
                f"{idx_within_scene * self.L + l:05d}.png",
            )
            img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
            imgs.append(img_l)
        imgs = np.stack(imgs, axis=0).astype(np.float32)  # (L, H, W, 3)

        # --- Optional roll/pitch augmentation ---
        if self.add_rp:
            masks = []
            for l in range(self.L):
                r = (np.random.random() - 0.5) * 2 * self.roll
                p = (np.random.random() - 0.5) * 2 * self.pitch
                mask = np.ones(imgs[l].shape[:2])
                mask = gravity_align(mask, r, p, mode=1)
                mask[mask < 1] = 0
                masks.append(mask.astype(np.uint8))
            masks = np.stack(masks, axis=0)  # (L, H, W)
            data_dict["masks"] = masks

        # --- Normalise (ImageNet stats) and apply mask ---
        for l in range(self.L):
            imgs[l] = cv2.cvtColor(imgs[l], cv2.COLOR_BGR2RGB) / 255.0
            imgs[l] -= (0.485, 0.456, 0.406)
            imgs[l] /= (0.229, 0.224, 0.225)
            if self.add_rp:
                imgs[l][masks[l] == 0, :] = 0

        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)  # (L, 3, H, W)
        data_dict["imgs"] = imgs

        return data_dict


class LaMARHGETrajectoryDataset(Dataset):
    """
    Trajectory dataset for the HGE customized setting
    with DepthAnything-V2 compatible preprocessing.

    Each sample is a trajectory of L consecutive frames, gravity-aligned
    using the recorded Euler angles and the known camera intrinsics.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root for the active scene split.
    scene_names : list of str
        Scene identifiers belonging to this split.
    L : int
        Trajectory length (number of frames per sample).
    depth_dir : str or None
        Directory containing depth files. Defaults to ``dataset_dir``.
    depth_suffix : str
        Filename stem for the depth text file (e.g. ``"depth90"``).
    add_rp : bool
        (Unused, kept for interface compatibility.)
    roll, pitch : float
        (Unused, kept for interface compatibility.)
    without_depth : bool
        If True, skip loading ground-truth depths.
    """

    # Camera intrinsics for the HGE capture rig
    K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]], dtype=np.float32)

    def __init__(
        self,
        dataset_dir: str,
        scene_names: list[str],
        L: int,
        depth_dir: str | None = None,
        depth_suffix: str = "depth90",
        add_rp: bool = False,
        roll: float = 0,
        pitch: float = 0,
        without_depth: bool = False,
    ) -> None:

        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.depth_dir = depth_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch
        self.without_depth = without_depth

        self.scene_start_idx = []
        self.gt_depth = []
        self.gt_pose = []
        self.gt_euler_angles = []

        self._load_metadata()
        self.N = self.scene_start_idx[-1]

    def __len__(self):
        return self.N

    # ------------------------------------------------------------------------------------
    # Internal: load depths, poses, Euler angles for all scenes
    # ------------------------------------------------------------------------------------

    def _load_metadata(self):
        self.scene_start_idx.append(0)
        start_idx = 0

        for scene in self.scene_names:
            # --- Depths ---
            if not self.without_depth:
                depth_base = self.depth_dir if self.depth_dir else self.dataset_dir
                depth_file = os.path.join(
                    depth_base, scene, self.depth_suffix + ".txt"
                )
                with open(depth_file, "r") as f:
                    depths_txt = [line.strip() for line in f.readlines()]

            # --- Poses ---
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            # --- Euler angles ---
            euler_file = os.path.join(self.dataset_dir, scene, "euler_angles.txt")
            with open(euler_file, "r") as f:
                euler_txt = [line.strip() for line in f.readlines()]

            # Trim to whole trajectories of length L
            traj_len = len(poses_txt)
            traj_len -= traj_len % self.L

            scene_depths = []
            scene_poses = []
            scene_eulers = []
            for state_id in range(traj_len):
                if not self.without_depth:
                    depth = np.array(
                        depths_txt[state_id].split(" "), dtype=np.float32
                    )
                    scene_depths.append(depth)

                pose = np.array(
                    poses_txt[state_id].split(" "), dtype=np.float32
                )
                scene_poses.append(pose)

                euler = np.array(
                    euler_txt[state_id].split(" "), dtype=np.float32
                )
                scene_eulers.append(euler)

            start_idx += traj_len // self.L
            self.scene_start_idx.append(start_idx)
            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)
            self.gt_euler_angles.append(scene_eulers)

    # ------------------------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        """
        Returns
        -------
        data_dict : dict
            imgs : np.ndarray, (L, 3, H, W) float32 — normalised, masked.
            masks : np.ndarray, (L, H, W) uint8 — gravity-alignment mask.
            poses : list of np.ndarray — raw 3-DoF poses per frame.
            euler_angles : list of np.ndarray — roll/pitch/yaw per frame.
            gt_depth : list of np.ndarray — (only if ``without_depth=False``).
        """
        # Identify scene and local index
        scene_idx = np.sum(idx >= np.array(self.scene_start_idx)) - 1
        scene_name = self.scene_names[scene_idx]
        idx_within_scene = idx - self.scene_start_idx[scene_idx]

        sl = slice(idx_within_scene * self.L,
                   idx_within_scene * self.L + self.L)

        data_dict = {}
        if not self.without_depth:
            data_dict["gt_depth"] = self.gt_depth[scene_idx][sl]

        data_dict["poses"] = self.gt_pose[scene_idx][sl]

        ref_euler = self.gt_euler_angles[scene_idx][sl]
        data_dict["euler_angles"] = ref_euler

        # --- Load and gravity-align images ---
        imgs = []
        for l in range(self.L):
            image_path = os.path.join(
                self.dataset_dir, scene_name, "rgb",
                f"{idx_within_scene * self.L + l:05d}-0.jpg",
            )
            img_l = cv2.imread(image_path, cv2.IMREAD_COLOR)
            imgs.append(img_l)
        imgs = np.stack(imgs, axis=0).astype(np.float32)  # (L, H, W, 3)

        masks = []
        for l in range(self.L):
            roll = ref_euler[l][0]
            pitch = ref_euler[l][1]
            # Gravity-align image
            imgs[l] = gravity_align(
                imgs[l], r=pitch, p=-(roll + np.pi / 2), mode=1, K=self.K
            )
            # Build validity mask from the alignment warp
            mask = np.ones(imgs[l].shape[:2])
            mask = gravity_align(mask, r=pitch, p=-(roll + np.pi / 2), mode=1, K=self.K)
            mask[mask < 1] = 0
            masks.append(mask.astype(np.uint8))
        masks = np.stack(masks, axis=0)  # (L, H, W)
        data_dict["masks"] = masks

        # Normalise (ImageNet stats) and apply mask
        for l in range(self.L):
            imgs[l] = cv2.cvtColor(imgs[l], cv2.COLOR_BGR2RGB) / 255.0
            imgs[l] -= (0.485, 0.456, 0.406)
            imgs[l] /= (0.229, 0.224, 0.225)
            imgs[l][masks[l] == 0, :] = 0

        imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)  # (L, 3, H, W)
        data_dict["imgs"] = imgs

        return data_dict


# ===========================================================================
# Frame datasets (training — one frame per sample)
# ===========================================================================

class LaMARHGEFrameDataset(Dataset):
    """
    Single-frame dataset for training the depth prediction head.

    Each sample is one gravity-aligned image together with its ground-truth
    depth vector, mask, and Euler angles.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root for the active scene split.
    scene_names : list of str
        Scene identifiers belonging to this split.
    depth_dir : str or None
        Directory containing depth files. Defaults to ``dataset_dir``.
    depth_suffix : str
        Filename stem for the depth text file (e.g. ``"depth90"``).
    """

    # Camera intrinsics for the HGE capture rig
    K = np.array([[1596, 0, 960], [0, 1596, 720], [0, 0, 1]], dtype=np.float32)

    def __init__(
        self,
        dataset_dir: str,
        scene_names: list[str],
        depth_dir: str | None = None,
        depth_suffix: str = "depth90",
    ) -> None:

        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.depth_dir = depth_dir or dataset_dir
        self.depth_suffix = depth_suffix

        # Flat index → (scene_idx, frame_idx) mapping
        self.samples = []   # list of (scene_idx, frame_idx)
        self.gt_depth = []  # per-scene list of depth arrays
        self.gt_pose = []   # per-scene list of pose arrays
        self.gt_euler = []  # per-scene list of euler arrays

        self._load_metadata()

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------------------------

    def _load_metadata(self):
        for scene_idx, scene in enumerate(self.scene_names):
            # Depths
            depth_file = os.path.join(
                self.depth_dir, scene, self.depth_suffix + ".txt"
            )
            with open(depth_file, "r") as f:
                depths_txt = [line.strip() for line in f.readlines()]

            # Poses
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            # Euler angles
            euler_file = os.path.join(self.dataset_dir, scene, "euler_angles.txt")
            with open(euler_file, "r") as f:
                euler_txt = [line.strip() for line in f.readlines()]

            n_frames = len(poses_txt)
            scene_depths = []
            scene_poses = []
            scene_eulers = []

            for i in range(n_frames):
                depth = np.array(depths_txt[i].split(" "), dtype=np.float32)
                scene_depths.append(depth)

                pose = np.array(poses_txt[i].split(" "), dtype=np.float32)
                scene_poses.append(pose)

                euler = np.array(euler_txt[i].split(" "), dtype=np.float32)
                scene_eulers.append(euler)

                self.samples.append((scene_idx, i))

            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)
            self.gt_euler.append(scene_eulers)

    # ------------------------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        """
        Returns
        -------
        data_dict : dict
            ref_img : np.ndarray, (3, H, W) float32 — normalised, gravity-aligned.
            ref_mask : np.ndarray, (H, W) uint8 — gravity-alignment validity mask.
            ref_depth : np.ndarray, (fW,) float32 — ground-truth depth vector.
            ref_pose : np.ndarray, (3,) float32 — pose [x, y, θ].
            euler_angles : np.ndarray, (3,) float32 — [roll, pitch, yaw].
        """
        scene_idx, frame_idx = self.samples[idx]
        scene_name = self.scene_names[scene_idx]

        data_dict = {
            "ref_depth": self.gt_depth[scene_idx][frame_idx],
            "ref_pose": self.gt_pose[scene_idx][frame_idx],
            "euler_angles": self.gt_euler[scene_idx][frame_idx],
        }

        # --- Load image ---
        image_path = os.path.join(
            self.dataset_dir, scene_name, "rgb",
            f"{frame_idx:05d}-0.jpg",
        )
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)

        # --- Gravity alignment ---
        euler = data_dict["euler_angles"]
        roll, pitch = euler[0], euler[1]
        img = gravity_align(img, r=pitch, p=-(roll + np.pi / 2), mode=1, K=self.K)

        mask = np.ones(img.shape[:2])
        mask = gravity_align(mask, r=pitch, p=-(roll + np.pi / 2), mode=1, K=self.K)
        mask[mask < 1] = 0
        ref_mask = mask.astype(np.uint8)
        data_dict["ref_mask"] = ref_mask

        # --- Normalise (ImageNet stats) and apply mask ---
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img -= (0.485, 0.456, 0.406)
        img /= (0.229, 0.224, 0.225)
        img[ref_mask == 0, :] = 0

        # (H, W, 3) → (3, H, W)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        data_dict["ref_img"] = img

        return data_dict


class GibsonFrameDataset(Dataset):
    """
    Single-frame dataset for training the depth prediction head (Gibson).

    Each sample is one image together with its ground-truth depth vector
    and pose.  Optional roll/pitch augmentation can be applied.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root for the active scene split.
    scene_names : list of str
        Scene identifiers belonging to this split.
    L : int
        Window size: one sample is drawn per (L+1)-frame window, with the
        reference frame at position L. Default: 3 (matching F³Loc, given 
        by Gibson structure).
    depth_dir : str or None
        Directory containing depth files. Defaults to ``dataset_dir``.
    depth_suffix : str
        Filename stem for the depth text file (e.g. ``"depth40"``).
    add_rp : bool
        If True, apply random virtual roll/pitch augmentation.
    roll, pitch : float
        Maximum roll/pitch perturbation (radians) when ``add_rp=True``.
    """

    def __init__(
        self,
        dataset_dir: str,
        scene_names: list[str],
        L: int = 3,
        depth_dir: str | None = None,
        depth_suffix: str = "depth40",
        add_rp: bool = False,
        roll: float = 0,
        pitch: float = 0,
    ) -> None:

        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.L = L
        self.depth_dir = depth_dir or dataset_dir
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch

        # Flat index -> (scene_idx, window_idx, ref_frame_idx) mapping
        self.samples = []
        self.gt_depth = []  # per-scene list of depth arrays
        self.gt_pose = []   # per-scene list of pose arrays

        self._load_metadata()

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------------------------

    def _load_metadata(self):
        for scene_idx, scene in enumerate(self.scene_names):
            # Depths
            depth_file = os.path.join(
                self.depth_dir, scene, self.depth_suffix + ".txt"
            )
            with open(depth_file, "r") as f:
                depths_txt = [line.strip() for line in f.readlines()]

            # Poses
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                poses_txt = [line.strip() for line in f.readlines()]

            n_frames = len(poses_txt)
            scene_depths = []
            scene_poses = []

            for i in range(n_frames):
                depth = np.array(depths_txt[i].split(" "), dtype=np.float32)
                scene_depths.append(depth)

                pose = np.array(poses_txt[i].split(" "), dtype=np.float32)
                scene_poses.append(pose)

            # One sample per window of (L+1) frames; reference = last frame
            n_windows = n_frames // (self.L + 1)
            for w in range(n_windows):
                ref_frame_idx = w * (self.L + 1) + self.L
                self.samples.append((scene_idx, w, ref_frame_idx))

            self.gt_depth.append(scene_depths)
            self.gt_pose.append(scene_poses)

    # ------------------------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        """
        Returns
        -------
        data_dict : dict
            ref_img : np.ndarray, (3, H, W) float32 — normalised and masked.
            ref_mask : np.ndarray, (H, W) uint8 — all-ones by default,
                    modified by gravity_align when ``add_rp=True``.
            ref_depth : np.ndarray, (fW,) float32 — ground-truth depth vector.
            ref_pose : np.ndarray, (3,) float32 — pose [x, y, θ].
        """
        scene_idx, window_idx, ref_frame_idx = self.samples[idx]
        scene_name = self.scene_names[scene_idx]

        data_dict = {
            "ref_depth": self.gt_depth[scene_idx][ref_frame_idx],
            "ref_pose":  self.gt_pose[scene_idx][ref_frame_idx],
        }

        # --- Load image ---
        image_path = os.path.join(
            self.dataset_dir, scene_name, "rgb",
            f"{window_idx:05d}-{self.L}.png",
        )
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(
                f"cv2.imread returned None for {image_path!r} "
                f"(exists={os.path.exists(image_path)})"
            )
        img = img.astype(np.float32)

        # --- Mask: all-ones by default, modified by optional roll/pitch aug ---
        if self.add_rp:
            r = (np.random.random() - 0.5) * 2 * self.roll
            p = (np.random.random() - 0.5) * 2 * self.pitch
            mask = np.ones(img.shape[:2])
            mask = gravity_align(mask, r, p, mode=1)
            mask[mask < 1] = 0
        else:
            mask = np.ones(img.shape[:2])
        ref_mask = mask.astype(np.uint8)
        data_dict["ref_mask"] = ref_mask

        # --- Normalise (ImageNet stats) and apply mask ---
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img -= (0.485, 0.456, 0.406)
        img /= (0.229, 0.224, 0.225)
        img[ref_mask == 0, :] = 0   # no-op when mask is all ones

        # (H, W, 3) -> (3, H, W)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        data_dict["ref_img"] = img

        return data_dict
