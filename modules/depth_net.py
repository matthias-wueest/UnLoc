# Adapted from F3Loc (Chen et al., CVPR 2024; MIT License).
# https://github.com/felix-ch/f3loc
"""
Depth prediction network: predicts per-column depth (location) and
uncertainty (Laplace scale) from a single image using a frozen
DepthAnything-V2 backbone followed by a trainable cross-attention head.
"""

import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision.transforms import Compose

# depth_anything_v2 is vendored as a git submodule. The doubled path reflects
# the upstream package layout (repo/package/module).
from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.depth_anything_v2.util.transform import Resize, PrepareForNet
from modules.network_utils import Attention, ConvBnReLU


class UnLocDepthNet(nn.Module):
    """
    Predict per-ray depth location (mu) and Laplace scale (b) from a
    single image.

    Architecture:
        image → DepthAnything-V2 features → Conv → Cross-Attention
              → FC_loc (mu) + FC_scale (softplus → b)
    """

    def __init__(self) -> None:
        super().__init__()
        self.depth_feature = UnLocDepthFeatures()

        self.fc_loc = nn.Linear(128, 1)    # location parameter (mu)
        self.fc_scale = nn.Linear(128, 1)  # scale parameter (b)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """Predict per-ray depth location (mu) and Laplace scale (b).
        Returns
        -------
        loc : (N, fW) — depth prediction (Laplace location parameter mu).
        scale : (N, fW) — depth uncertainty (Laplace scale b, positive).
        attn : attention weights from the cross-attention head.
        prob : softmax over feature channels.
        """

        x, attn = self.depth_feature(x, mask)   # (N, fW, 128)

        prob = F.softmax(x, dim=-1)             # (N, fW, D)

        loc = self.fc_loc(x).squeeze(-1)                     # (N, fW)
        scale = F.softplus(self.fc_scale(x)).squeeze(-1)     # (N, fW), > 0

        return loc, scale, attn, prob


# ---------------------------------------------------------------------------
# Feature extractor (frozen DepthAnything backbone + trainable attention)
# ---------------------------------------------------------------------------

class UnLocDepthFeatures(nn.Module):
    """
    DepthAnything-V2 encoder → 1×1 conv → cross-attention pooling.

    The DepthAnything backbone is frozen; only the conv, positional MLPs,
    and attention projections are trained.

    Image preprocessing for the DA backbone matches the paper's pipeline:
    aspect-ratio-preserving resize with `ensure_multiple_of=14`, lower-bound
    sizing at 518, and `cv2.INTER_CUBIC` interpolation.

    Parameters
    ----------
    encoder : str
        ViT variant: ``'vits'``, ``'vitb'``, or ``'vitl'``.
    da_dataset : str
        Pre-training dataset for the DA checkpoint (``'hypersim'`` / ``'vkitti'``).
    da_checkpoint_dir : str
        Directory containing DA checkpoint files.
    """

    def __init__(
        self,
        encoder: str = "vitl",
        da_dataset: str = "hypersim",
        da_checkpoint_dir: str = "depth_anything_v2/checkpoints",
    ) -> None:

        super().__init__()

        # --- DepthAnything-V2 backbone (frozen) ---
        model_configs = {
            "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        encoder_out_channels = {"vits": 384, "vitb": 768, "vitl": 1024}
        self.da_out_channels = encoder_out_channels[encoder]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.da = DepthAnythingV2(**model_configs[encoder])
        ckpt_path = os.path.join(
            da_checkpoint_dir,
            f"depth_anything_v2_metric_{da_dataset}_{encoder}.pth",
        )
        self.da.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        self.da.eval()
        for param in self.da.parameters():
            param.requires_grad = False

        # --- DA preprocessing transform (matches paper pipeline) ---
        # Produces the same output as DA's `image2tensor_simplified`:
        # aspect-ratio-preserving resize, shorter side to 518, each dim a
        # multiple of 14, cubic interpolation.
        self.da_transform = Compose([
            Resize(
                width=518,
                height=518,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet(),
        ])

        # --- Learnable layers ---
        self.conv = ConvBnReLU(
            in_channels=self.da_out_channels,
            out_channels=128, kernel_size=3, padding=1, stride=1,
        )
        self.pos_mlp_2d = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
        )
        self.pos_mlp_1d = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
        )
        self.q_proj = nn.Linear(160, 128, bias=False)
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)
        self.attn = Attention()

    # ------------------------------------------------------------------ #
    def _encode_with_da(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Run the frozen DA backbone on a batch of (B, 3, H, W) images,
        using the paper's preprocessing pipeline.
        """
        B, C, H, W = imgs.shape
        fH, fW = H // 16, W // 16

        imgs_np = imgs.cpu().numpy()
        resized = []
        for i in range(B):
            img_hwc = imgs_np[i].transpose(1, 2, 0)            # CHW -> HWC
            img_t = self.da_transform({"image": img_hwc})["image"]  # returns CHW numpy
            resized.append(torch.from_numpy(img_t))
        imgs_resized = torch.stack(resized).to(imgs.device)

        with torch.no_grad():
            feats = self.da.pretrained.get_intermediate_layers(
                imgs_resized, reshape=True,
            )[0]
            feats = F.interpolate(
                feats, size=(fH, fW), mode="bilinear", align_corners=True,
            )
        return feats

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            (N, 3, H, W) or (N, L, 3, H, W) normalised images.
        mask : torch.Tensor or None
            (N, H, W) or (N, L, H, W) validity masks.

        Returns
        -------
        out : (N, fW, 128) — attention-pooled depth features.
        attn_w : attention weights.
        """
        # Support (N, C, H, W) and (N, L, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)
            squeeze_L = True
        else:
            squeeze_L = False

        N, L, C, H, W = x.shape
        NL = N * L

        # 1. DA encoding
        x_flat = self._encode_with_da(x.view(NL, C, H, W))

        # 2. Conv
        x_flat = self.conv(x_flat)
        fH, fW = x_flat.shape[2], x_flat.shape[3]

        # 3. Vertical pooling → query
        query = x_flat.mean(dim=2).permute(0, 2, 1)  # (NL, fW, 128)

        # 4. Keys / values
        kv = x_flat.view(NL, 128, -1).permute(0, 2, 1)  # (NL, fH*fW, 128)

        # 5. 2-D positional encoding
        pos_x = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, fH, device=x.device) - 0.5
        gx, gy = torch.meshgrid(pos_x, pos_y, indexing="ij")
        pos_2d = self.pos_mlp_2d(torch.stack((gx, gy), dim=-1))
        pos_2d = pos_2d.reshape(1, -1, 32).repeat(NL, 1, 1)
        kv = torch.cat((kv, pos_2d), dim=-1)

        # 6. 1-D positional encoding
        pos_v = torch.linspace(0, 1, fW, device=x.device) - 0.5
        pos_1d = self.pos_mlp_1d(pos_v.reshape(-1, 1))
        pos_1d = pos_1d.reshape(1, fW, 32).repeat(NL, 1, 1)
        query = torch.cat((query, pos_1d), dim=-1)

        # 7. Projections
        q = self.q_proj(query)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # 8. Mask
        attn_mask = None
        if mask is not None:
            mask_flat = mask.view(NL, 1, H, W).float()
            mask_flat = fn.resize(
                mask_flat, (fH, fW), fn.InterpolationMode.NEAREST,
            ).squeeze(1).bool()
            attn_mask = torch.logical_not(mask_flat).reshape(NL, 1, -1)
            attn_mask = attn_mask.repeat(1, fW, 1)

        # 9. Attention
        out, attn_w = self.attn(q, k, v, attn_mask=attn_mask)

        # 10. Reshape
        out = out.view(N, L, fW, 128)
        if squeeze_L:
            out = out.squeeze(1)

        return out, attn_w