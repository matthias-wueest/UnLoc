# Based on F3Loc (Chen et al., CVPR 2024; MIT License).
# https://github.com/felix-ch/f3loc

"""
Neural network building blocks used by the depth prediction network.

Contains attention modules and convolution helpers used by the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Scaled dot-product attention with optional masking and value clamping."""

    def forward(self, Q, K, V, attn_mask=None):
        """
        Parameters
        ----------
        Q : (N, L, D) — queries.
        K : (N, S, D) — keys.
        V : (N, S, D) — values.
        attn_mask : (N, L, S) or None — True entries are masked out.

        Returns
        -------
        x : (N, L, D) — attended values.
        attn_weights : (N, L, S) — attention weights.
        """
        QK = torch.einsum("nld,nsd->nls", Q, K)
        if attn_mask is not None:
            QK[attn_mask] = -torch.inf

        D = Q.shape[2]
        QK_scaled = torch.clamp(QK / (D ** 0.5), min=-1e11, max=1e11)
        attn_weights = torch.softmax(QK_scaled, dim=2)

        x = torch.einsum("nsd,nls->nld", V, attn_weights)
        return x, attn_weights


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------

class ConvBn(nn.Module):
    """Conv2d + BatchNorm2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.convbn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.convbn(x)


class ConvBnReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.convbn = ConvBn(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding)

    def forward(self, x):
        return F.relu(self.convbn(x))

