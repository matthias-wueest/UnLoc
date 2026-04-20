# Adapted from F3Loc (Chen et al., CVPR 2024; MIT License).
# https://github.com/felix-ch/f3loc

"""
PyTorch Lightning wrapper for the depth + uncertainty prediction network.

This module provides ``UnLocDepthModule``, which wraps
the core ``UnLocDepthNet`` encoder with Lightning
training/validation logic and a Laplace negative log-likelihood loss.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl

from modules.depth_net import UnLocDepthNet


class UnLocDepthModule(pl.LightningModule):
    """
    Lightning module for training the depth + uncertainty predictor.

    Loss: Laplace NLL + optional cosine-similarity shape loss.

    Parameters
    ----------
    shape_loss_weight : float or None
        Weight for the cosine-similarity shape regulariser. If None,
        only the Laplace NLL is used.
    lr : float
        Learning rate for Adam.
    F_W : float
        Focal-length / image-width ratio.
    """

    def __init__(
        self,
        shape_loss_weight: float | None = None,
        lr: float = 1e-3,
        F_W: float = 3 / 8,
    ) -> None:

        super().__init__()
        self.lr = lr
        self.F_W = F_W
        self.encoder = UnLocDepthNet()
        self.shape_loss_weight = shape_loss_weight
        self.save_hyperparameters()

        self.training_metrics = []
        self.validation_metrics = []

    # ------------------------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------------------------

    @staticmethod
    def laplace_nll_loss(y_true, loc, scale):
        """Negative log-likelihood under a Laplace distribution."""
        return torch.mean(torch.abs(y_true - loc) / scale + torch.log(2 * scale))

    # ------------------------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------------------------

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    # ------------------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loc, scale, _, _ = self.encoder(batch["ref_img"], batch["ref_mask"])

        if torch.isnan(loc).any() or torch.isnan(scale).any() or torch.isnan(batch["ref_depth"]).any():
            return None

        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        loss = nll_loss
        metrics = {"nll_loss": nll_loss}

        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            if torch.isnan(cosine_sim).any():
                return None
            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            loss = loss + shape_loss
            metrics["shape_loss"] = shape_loss

        metrics["loss"] = loss
        self.training_metrics.append(metrics)
        self.log_dict(
            {f"{k}-train": v for k, v in metrics.items()},
            prog_bar=True, logger=True,
        )
        return loss

    def on_train_epoch_end(self):
        if self.training_metrics:
            avg = {
                key: torch.stack([m[key] for m in self.training_metrics]).mean()
                for key in self.training_metrics[0]
            }
            self.log_dict(
                {f"avg_{k}-train": v for k, v in avg.items()},
                prog_bar=True, logger=True,
            )
        self.training_metrics = []

    # ------------------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        loc, scale, _, _ = self.encoder(batch["ref_img"], batch["ref_mask"])
        nll_loss = self.laplace_nll_loss(batch["ref_depth"], loc, scale)
        loss = nll_loss
        metrics = {"nll_loss": nll_loss}

        if self.shape_loss_weight is not None:
            cosine_sim = F.cosine_similarity(loc, batch["ref_depth"], dim=-1).mean()
            shape_loss = self.shape_loss_weight * (1 - cosine_sim)
            loss = loss + shape_loss
            metrics["shape_loss"] = shape_loss

        metrics["loss"] = loss
        self.validation_metrics.append(metrics)
        self.log_dict(
            {f"{k}-valid": v for k, v in metrics.items()},
            prog_bar=True, logger=True,
        )
        return loss

    def on_validation_epoch_end(self):
        if self.validation_metrics:
            avg = {
                key: torch.stack([m[key] for m in self.validation_metrics]).mean()
                for key in self.validation_metrics[0]
            }
            self.log_dict(
                {f"avg_{k}-valid": v for k, v in avg.items()},
                prog_bar=True, logger=True,
            )
        self.validation_metrics = []
