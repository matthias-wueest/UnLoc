"""
Training script for the DepthAnything uncertainty depth prediction network.

Supports two datasets:
  - gibson_t   (Gibson synthetic)
  - lamar_hge  (LaMAR HGE)

Usage:
    python train.py --dataset_path /path/to/Gibson_Floorplan_Localization_Dataset --dataset gibson_t
    python train.py --dataset_path /path/to/LaMAR_HGE --dataset lamar_hge
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

import lightning as Lightning
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from modules.depth_net_pl import UnLocDepthModule
from utils.data_utils import (
    LaMARHGEFrameDataset,
    GibsonFrameDataset,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the DepthAnything uncertainty depth prediction network."
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Root path to the dataset.",
    )
    parser.add_argument(
        "--dataset", type=str, default="lamar_hge",
        choices=["gibson_f", "lamar_hge"],
        help="Which dataset variant to train on.",
    )

    # Model hyperparameters
    parser.add_argument("--shape_loss_weight", type=float, default=None, help="Weight for shape (cosine) loss. None = disabled.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_checkpointing", action="store_true", help="Disable checkpointing.")

    # Augmentation
    parser.add_argument("--add_rp", action="store_true", default=False, help="Enable roll/pitch augmentation.")
    parser.add_argument("--roll", type=float, default=0.1, help="Maximum roll augmentation [rad].")
    parser.add_argument("--pitch", type=float, default=0.1, help="Maximum pitch augmentation [rad].")

    args = parser.parse_args()

    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_datasets(args, dataset_dir, depth_dir, split):
    """Instantiate train and validation datasets."""
    if args.dataset == "lamar_hge":
        depth_suffix = "depth90"
        make_ds = lambda scene_names: LaMARHGEFrameDataset(
            dataset_dir,
            scene_names,
            depth_dir=depth_dir,
            depth_suffix=depth_suffix,
        )
    else:  # gibson_t
        depth_suffix = "depth40"
        make_ds = lambda scene_names: GibsonFrameDataset(
            dataset_dir,
            scene_names,
            depth_dir=depth_dir,
            depth_suffix=depth_suffix,
            add_rp=args.add_rp,
            roll=args.roll,
            pitch=args.pitch,
        )

    train_dataset = make_ds(split["train"])
    val_dataset = make_ds(split["val"])
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    args = parse_args()

    # --- Paths ---
    dataset_dir = os.path.join(args.dataset_path, args.dataset)
    depth_dir = dataset_dir

    # --- Seed ---
    seed_everything(args.seed, workers=True)

    # --- Dataset ---
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = yaml.safe_load(f)

    train_dataset, val_dataset = build_datasets(args, dataset_dir, depth_dir, split)
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size:   {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
    )

    # --- Model ---
    model = UnLocDepthModule(
        shape_loss_weight=args.shape_loss_weight,
        lr=args.lr,
    )

    # --- Trainer ---
    logger = TensorBoardLogger("tb_logs", name="my_model")

    checkpoint_callback = ModelCheckpoint(
        monitor="loss-valid",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    trainer = Lightning.Trainer(
        max_epochs=args.max_epochs,
        enable_checkpointing=not args.no_checkpointing,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()