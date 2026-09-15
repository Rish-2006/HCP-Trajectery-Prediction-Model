"""
Diagnostic: separates reg_loss (coordinate accuracy) from cls_loss
(confidence/classification) using the checkpoint's real predictions on real
data — the training loop only ever tracked their combined sum, so we've
never actually seen this breakdown before.

Also reports the raw scale of predicted vs. ground-truth coordinates, to
directly check whether the model's regression output has converged to a
realistic magnitude or is still wildly miscalibrated.
"""
import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hcp_project.data.dataset_router import DatasetRouter
from hcp_project.data.dataset_streamer import build_streaming_dataloader
from hcp_project.mtr_core.train import MTRMotionTransformer, compute_gmm_loss
from hcp_project.hcp.pruner import HierarchicalCombinatorialPruner
from hcp_project.eval.evaluate import build_batch_tensors, build_hcp_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--nuscenes_dir", type=str, default="hcp_project/data/nuscenes")
    parser.add_argument("--waymo_dir", type=str, default="hcp_project/data/waymo")
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--use_hcp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    base_dataset = DatasetRouter(args.nuscenes_dir, args.waymo_dir, mode="nuscenes")
    dataloader = build_streaming_dataloader(
        list(range(len(base_dataset))), base_dataset,
        batch_size=args.batch_size, num_workers=0, shuffle=False,
    )

    model = MTRMotionTransformer(d_model=256, n_modes=6).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    pruner = HierarchicalCombinatorialPruner().to(device) if args.use_hcp else None

    reg_losses, cls_losses = [], []
    pred_abs_means, gt_abs_means = [], []

    with torch.no_grad():
        for step_idx, collated in enumerate(dataloader):
            if step_idx >= args.num_batches:
                break
            hist_padded, fut_padded, camera_images, has_image_mask, map_polylines_batch, batch_splits = \
                build_batch_tensors(collated, device)

            hcp_mask = None
            if args.use_hcp:
                hcp_mask = build_hcp_mask(fut_padded, hist_padded, map_polylines_batch, batch_splits, pruner, device)

            pred_trajs, confidences = model(
                hist_padded, map_polylines_batch, hcp_mask,
                camera_images=camera_images, has_image_mask=has_image_mask,
            )
            total_loss, reg_loss, cls_loss = compute_gmm_loss(pred_trajs, confidences, fut_padded)
            reg_losses.append(reg_loss)
            cls_losses.append(cls_loss)

            pred_abs_means.append(pred_trajs[..., :2].abs().mean().item())
            gt_abs_means.append(fut_padded[..., :2].abs().mean().item())

    print("\n=== Loss Breakdown (real checkpoint, real data) ===")
    print(f"reg_loss  (coordinate accuracy) : mean={np.mean(reg_losses):.4f}  min={np.min(reg_losses):.4f}  max={np.max(reg_losses):.4f}")
    print(f"cls_loss  (classification)      : mean={np.mean(cls_losses):.4f}  min={np.min(cls_losses):.4f}  max={np.max(cls_losses):.4f}")
    print(f"2 * cls_loss contribution to total loss: {2*np.mean(cls_losses):.4f}")
    print(f"reg_loss contribution to total loss:     {np.mean(reg_losses):.4f}")
    print(f"\n=== Coordinate Scale Check ===")
    print(f"Mean |predicted x,y| : {np.mean(pred_abs_means):.4f}")
    print(f"Mean |ground-truth x,y| : {np.mean(gt_abs_means):.4f}")
    print(f"Ratio (pred/gt) : {np.mean(pred_abs_means)/max(np.mean(gt_abs_means), 1e-8):.2f}x")


if __name__ == "__main__":
    main()