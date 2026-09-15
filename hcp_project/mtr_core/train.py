"""
train.py — HCP + MTR training loop (optimised)
------------------------------------------------
Integrations vs. the original:
  1. ``LargeScaleDrivingStreamer`` + ``dynamic_collate_fn`` replace the
     plain ``DataLoader(DatasetRouter, …)`` call → no max-agent padding,
     multi-worker streaming, fault-tolerant loading.
  2. ``ScaleTrainingManager`` wraps the optimizer → AMP (float16/bfloat16)
     + gradient accumulation.  Manual ``loss.backward(); optimizer.step()``
     blocks are removed.
  3. Map tokenisation distance computation is fully vectorised using
     ``torch.cdist`` — the old O(M × N) Python loop is gone.
  4. New CLI args: ``--grad_accum`` (default 4), ``--num_workers`` (default 2).
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from hcp_project.data.dataset_router import DatasetRouter
from hcp_project.data.dataset_streamer import (
    LargeScaleDrivingStreamer,
    dynamic_collate_fn,
    build_streaming_dataloader,
)
from hcp_project.mtr_core.tokenizer import MapTokenizer, AgentTokenizer
from hcp_project.mtr_core.encoder import MTREncoder
from hcp_project.fusion.fusion import CrossAttentionFusionLayer
from hcp_project.mtr_core.decoder import MTRDecoder
from hcp_project.mtr_core.image_encoder import ImageEncoder
from hcp_project.hcp.pruner import HierarchicalCombinatorialPruner, generate_kinematic_candidates
from hcp_project.utils.mixed_precision import ScaleTrainingManager


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MTRMotionTransformer(nn.Module):
    def __init__(self, d_model: int = 256, n_modes: int = 6, T_fut: int = 12,
                 use_image: bool = True, pretrained_image: bool = True):
        super().__init__()
        self.map_tokenizer   = MapTokenizer(in_channels=3,  d_model=d_model)
        self.agent_tokenizer = AgentTokenizer(in_channels=6, d_model=d_model)
        self.encoder         = MTREncoder(d_model=d_model, nhead=8, num_layers=4)
        self.fusion          = CrossAttentionFusionLayer(d_model=d_model, nhead=8)
        self.decoder         = MTRDecoder(d_model=d_model, n_modes=n_modes, T_fut=T_fut)

        # Optional CAM_FRONT image branch. Backbone stays frozen (see
        # ImageEncoder docstring) — only its projection head trains, since
        # this dataset is small relative to what fine-tuning a ResNet wants.
        # use_image=False fully disables this (forward() ignores any
        # camera_images passed in), for training/testing without vision.
        self.use_image = use_image
        if use_image:
            self.image_encoder = ImageEncoder(out_dim=d_model, pretrained=pretrained_image,
                                               freeze_backbone=True)
        else:
            self.image_encoder = None

    def forward(self, history_traj, map_polylines, hcp_mask=None,
                camera_images=None, has_image_mask=None):
        """
        Args:
            history_traj    : (B, N, T_hist, 6)
            map_polylines   : list[list[np.ndarray]] — outer = batch, inner = polylines
            hcp_mask        : (B, N, n_modes) bool, optional
            camera_images   : (B, 3, 224, 224), optional — preprocessed CAM_FRONT
                               keyframes (see image_encoder.preprocess_image).
                               Ignored if use_image=False or not provided.
            has_image_mask  : (B,) bool/float, optional — which scenes in the
                               batch have a *real* image vs. a zero placeholder
                               (see UnifiedBatch.has_image). Scenes without a
                               real image get zero image-context contribution
                               rather than noise from encoding a blank image.
        """
        B, N, T_hist, _ = history_traj.shape
        device = history_traj.device

        # 1. Tokenise agent histories -------------------------------------------
        flat_hist   = history_traj.view(B * N, T_hist, -1)
        agent_tokens = self.agent_tokenizer(flat_hist)   # (B*N, d_model)
        agent_tokens = agent_tokens.view(B, N, -1)        # (B, N, d_model)

        # 1b. Fuse scene-level image context into every agent token --------------
        # CAM_FRONT is a single ego-centric view of the whole scene, not
        # per-agent, so it's broadcast-added to every agent's token — the
        # same additive-conditioning pattern used elsewhere in this
        # codebase, letting the self-attention encoder below propagate that
        # context across agents rather than just tacking it on afterward.
        if self.use_image and camera_images is not None:
            img_emb = self.image_encoder(camera_images)   # (B, d_model)
            if has_image_mask is not None:
                mask = has_image_mask.to(device=device, dtype=img_emb.dtype).view(B, 1)
                img_emb = img_emb * mask
            agent_tokens = agent_tokens + img_emb.unsqueeze(1)  # (B, 1, d_model) broadcast over N

        # 2. Tokenise map polylines ----------------------------------------------
        max_polys = 20
        padded_map = torch.zeros((B, max_polys, 10, 3), device=device)           # (B, M, P, C)
        distances  = torch.zeros((B, max_polys, N),      device=device)           # (B, M, N)

        for b in range(B):
            polys = map_polylines[b]
            for m in range(min(len(polys), max_polys)):
                poly = polys[m]
                pts  = poly[:, :3] if hasattr(poly, '__len__') else np.asarray(poly)[:, :3]
                if len(pts) > 10:
                    pts = pts[:10]
                elif len(pts) < 10:
                    pts = np.pad(pts, ((0, 10 - len(pts)), (0, 0)), mode='edge')
                padded_map[b, m] = torch.tensor(pts, dtype=torch.float32, device=device)

            # --- Vectorised distance computation (replaces per-(m,n) Python loop) ---
            # poly_centers : (max_polys, 2) — mean of (x, y) over the 10 padded points
            poly_centers = padded_map[b, :, :, :2].mean(dim=1)                    # (M, 2)
            # agent_pos    : (N, 2) — last history step
            agent_pos    = history_traj[b, :, -1, :2]                             # (N, 2)
            # torch.cdist computes pairwise L2; result is (M, N)
            distances[b] = torch.cdist(poly_centers.float(), agent_pos.float())   # (M, N)

        # Tokenise map
        flat_map   = padded_map.view(B * max_polys, 10, -1)
        map_tokens = self.map_tokenizer(flat_map)    # (B*M, d_model)
        map_tokens = map_tokens.view(B, max_polys, -1)  # (B, M, d_model)

        # 3. Encode (6-layer RoPE transformer) -----------------------------------
        agent_context = self.encoder(agent_tokens)    # (B, N, d_model)

        # 4. Cross-attention fusion ----------------------------------------------
        fused_map  = self.fusion(map_tokens, agent_context, distances)  # (B, M, d_model)

        # 5. Decode --------------------------------------------------------------
        traj_out, confidences = self.decoder(agent_context, fused_map, hcp_mask)

        return traj_out, confidences


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_gmm_loss(pred_trajs, confidences, gt_trajs):
    """
    GMM Negative Log-Likelihood loss.

    pred_trajs  : (B, N, K, T, 5)   x, y at indices 0, 1
    confidences : (B, N, K)
    gt_trajs    : (B, N, T, 5)
    """
    B, N, K, T, _ = pred_trajs.shape

    pred_xy = pred_trajs[..., :2]
    gt_xy   = gt_trajs[..., :2].unsqueeze(2)         # (B, N, 1, T, 2)

    dists = torch.norm(pred_xy - gt_xy, dim=-1)       # (B, N, K, T)
    ade   = dists.mean(dim=-1)                         # (B, N, K)

    best_mode_idx = ade.argmin(dim=-1)                 # (B, N)

    # Regression on best mode (x, y only)
    best_idx_exp  = (best_mode_idx
                     .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                     .expand(-1, -1, -1, T, 2))
    best_pred_xy  = torch.gather(pred_xy, 2, best_idx_exp).squeeze(2)  # (B, N, T, 2)
    reg_loss      = F.smooth_l1_loss(best_pred_xy, gt_trajs[..., :2])

    # Classification (winner-takes-all CE)
    flat_conf   = confidences.view(B * N, K)
    flat_target = best_mode_idx.view(B * N)
    cls_loss    = F.nll_loss(torch.log(flat_conf + 1e-8), flat_target)

    return reg_loss + 2.0 * cls_loss, reg_loss.item(), cls_loss.item()


# ---------------------------------------------------------------------------
# Loss wrapper used by ScaleTrainingManager.execute_step
# ---------------------------------------------------------------------------

def _compute_loss_for_manager(model, batch_data, device):
    """
    Bridges the ScaleTrainingManager API to the model's forward + GMM loss.
    batch_data must contain 'hist_traj', 'gt_traj', 'map_polylines', 'hcp_mask'.
    Optionally contains 'camera_images' / 'has_image_mask' for the image branch.
    """
    hist_traj    = batch_data["hist_traj"].to(device)
    gt_traj      = batch_data["gt_traj"].to(device)
    map_polylines = batch_data["map_polylines"]
    hcp_mask     = batch_data["hcp_mask"].to(device) if batch_data.get("hcp_mask") is not None else None
    camera_images  = batch_data.get("camera_images")
    has_image_mask = batch_data.get("has_image_mask")
    if camera_images is not None:
        camera_images = camera_images.to(device)

    pred_trajs, confidences = model(hist_traj, map_polylines, hcp_mask,
                                     camera_images=camera_images, has_image_mask=has_image_mask)
    return compute_gmm_loss(pred_trajs, confidences, gt_traj)


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

def train_model(
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
    grad_accum: int = 4,
    num_workers: int = 2,
    resume_model: str = None,
    save_every: int = 1,
    max_steps_per_epoch: int = None,
    profile_steps: int = 0,
    save_every_steps: int = 500,
    lr_patience: int = 2,
    lr_factor: float = 0.5,
    override_lr: float = None,
):
    print("Initialising training pipeline …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # ------------------------------------------------------------------ data
    data_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nuscenes_dir = os.path.join(data_dir, "data", "nuscenes")
    waymo_dir    = os.path.join(data_dir, "data", "waymo")

    # Build base dataset router (index-addressable)
    print("Building DatasetRouter (this loads and processes all nuScenes metadata "
          "into trajectory slices — can take a couple of minutes on the full trainval split)...")
    base_dataset = DatasetRouter(nuscenes_dir, waymo_dir, mode="nuscenes")

    # High-throughput streaming dataloader
    scenario_indices = list(range(len(base_dataset)))
    dataloader = build_streaming_dataloader(
        scenarios_list=scenario_indices,
        parser_instance=base_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    # -------------------------------------------------------------- model
    print("Building model (downloads ImageNet-pretrained ResNet18 weights on first "
          "run if not already cached — needs internet access)...")
    model  = MTRMotionTransformer(d_model=256, n_modes=6).to(device)
    print("Model built.")
    pruner = HierarchicalCombinatorialPruner().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning-rate scheduler: halves LR when the per-chunk loss stops
    # improving for `lr_patience` consecutive chunks — the standard fix for
    # a genuine training plateau (loss flat across several full chunks,
    # not just step-to-step noise within one chunk).
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience,
    )

    # AMP + gradient accumulation manager
    scale_mgr = ScaleTrainingManager(
        model=model,
        optimizer=optimizer,
        grad_accumulation_steps=grad_accum,
    )

    history_loss: list = []
    start_epoch = 0
    total_steps_trained = 0  # persists across epochs/chunks and across --resume_model

    # ------------------------------------------------------- resume, if asked
    if resume_model:
        if not os.path.exists(resume_model):
            raise FileNotFoundError(f"--resume_model path does not exist: {resume_model}")
        print(f"Resuming from checkpoint: {resume_model}")
        ckpt = torch.load(resume_model, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            # New-style checkpoint (saved by this updated train.py)
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if missing or unexpected:
                print(f"Note: resumed with a checkpoint saved before the image branch "
                      f"existed (or a mismatched architecture) — {len(missing)} new "
                      f"param(s) initialised fresh (e.g. image_encoder.*), "
                      f"{len(unexpected)} old param(s) ignored.")
            if ckpt.get("optimizer_state_dict") is not None:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: could not restore optimizer state ({e}); "
                          f"continuing with a freshly-initialised optimizer.")
            if ckpt.get("scheduler_state_dict") is not None:
                try:
                    lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    print(f"Resumed LR scheduler state (current LR: "
                          f"{optimizer.param_groups[0]['lr']:.2e}).")
                except Exception as e:
                    print(f"Warning: could not restore LR scheduler state ({e}); "
                          f"continuing with a freshly-initialised scheduler.")
            else:
                print("Note: checkpoint predates the LR scheduler — starting it fresh "
                      "from the current LR (no restart needed, this is expected for "
                      "any checkpoint saved before this feature was added).")
            start_epoch = ckpt.get("epoch", 0)
            history_loss = ckpt.get("history_loss", [])
            total_steps_trained = ckpt.get("total_steps", 0)
            print(f"Resumed model + optimizer from epoch {start_epoch}, "
                  f"{total_steps_trained:,} total steps previously trained "
                  f"({len(history_loss)} prior loss entries).")

            if override_lr is not None:
                # Explicitly reset the LR after loading the checkpoint's saved
                # optimizer/scheduler state — useful when something about the
                # training setup changed (e.g. a bug fix) such that the old
                # plateau/LR-decay history no longer reflects genuine
                # convergence, and the model needs room to actually adapt to
                # the new situation rather than crawling at whatever tiny LR
                # the scheduler had decayed down to beforehand.
                old_lr = optimizer.param_groups[0]["lr"]
                for g in optimizer.param_groups:
                    g["lr"] = override_lr
                # Also reset the scheduler's plateau-tracking state (best loss
                # seen, bad-epoch count, cooldown) — that history was measured
                # under the old conditions and isn't a meaningful baseline for
                # judging whether the model has plateaued under the new ones.
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=lr_factor, patience=lr_patience,
                )
                print(f"  --override_lr set: LR reset {old_lr:.2e} -> {override_lr:.2e}, "
                      f"and the LR scheduler's plateau-tracking history was reset fresh.")
        else:
            # Old-style checkpoint (a bare model.state_dict(), e.g. the
            # original nuScenes-mini run) — weights only, no epoch/optimizer
            # info to restore, so we start counting epochs from 0 but keep
            # the pretrained weights instead of random init.
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print("Resumed model weights from a legacy (state-dict-only) "
                  "checkpoint. No epoch/optimizer info was stored in it, so "
                  "epoch counting restarts at 0, but training continues from "
                  "these weights rather than from scratch.")
            if missing or unexpected:
                print(f"Note: {len(missing)} new param(s) initialised fresh "
                      f"(e.g. image_encoder.* if this checkpoint predates the "
                      f"image branch), {len(unexpected)} old param(s) ignored.")

    print(f"Starting training … (effective batch = {batch_size} × {grad_accum} = "
          f"{batch_size * grad_accum} scenes)")

    out_dir = os.path.join(data_dir, "outputs")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss = epoch_reg = epoch_cls = 0.0
        n_steps    = 0
        start_time = time.time()
        t_prev_end = start_time
        total_steps_estimate = -(-len(base_dataset) // batch_size)  # ceil division
        effective_total = min(total_steps_estimate, max_steps_per_epoch) if max_steps_per_epoch else total_steps_estimate
        print(f"Epoch {epoch + 1}: ~{total_steps_estimate:,} steps in a full pass "
              f"({len(base_dataset):,} scenes / batch_size {batch_size})"
              + (f", capped to {max_steps_per_epoch:,} steps this run." if max_steps_per_epoch else "."))

        for step_idx, collated in enumerate(dataloader):
            if max_steps_per_epoch is not None and step_idx >= max_steps_per_epoch:
                print(f"  Reached --max_steps_per_epoch={max_steps_per_epoch}, ending epoch early.")
                break
            if profile_steps > 0 and step_idx < profile_steps:
                t_data_ready = time.time()
                t_compute_start = t_data_ready
            # ---- Unpack collated batch ----------------------------------------
            # collated["history_traj"] : (sum_N, T_hist, 6) — all agents packed
            # collated["batch_splits"] : list[int]          — agents per scene
            # We need (B, N, T_hist, 6) for the model; pad to max N in this batch.
            packed_hist  = collated["history_traj"]   # (sum_N, T_hist, 6)
            packed_fut   = collated["future_traj"]    # (sum_N, T_fut,  5)
            batch_splits = collated["batch_splits"]   # list[int]
            B            = len(batch_splits)
            N_max        = max(batch_splits)
            T_hist       = packed_hist.shape[1]
            T_fut        = packed_fut.shape[1]

            # Pad to (B, N_max, T, C) — padding only within the micro-batch,
            # NOT to a global maximum across the dataset.
            hist_padded = torch.zeros((B, N_max, T_hist, 6),  dtype=torch.float32)
            fut_padded  = torch.zeros((B, N_max, T_fut,  5),  dtype=torch.float32)
            cursor = 0
            for b, n in enumerate(batch_splits):
                hist_padded[b, :n] = packed_hist[cursor:cursor + n]
                fut_padded[b,  :n] = packed_fut[cursor:cursor + n]
                cursor += n

            hist_padded = hist_padded.to(device)
            fut_padded  = fut_padded.to(device)

            # ---- Camera images (one CAM_FRONT keyframe per scene) -------------
            camera_images  = collated.get("camera_images")   # (B, 3, 224, 224) or None
            has_image_mask = collated.get("has_image")       # (B,) bool or None

            # Reconstruct per-scene polyline lists for the model
            map_tensors = collated["map_tensors"]   # flat list of (P_i, 3) tensors
            map_splits  = collated["map_splits"]    # polylines per scene
            map_polylines_batch: list = []
            mc = 0
            for nm in map_splits:
                scene_polys = [t.numpy() for t in map_tensors[mc:mc + nm]]
                map_polylines_batch.append(scene_polys)
                mc += nm

            # ---- HCP pruning mask (real kinematic candidates from history —
            # no ground truth involved, matching what's actually available
            # at real inference time) --------------------------------------
            dense_candidates = torch.zeros((B, N_max, 6, T_fut, 5), device=device)
            for b in range(B):
                for n in range(batch_splits[b]):
                    dense_candidates[b, n] = generate_kinematic_candidates(
                        hist_padded[b, n], T_fut=T_fut, dt=0.5, K=6)

            hcp_masks = []
            for b in range(B):
                _, mask, _ = pruner(
                    dense_candidates[b, :batch_splits[b]],
                    hist_padded[b, :batch_splits[b]],
                    map_polylines_batch[b],
                )
                # Pad mask to N_max
                pad_rows = N_max - batch_splits[b]
                if pad_rows > 0:
                    mask = torch.cat(
                        [mask, torch.zeros((pad_rows, 6), dtype=torch.bool, device=device)],
                        dim=0)
                hcp_masks.append(mask)
            hcp_mask = torch.stack(hcp_masks)   # (B, N_max, 6)

            # ---- AMP forward + backward via ScaleTrainingManager --------------
            batch_data = {
                "hist_traj":      hist_padded,
                "gt_traj":        fut_padded,
                "map_polylines":  map_polylines_batch,
                "hcp_mask":       hcp_mask,
                "camera_images":  camera_images,
                "has_image_mask": has_image_mask,
            }
            step_loss = scale_mgr.execute_step(
                batch_data, _compute_loss_for_manager, step_idx)

            epoch_loss += step_loss
            n_steps    += 1
            total_steps_trained += 1

            # ---------------------------------------------- step-based checkpoint
            # Saves every save_every_steps steps, independent of chunk/epoch size —
            # so an interruption never costs more than save_every_steps of progress,
            # regardless of how large --max_steps_per_epoch is set to.
            if save_every_steps > 0 and total_steps_trained % save_every_steps == 0:
                ckpt_payload = {
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "epoch":                epoch,  # chunk in progress, not yet complete
                    "history_loss":         history_loss,
                    "total_steps":          total_steps_trained,
                }
                torch.save(ckpt_payload, os.path.join(out_dir, "mtr_checkpoint.pth"))
                print(f"  [checkpoint] saved at {total_steps_trained:,} total steps "
                      f"(mtr_checkpoint.pth updated)")

            if profile_steps > 0 and step_idx < profile_steps:
                torch.cuda.synchronize() if device.type == "cuda" else None
                t_compute_end = time.time()
                data_wait_time = t_data_ready - t_prev_end if step_idx > 0 else 0.0
                compute_time   = t_compute_end - t_compute_start
                print(f"  [profile] step {step_idx}: data_wait={data_wait_time:.3f}s "
                      f"compute={compute_time:.3f}s total={data_wait_time + compute_time:.3f}s")
                t_prev_end = t_compute_end

            if step_idx > 0 and step_idx % 50 == 0:
                elapsed        = time.time() - start_time
                steps_per_sec  = step_idx / elapsed
                running_avg    = epoch_loss / n_steps
                remaining      = effective_total - step_idx
                eta_hours      = (remaining / steps_per_sec / 3600) if steps_per_sec > 0 else float("nan")
                print(f"  step {step_idx:,}/{effective_total:,} | "
                      f"loss (running avg): {running_avg:.4f} | "
                      f"{steps_per_sec:.2f} steps/s | "
                      f"ETA for this epoch: {eta_hours:.1f}h")

        # ---------------------------------------------------------------- epoch summary
        avg_loss = epoch_loss / max(n_steps, 1)
        history_loss.append(avg_loss)
        elapsed  = time.time() - start_time
        final_epoch_number = epoch + 1   # 1-indexed, absolute (accounts for resume)
        print(f"Epoch {final_epoch_number} (target end: {start_epoch + epochs}) | "
              f"Loss: {avg_loss:.4f} | Steps: {n_steps} | Time: {elapsed:.2f}s")

        # ---------------------------------------------------- LR scheduler step
        # Stepped once per completed chunk using that chunk's average loss —
        # deliberately NOT per-step, since per-step loss is noisy (see the
        # running-avg wobble in the printed step lines) and would trigger
        # spurious reductions. A per-chunk average is a much more reliable
        # plateau signal.
        lr_before = optimizer.param_groups[0]["lr"]
        lr_scheduler.step(avg_loss)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            print(f"  [lr scheduler] loss plateaued for {lr_patience} chunks — "
                  f"reducing LR: {lr_before:.2e} -> {lr_after:.2e}")

        # ------------------------------------------------ per-epoch checkpoint
        if save_every > 0 and final_epoch_number % save_every == 0:
            ckpt_payload = {
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "epoch":                final_epoch_number,
                "history_loss":         history_loss,
                "total_steps":          total_steps_trained,
            }
            epoch_ckpt_path = os.path.join(ckpt_dir, f"mtr_epoch_{final_epoch_number}.pth")
            torch.save(ckpt_payload, epoch_ckpt_path)
            # Also overwrite the "latest" pointer used by --resume_model
            torch.save(ckpt_payload, os.path.join(out_dir, "mtr_checkpoint.pth"))
            print(f"  Saved checkpoint: {epoch_ckpt_path} (and updated mtr_checkpoint.pth)")

    # ------------------------------------------------------------------ save outputs
    plt.figure()
    plt.plot(range(1, len(history_loss) + 1), history_loss, marker='o', color='#1D9E75')
    plt.title("HCP + MTR Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    final_payload = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "epoch":                start_epoch + epochs,
        "history_loss":         history_loss,
        "total_steps":          total_steps_trained,
    }
    torch.save(final_payload, os.path.join(out_dir, "mtr_checkpoint.pth"))
    print("Training finished.  Checkpoint saved.")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HCP + MTR autonomous driving model")
    parser.add_argument("--epochs",      type=int,   default=3,    help="Training epochs")
    parser.add_argument("--batch_size",  type=int,   default=2,    help="Scenes per batch")
    parser.add_argument("--lr",          type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad_accum",  type=int,   default=4,
                        help="Gradient accumulation steps (effective batch × this)")
    parser.add_argument("--num_workers", type=int,   default=2,
                        help="DataLoader worker processes (0 = main process)")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="Path to a checkpoint (.pth) to resume from. Accepts "
                             "either this script's own checkpoint format (model + "
                             "optimizer + epoch) or a bare model.state_dict() from "
                             "an older run — the latter restores weights only and "
                             "restarts epoch counting at 0.")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save a checkpoint every N epochs (default: every epoch). "
                             "Set to 0 to only save once at the very end.")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="Cap each epoch to this many steps, then move on. Useful "
                             "for smoke-testing on a huge dataset (e.g. full nuScenes "
                             "trainval, ~195k steps/epoch at batch_size=2) without "
                             "waiting hours to see if training works at all. Omit "
                             "for a real full-dataset training run.")
    parser.add_argument("--profile_steps", type=int, default=0,
                        help="Print a data-loading-time vs. compute-time breakdown "
                             "for the first N steps, to diagnose what's actually slow "
                             "before choosing how to speed up training. 0 = off.")
    parser.add_argument("--save_every_steps", type=int, default=500,
                        help="Save a checkpoint every N steps, independent of chunk/epoch "
                             "size — so an interruption never costs more than this many "
                             "steps of progress, no matter how large --max_steps_per_epoch "
                             "is. Set to 0 to disable (fall back to only per-epoch saves).")
    parser.add_argument("--lr_patience", type=int, default=2,
                        help="Number of consecutive chunks with no loss improvement "
                             "before the learning rate is reduced. Checked once per "
                             "completed chunk (--max_steps_per_epoch), using that "
                             "chunk's average loss.")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="Multiplier applied to the learning rate each time the "
                             "plateau patience is exceeded (0.5 = halve it).")
    parser.add_argument("--override_lr", type=float, default=None,
                        help="When resuming, explicitly reset the learning rate to this "
                             "value (and reset the LR scheduler's plateau-tracking "
                             "history) instead of continuing from whatever the "
                             "checkpoint's saved optimizer state has. Useful after a "
                             "bug fix or other change invalidates the old plateau "
                             "history, so the model gets real room to adapt rather "
                             "than crawling at an already-decayed-down LR.")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
        resume_model=args.resume_model,
        save_every=args.save_every,
        max_steps_per_epoch=args.max_steps_per_epoch,
        profile_steps=args.profile_steps,
        save_every_steps=args.save_every_steps,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        override_lr=args.override_lr,
    )