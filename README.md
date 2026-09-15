# HCP + MTR: Hierarchical Pruning for Efficient Trajectory Prediction

An autonomous-driving trajectory prediction system that combines a Motion Transformer (MTR) core with a Hierarchical Combinatorial Pruning (HCP) module. The project's central research question: **does staged, cheap candidate pruning reduce inference cost without meaningfully hurting trajectory prediction accuracy?**

Trained and evaluated on the real [nuScenes](https://www.nuscenes.org/) dataset (trainval split).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Current Results](#current-results)
- [Known Limitations](#known-limitations)
- [Project Structure](#project-structure)

---

## Overview

Trajectory prediction models typically score a fixed set of candidate future paths for each agent. HCP aims to reduce computational cost by cutting that candidate set down using staged filters for kinematic feasibility, spatial/map reachability, and social compatibility. This project implements the pruning module and transformer core to measure the accuracy/cost trade-off. The current implementation does not yet skip decoder computation for pruned candidates, as explained under Known Limitations.

**Inputs the model conditions on:**

- Agent trajectory history (position, velocity, heading)
- Real HD map geometry (lane centerlines, crosswalks, drivable area) via `nuscenes-devkit`
- Real CAM_FRONT camera imagery (via a frozen, ImageNet-pretrained ResNet18 branch)

---

## Architecture

**Stage 1: HCP Pruner** (`hcp_project/hcp/`)

| Filter | Purpose | Uses |
|---|---|---|
| KFF (Kinematic Feasibility Filter) | Rejects candidates violating curvature/jerk/acceleration limits | Candidate geometry only |
| SRF (Spatial Reachability Filter) | Rejects candidates leaving the road / colliding with lane boundaries | Candidates + map |
| SCF (Social Compatibility Filter) | Learned (GraphSAGE GNN) agent-interaction filter | Candidates + agent history |

Candidates fed into the pruner are generated purely from each agent's own recent history (constant-velocity + a bank of turn-rate variations). **No ground truth is used**, matching what would be available at real inference time.

**Stage 2: MTR Core** (`hcp_project/mtr_core/`)

- PointNet-style tokenizer (agent history + map polylines)
- Transformer encoder with Rotary Position Embeddings (RoPE)
- Cross-attention fusion (agent ↔ map, RBF-distance-biased)
- CAM_FRONT image branch (frozen ResNet18 → 256-d embedding, broadcast-fused into every agent token)
- GMM decoder: 6 intention-anchor modes, winner-takes-all regression + classification loss

---

## Setup

```bash
git clone https://github.com/KarthigayanR-2005/HCP-Trajectery-Prediction-Model.git
cd HCP-Trajectery-Prediction-Model

python -m venv hcp_env
hcp_env\Scripts\activate          # Windows
# source hcp_env/bin/activate     # Linux/Mac

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r hcp_project/requirements.txt
pip install ijson nuscenes-devkit
```

Requires an NVIDIA GPU with CUDA support (developed and tested on an RTX 3050, 6GB VRAM).

---

## Data Preparation

1. Register at the [nuScenes download page](https://www.nuscenes.org/download).
2. Download, from the **Trainval** section:
   - `v1.0-trainval_meta.tgz` (metadata, required)
   - `nuScenes-map-expansion-v1.3.zip` (real map geometry, required)
   - At least one `File blobs of 85 scenes` part (camera/LiDAR, used for the image branch; more parts provide more scenes with real images)
3. Place the downloaded files in the repo root, then extract:

```bash
python hcp_project/data/extractor.py --extract
```

The extractor auto-detects whichever files are present, streams large tables with `ijson` to keep memory bounded, and reports which blob parts (if any) are missing.

---

## Training

```bash
python -m hcp_project.mtr_core.train \
    --epochs 40 \
    --batch_size 2 \
    --max_steps_per_epoch 5000 \
    --save_every_steps 1000
```

**Resuming** (recommended for any run beyond the first; checkpoints save every `save_every_steps`, independent of chunk size):

```bash
python -m hcp_project.mtr_core.train \
    --resume_model hcp_project/outputs/mtr_checkpoint.pth \
    --epochs 40 --batch_size 2 --max_steps_per_epoch 5000 --save_every_steps 1000
```

**Key flags:**

| Flag | Purpose |
|---|---|
| `--max_steps_per_epoch` | Caps each "epoch" to N steps; useful since one true full pass over the dataset is ~195k steps |
| `--save_every_steps` | Checkpoint frequency, independent of chunk size |
| `--lr_patience` / `--lr_factor` | `ReduceLROnPlateau` scheduler settings (halves LR after N stagnant chunks by default) |
| `--override_lr` | Explicitly reset LR (and scheduler history) on resume; useful after a fix that changes model behavior |
| `--profile_steps N` | Print a data-loading-time vs. compute-time breakdown for the first N steps |

---

## Evaluation

```bash
# Real minADE / minFDE / Miss Rate, comparing HCP-on vs. HCP-off
python hcp_project/eval/evaluate.py \
    --checkpoint hcp_project/outputs/mtr_checkpoint.pth \
    --compare_hcp --num_samples 2000

# Restrict to the official nuScenes val split (150 scenes)
python hcp_project/eval/evaluate.py \
    --checkpoint hcp_project/outputs/mtr_checkpoint.pth \
    --compare_hcp --num_samples 2000 --val_split_only

# Diagnostic: separates regression loss from classification loss,
# and checks predicted-vs-ground-truth coordinate scale
python hcp_project/eval/diagnose_loss.py \
    --checkpoint hcp_project/outputs/mtr_checkpoint.pth \
    --use_hcp --num_batches 100
```

`evaluate.py` runs genuine model inference on real data and writes results to `hcp_project/outputs/eval_real_<timestamp>.json`, including an explicit note on data-split caveats for every run.

---

## Current Results

Reported full-dataset evaluation results after ~1.2M training steps (~6 full passes over the available trajectory data):

| Metric | Value |
|---|---|
| minADE | ~24.6 m |
| minFDE | ~24.3 m |
| Miss Rate (2m) | ~98.5% |
| Inference latency (HCP on / off) | ~9-16 ms / ~8-9 ms per agent |

On the official 150-scene val split (soft check; see limitations below): minADE 25.82m, minFDE 25.17m, Miss Rate 98.9%, closely matching the full-dataset numbers.

---

## Known Limitations

- **No true held-out validation split.** The current model trained on the full 850-scene dataset before any val/train separation was introduced. The val-split numbers above are an approximate check on previously seen data, not a rigorous generalization measure. Held-out validation requires training a fresh model on only the official 700 train scenes and evaluating on the 150 validation scenes.
- **HCP pruning does not yet reduce real inference latency.** The pruning mask uses only history/map, but the current decoder always computes all 6 candidate modes regardless of the mask. Pruning affects which mode is trusted, not how much is computed. Making pruning skip real computation requires a decoder architecture change, not yet implemented.
- **Only partial image coverage.** Only 1 of 10 nuScenes camera/LiDAR blob parts has been downloaded; the majority of training examples fall back to a zero-image placeholder rather than a real photo.
- **Miss Rate remains high (~98%)** at a 2m threshold. The model is not yet at production-grade accuracy.

---

## Project Structure

```text
hcp_project/
├── data/            # Dataset extraction, parsing, streaming (nuScenes + WOMD)
├── mtr_core/        # Transformer model: tokenizer, encoder, decoder, training loop, image encoder
├── hcp/             # Pruning filters: KFF, SRF, SCF
├── fusion/          # Cross-attention fusion layer
├── eval/            # Real evaluation and diagnostic scripts
├── utils/           # Mixed-precision training utilities
└── outputs/         # Checkpoints, logs, evaluation results (generated, not tracked)
```

---

## Acknowledgements

Built on the [nuScenes](https://www.nuscenes.org/) dataset (Caesar et al.) and uses [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) for map and split utilities.