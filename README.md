# HCP + MTR: Hierarchical Combinatorial Pruning for Multimodal Motion Transformers

This repository contains the official implementation of **Hierarchical Combinatorial Pruning (HCP)** integrated with a **Multimodal Motion Transformer (MTR)** backbone for real-time trajectory prediction in autonomous driving environments.
 
```
Dense candidates (N_agents × K_modes × T_steps) 
                     ↓
      Stage 1: KFF (Kinematic constraints)
                     ↓
      Stage 2: SRF (Spatial boundaries)
                     ↓
      Stage 3: SCF (Social interactions GNN)
                     ↓
               Sparse Set
                     ↓
      MTR Decoder (GMM Forecasting)
```

---

## 🌐 Public Live Interactive Web Demo

Click the link below to open and test the interactive Control Room Dashboard directly in your web browser without downloading any files:

👉 **[https://Rish-2006.github.io/HCP-Trajectery-Prediction-Model/](https://Rish-2006.github.io/HCP-Trajectery-Prediction-Model/)**

---

## 🚀 Local Execution

Run the application locally to access the backend API and local control room interface:

```bash
# 1. Unpack datasets and generate mock splits
python hcp_project/data/extractor.py --test

# 2. Run benchmarking evaluation (minADE/FDE/MR and latency metrics)
python hcp_project/eval/evaluate.py

# 3. Launch live telemetry control room dashboard
python hcp_project/backend/main.py
```

After launching the backend, open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## Directory Structure

```
.
├── index.html             # Standalone live interactive web dashboard for GitHub Pages
├── hcp_project/
│   ├── data/              # Data parsing and Unified Dataset Router
│   ├── hcp/               # Hierarchical Combinatorial Pruning (KFF, SRF, SCF)
│   ├── mtr_core/          # MTR tokenizers, encoder, decoder, and training
│   ├── fusion/            # Geometry-conditioned cross-attention layer
│   ├── outputs/           # Modality results: route graphs, maps, motion states
│   ├── eval/              # Benchmarking and metrics evaluation
│   ├── paper/             # Main LaTeX draft template filled with metrics
│   ├── backend/           # FastAPI telemetries and stream endpoints
│   ├── ui/                # Vite + React frontend dashboard codebase
│   └── Dockerfile         # Container deployment configuration
└── README.md
```

---

## Core Algorithmic Components

### 1. HCP Pruner (`hcp_project/hcp/`)
- **Stage 1: Kinematic Feasibility Filter (KFF):** Prunes trajectories violating maximum curvature ($\kappa_{max}=0.2$ rad/m), jerk ($j_{max}=5.0$ m/s³), and lateral acceleration ($a_{lat\_max}=4.0$ m/s²).
- **Stage 2: Spatial Reachability Filter (SRF):** Employs a SciPy KD-Tree index over static lane boundaries to detect off-road collisions.
- **Stage 3: Social Compatibility Filter (SCF):** A 3-layer GraphSAGE GNN representing agent-to-agent conflicts.

### 2. MTR Core (`hcp_project/mtr_core/`)
- **Tokenizers:** Polyline PointNet and agent trajectory MLPs.
- **Encoder:** Transformer encoder utilizing Rotary Position Embeddings (RoPE).
- **Fusion Layer:** Fuses map and agent representations using a physical distance-conditioned RBF kernel attention bias.

---

## Telemetry Outputs & Live Dashboard

The control room dashboard provides three output modalities:
1. **Modality 1 (TNT Route Graph):** Directed NetworkX waypoints and auto-playing synthesized text-to-speech navigation cues.
2. **Modality 2 (BEV Map Crop):** Geographic & ego-centric rendering displaying lanes, crosswalks, agent bounding boxes, and predicted trajectories.
3. **Modality 3 (Motion State NLG):** Natural Language explanations for vehicle risks and vector field velocity plots.

---

## Docker Execution

To build and run using Docker:

```bash
cd hcp_project
docker build -t hcp-trajectory-dashboard .
docker run -p 8000:8000 hcp-trajectory-dashboard
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.
