"""
dataset_streamer.py
-------------------
High-throughput PyTorch streaming data pipeline for the HCP + MTR
autonomous driving model.

Design goals
~~~~~~~~~~~~
* **No max-agent padding** — heterogeneous agent counts per scene are
  handled by a custom collate function that concatenates agents along
  dimension-0 and records scene boundaries via ``batch_splits``.
* **Worker-safe** — no NetworkX DiGraph objects are kept in dataset
  state; route-graph edge lists are serialised to plain Python lists
  so they survive multiprocessing fork/spawn.
* **Map-aware** — the collate function also packs variable lists of
  map polyline tensors and records per-scene polyline counts via
  ``map_splits``.
* **Fault-tolerant** — corrupted / empty scenarios return a minimal
  placeholder so a single bad record never aborts the epoch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LargeScaleDrivingStreamer(IterableDataset):
    """
    Streaming ``IterableDataset`` that yields one processed scenario at a
    time, converting ``UnifiedBatch`` objects (produced by *parser_instance*)
    into plain-tensor dictionaries that are safe to transfer across
    ``DataLoader`` worker processes.

    Args:
        scenarios_list   : ordered list of scenario IDs to stream.
        parser_instance  : a ``DatasetRouter`` (or any object whose
                           ``__getitem__`` accepts an integer index and
                           returns a ``UnifiedBatch``).
        shuffle          : if True, iterate in a random order each epoch.
        seed             : RNG seed used when shuffle=True.
    """

    def __init__(self,
                 scenarios_list: List[Any],
                 parser_instance,
                 shuffle: bool = False,
                 seed: int = 42):
        super().__init__()
        self.scenarios      = list(scenarios_list)
        self.parser         = parser_instance
        self.shuffle        = shuffle
        self.seed           = seed

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.scenarios)

    # ------------------------------------------------------------------
    def __iter__(self):
        """
        Yield one scenario dictionary per call.  Worker info is used to
        shard the index list so each DataLoader worker processes a disjoint
        subset.
        """
        worker_info = torch.utils.data.get_worker_info()
        indices     = list(range(len(self.scenarios)))

        if self.shuffle:
            rng = np.random.default_rng(
                self.seed + (worker_info.id if worker_info else 0))
            rng.shuffle(indices)

        if worker_info is not None:
            # Slice indices for this worker
            indices = indices[worker_info.id :: worker_info.num_workers]

        for idx in indices:
            yield self._load(idx)

    # ------------------------------------------------------------------
    def _load(self, idx: int) -> Dict[str, Any]:
        """Load one scenario, converting to tensors.  Returns a placeholder
        dict on any error so the epoch is never aborted."""
        try:
            batch = self.parser[idx]

            hist  = torch.as_tensor(np.array(batch.history_traj), dtype=torch.float32)
            fut   = torch.as_tensor(np.array(batch.future_traj),  dtype=torch.float32)

            # Convert list[np.ndarray] → list[Tensor] (avoid stacking —
            # polylines have variable point counts)
            map_tensors: List[torch.Tensor] = [
                torch.as_tensor(p, dtype=torch.float32)
                for p in batch.map_polylines
            ]

            # Serialise route graph as a plain edge list (worker-safe)
            G    = batch.sdc_route_graph
            sdc_edges = list(G.edges(data=True)) if G is not None else []

            return {
                "history":     hist,               # (N, T_hist, 6)
                "future":      fut,                # (N, T_fut,  5)
                "map":         map_tensors,         # list of (P_i, 3) tensors
                "agent_types": batch.agent_types,   # list[str]
                "sdc_edges":   sdc_edges,           # list of (u, v, attr)
                "scenario_id": batch.scenario_id,
                "num_agents":  hist.shape[0],
                "camera_image": batch.camera_image, # (3, 224, 224) tensor
                "has_image":    batch.has_image,    # bool
            }

        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load scenario idx=%d: %s", idx, exc)
            return {
                "history":     torch.zeros((1, 5, 6),  dtype=torch.float32),
                "future":      torch.zeros((1, 12, 5), dtype=torch.float32),
                "map":         [torch.zeros((2, 3),    dtype=torch.float32)],
                "agent_types": ["unknown"],
                "sdc_edges":   [],
                "scenario_id": f"error_idx_{idx}",
                "num_agents":  1,
                "camera_image": torch.zeros(3, 224, 224),
                "has_image":    False,
            }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def dynamic_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a list of scenario dicts into a single batch dict without
    padding agents to a global maximum.

    Each scene may have a different number of agents (N_i) and a different
    number of map polylines (M_i).  Rather than allocating a dense
    (B, N_max, …) tensor, agents are concatenated along dimension-0 and
    the per-scene agent/polyline counts are stored in ``batch_splits`` /
    ``map_splits``.  Downstream code can use these to recover per-scene
    slices:
        ``torch.split(packed_hist, batch_splits, dim=0)``

    Returns
    -------
    dict with keys:
        history_traj : (sum_N, T_hist, 6)  — all agents from all scenes
        future_traj  : (sum_N, T_fut,  5)
        map_tensors  : list[Tensor(P_i, 3)] — all polylines, all scenes
        agent_types  : flat list[str]
        batch_splits : list[int] — number of agents per scene
        map_splits   : list[int] — number of polylines per scene
        scenario_ids : list[str]
    """
    histories    : List[torch.Tensor] = []
    futures      : List[torch.Tensor] = []
    all_maps     : List[torch.Tensor] = []
    all_types    : List[str]          = []
    batch_splits : List[int]          = []
    map_splits   : List[int]          = []
    scenario_ids : List[str]          = []
    camera_images: List[torch.Tensor] = []
    has_images   : List[bool]         = []

    for item in batch:
        hist = item["history"]   # (N_i, T_hist, 6)
        fut  = item["future"]    # (N_i, T_fut,  5)

        # Guard: ensure 3-D tensors even for edge-case placeholders
        if hist.dim() == 2:
            hist = hist.unsqueeze(0)
        if fut.dim() == 2:
            fut  = fut.unsqueeze(0)

        histories.append(hist)
        futures.append(fut)
        batch_splits.append(hist.shape[0])

        polys = item["map"]  # list[Tensor]
        all_maps.extend(polys)
        map_splits.append(len(polys))

        all_types.extend(item["agent_types"])
        scenario_ids.append(item["scenario_id"])

        camera_images.append(item.get("camera_image", torch.zeros(3, 224, 224)))
        has_images.append(item.get("has_image", False))

    return {
        "history_traj": torch.cat(histories, dim=0),   # (sum_N, T_hist, 6)
        "future_traj":  torch.cat(futures,   dim=0),   # (sum_N, T_fut,  5)
        "map_tensors":  all_maps,                       # variable-length list
        "agent_types":  all_types,
        "batch_splits": batch_splits,
        "map_splits":   map_splits,
        "scenario_ids": scenario_ids,
        "camera_images": torch.stack(camera_images, dim=0),  # (B, 3, 224, 224)
        "has_image":     torch.tensor(has_images, dtype=torch.bool),  # (B,)
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_streaming_dataloader(
    scenarios_list,
    parser_instance,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True,
    seed: int = 42,
    prefetch_factor: Optional[int] = 2,
) -> DataLoader:
    """
    Convenience factory that wires together ``LargeScaleDrivingStreamer``
    and ``dynamic_collate_fn`` into a production-ready ``DataLoader``.

    Args:
        scenarios_list   : list of scenario IDs / indices accepted by parser.
        parser_instance  : ``DatasetRouter`` or compatible parser.
        batch_size       : number of *scenes* per batch.
        num_workers      : parallel data-loading workers (0 = main process).
        shuffle          : randomise iteration order each epoch.
        pin_memory       : pin host memory for faster GPU transfer.
        seed             : RNG seed for shuffle.
        prefetch_factor  : number of batches to prefetch per worker
                           (ignored when num_workers == 0).

    Returns:
        DataLoader configured for high-throughput streaming.
    """
    dataset = LargeScaleDrivingStreamer(
        scenarios_list=scenarios_list,
        parser_instance=parser_instance,
        shuffle=shuffle,
        seed=seed,
    )

    loader_kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        collate_fn=dynamic_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)