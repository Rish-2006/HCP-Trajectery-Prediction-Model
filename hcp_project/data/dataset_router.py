import os
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from data.nuscenes_parser import NuScenesParser, NuScenesMapWrapper, load_camera_image
    from data.womd_parser import WOMDParser
    from mtr_core.image_encoder import preprocess_image
except ModuleNotFoundError:
    from hcp_project.data.nuscenes_parser import NuScenesParser, NuScenesMapWrapper, load_camera_image
    from hcp_project.data.womd_parser import WOMDParser
    from hcp_project.mtr_core.image_encoder import preprocess_image



# ---------------------------------------------------------------------------
# Coordinate-transform helpers
# ---------------------------------------------------------------------------

def transform_to_ego(x, y, vx, vy, heading, ego_x, ego_y, ego_heading):
    """
    Transforms a *single* point/agent to ego-centric frame (x-forward, y-left).
    Retained for agent-level transforms where scalars or small arrays are passed.
    Complexity: O(1).
    """
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)

    dx = x - ego_x
    dy = y - ego_y

    x_new  =  dx * cos_h + dy * sin_h
    y_new  = -dx * sin_h + dy * cos_h
    vx_new =  vx * cos_h + vy * sin_h
    vy_new = -vx * sin_h + vy * cos_h

    h_new = (heading - ego_heading + np.pi) % (2 * np.pi) - np.pi
    return x_new, y_new, vx_new, vy_new, h_new


def transform_polylines_batch(polylines_raw, ego_x, ego_y, ego_heading):
    """
    Vectorized ego-centric transform for *all* map polylines simultaneously.

    Instead of looping over every polyline and every point (O(M * P) Python
    overhead), this function:
      1. Stacks all (x, y) coordinates from every polyline into a single
         (Total_pts, 2) NumPy array.
      2. Applies a single (2, 2) rotation matrix multiplication — one BLAS call.
      3. Splits the transformed coordinates back into per-polyline arrays.

    Complexity: O(1) sequential matrix operations regardless of M or P.

    Args:
        polylines_raw : list of array-like, each shape (P_i, ≥3) columns [x,y,type,…]
        ego_x, ego_y  : ego position in world frame
        ego_heading   : ego yaw angle in radians

    Returns:
        list[np.ndarray]: transformed polylines, each shape (P_i, 3) [x_ego, y_ego, type]
    """
    if not polylines_raw:
        return []

    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)
    # Rotation matrix: world → ego (x-forward, y-left)
    R = np.array([[ cos_h,  sin_h],
                  [-sin_h,  cos_h]], dtype=np.float64)  # (2, 2)

    # Collect raw point counts per polyline so we can split later
    counts = []
    xy_parts = []
    type_parts = []

    for poly in polylines_raw:
        poly = np.asarray(poly, dtype=np.float64)
        counts.append(len(poly))
        xy_parts.append(poly[:, :2])          # (P_i, 2)
        type_col = poly[:, 2:3] if poly.shape[1] > 2 else np.zeros((len(poly), 1))
        type_parts.append(type_col)

    # Stack → single matrix multiply → single subtract
    all_xy    = np.vstack(xy_parts)           # (Total_pts, 2)
    all_types = np.vstack(type_parts)         # (Total_pts, 1)

    # Translate then rotate: (pts - ego) @ R.T
    translated = all_xy - np.array([[ego_x, ego_y]], dtype=np.float64)
    rotated    = translated @ R.T              # (Total_pts, 2)

    # Reassemble into per-polyline list
    result = []
    split_indices = np.cumsum(counts[:-1])
    xy_splits   = np.split(rotated,    split_indices)
    type_splits = np.split(all_types,  split_indices)

    for xy, t in zip(xy_splits, type_splits):
        result.append(np.concatenate([xy, t], axis=1).astype(np.float32))

    return result


# ---------------------------------------------------------------------------
# UnifiedBatch — memory-mapped backing for large arrays
# ---------------------------------------------------------------------------

class UnifiedBatch:
    """
    Container for one scenario's data.

    When *mmap_dir* is provided the large float32 arrays (history_traj,
    future_traj) are written to numpy memmap files in that directory so that
    only the pages actually accessed are resident in RAM.  When mmap_dir is
    None the raw arrays are stored in memory as before (backward-compatible).

    Attributes
    ----------
    history_traj : np.ndarray or np.memmap  (N_agents, T_hist, 6)
    future_traj  : np.ndarray or np.memmap  (N_agents, T_fut,  5)
    map_polylines: list[np.ndarray]         each (P, 3)
    agent_types  : list[str]
    sdc_route_graph : networkx.DiGraph
    scenario_id  : str
    camera_image : torch.Tensor (3, 224, 224) — preprocessed CAM_FRONT
                   keyframe closest to the prediction pivot. Zeros if no
                   real image was available for this scenario (blob part
                   not downloaded, sensor_files missing, etc.) — check
                   has_image before trusting it as real data.
    has_image    : bool — whether camera_image is real data (True) or a
                   zero placeholder (False).
    """

    def __init__(self, history_traj, future_traj, map_polylines,
                 agent_types, sdc_route_graph, scenario_id, mmap_dir=None,
                 camera_image=None, has_image=False):

        if mmap_dir is not None and history_traj.size > 0:
            # Persist large arrays as memory-mapped files
            hist_path = os.path.join(mmap_dir, f"{scenario_id}_hist.npy")
            fut_path  = os.path.join(mmap_dir, f"{scenario_id}_fut.npy")

            hist_mm = np.memmap(hist_path, dtype=np.float32, mode='w+',
                                shape=history_traj.shape)
            hist_mm[:] = history_traj
            hist_mm.flush()

            fut_mm = np.memmap(fut_path, dtype=np.float32, mode='w+',
                               shape=future_traj.shape)
            fut_mm[:] = future_traj
            fut_mm.flush()

            # Re-open in read mode so subsequent access is read-only
            self.history_traj = np.memmap(hist_path, dtype=np.float32,
                                          mode='r', shape=history_traj.shape)
            self.future_traj  = np.memmap(fut_path,  dtype=np.float32,
                                          mode='r', shape=future_traj.shape)
        else:
            self.history_traj = history_traj
            self.future_traj  = future_traj

        self.map_polylines    = map_polylines
        self.agent_types      = agent_types
        self.sdc_route_graph  = sdc_route_graph
        self.scenario_id      = scenario_id
        self.camera_image     = camera_image if camera_image is not None else torch.zeros(3, 224, 224)
        self.has_image        = has_image


# ---------------------------------------------------------------------------
# DatasetRouter
# ---------------------------------------------------------------------------

class DatasetRouter(Dataset):
    """
    Unified PyTorch Dataset that routes to either nuScenes or WOMD data.

    Args:
        nuscenes_dir : path to nuScenes data root
        waymo_dir    : path to WOMD/Waymo data root
        mode         : "nuscenes" | "waymo"
        mmap_dir     : optional directory for memory-mapped array storage.
                       Pass None (default) to keep arrays in RAM.
    """

    def __init__(self, nuscenes_dir, waymo_dir, mode="nuscenes", mmap_dir=None, scene_filter=None):
        self.mode        = mode.lower()
        self.mmap_dir    = mmap_dir
        self.nuscenes_dir = nuscenes_dir

        self.nuscenes_parser = NuScenesParser(nuscenes_dir, scene_filter=scene_filter)
        self.womd_parser     = WOMDParser(waymo_dir)
        self.map_wrapper     = NuScenesMapWrapper(nuscenes_dir)

        self.nuscenes_data = self.nuscenes_parser.process_dataset()
        print(f"DatasetRouter: {len(self.nuscenes_data)} nuScenes trajectory slices ready.")
        self.waymo_ids     = self.womd_parser.get_all_scenario_ids()
        print("DatasetRouter: initialization complete.")

    # ------------------------------------------------------------------
    def __len__(self):
        if self.mode == "nuscenes":
            return max(len(self.nuscenes_data), 10)
        return len(self.waymo_ids)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Returns a UnifiedBatch for the requested index.
        Complexity: O(N + Total_pts) where N = agents, Total_pts = sum of all
        polyline point counts (one vectorized matrix multiply covers all of them).
        """
        if self.mode == "nuscenes" and len(self.nuscenes_data) > 0:
            batch, scenario_id = self._get_nuscenes(idx)
        else:
            batch, scenario_id = self._get_waymo(idx)

        return batch

    # ------------------------------------------------------------------
    # nuScenes branch
    # ------------------------------------------------------------------
    def _get_nuscenes(self, idx):
        item        = self.nuscenes_data[idx % len(self.nuscenes_data)]
        scenario_id = f"nuscenes_scene_{idx}"

        ego_pose = item["ego_pose"]
        ego_x, ego_y = ego_pose["translation"][0], ego_pose["translation"][1]
        qw, qx, qy, qz = ego_pose["rotation"]
        ego_heading = np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)
        )

        hist_list, fut_list, agent_types = [], [], []

        # --- SDC (ego) --- agent index 0
        sdc_hist, sdc_fut = [], []
        for h in item["history"]:
            x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(
                h["x"], h["y"], h["vx"], h["vy"], h["heading_rad"],
                ego_x, ego_y, ego_heading)
            sdc_hist.append([x_n, y_n, vx_n, vy_n, h_n, 0.0])
        for f in item["future"]:
            x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(
                f["x"], f["y"], f["vx"], f["vy"], f["heading_rad"],
                ego_x, ego_y, ego_heading)
            sdc_fut.append([x_n, y_n, vx_n, vy_n, h_n])

        hist_list.append(sdc_hist)
        fut_list.append(sdc_fut)
        agent_types.append("ego_vehicle")

        # --- Simulated neighbour (offset) ---
        other_hist, other_fut = [], []
        for h in item["history"]:
            x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(
                h["x"] + 4.0, h["y"] + 2.0, h["vx"], h["vy"], h["heading_rad"],
                ego_x, ego_y, ego_heading)
            other_hist.append([x_n, y_n, vx_n, vy_n, h_n, 0.0])
        for f in item["future"]:
            x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(
                f["x"] + 4.0, f["y"] + 2.0, f["vx"], f["vy"], f["heading_rad"],
                ego_x, ego_y, ego_heading)
            other_fut.append([x_n, y_n, vx_n, vy_n, h_n])

        hist_list.append(other_hist)
        fut_list.append(other_fut)
        agent_types.append("vehicle")

        # --- Camera image: CAM_FRONT from the pivot (most-recent-history) frame ---
        # item["history"][-1] is the pivot frame (t=0, the "now" the future is
        # predicted from) — that's the natural frame to pull the camera view from.
        camera_image, has_image = None, False
        sensor_files = item["history"][-1].get("sensor_files", {})
        cam_front_path = sensor_files.get("CAM_FRONT")
        if cam_front_path:
            try:
                pil_img = load_camera_image(self.nuscenes_dir, cam_front_path)
                if pil_img is not None:
                    camera_image = preprocess_image(pil_img)
                    has_image = True
            except Exception as e:
                print(f"Warning: failed to load/preprocess CAM_FRONT image "
                      f"({cam_front_path}): {e} — using zero placeholder for this scenario.")

        # --- Map: vectorized batch transform ---
        map_data  = self.map_wrapper.get_map_elements(ego_x, ego_y)
        raw_lanes = [
            np.array([[pt[0], pt[1], 1.0] for pt in lane.coords])
            for lane in map_data["lanes"]
        ]
        raw_cwalks = [
            np.array([[pt[0], pt[1], 2.0] for pt in cw.exterior.coords])
            for cw in map_data["crosswalks"]
        ]
        map_polylines = transform_polylines_batch(
            raw_lanes + raw_cwalks, ego_x, ego_y, ego_heading)

        # --- Dummy SDC route graph ---
        import networkx as nx  # lazy import — not needed at module load time
        sdc_route_graph = nx.DiGraph()
        prev_node = None
        for idx_pt in range(25):
            curr_node = f"p0_n{idx_pt}"
            sdc_route_graph.add_node(curr_node, x=float(idx_pt * 2.0), y=0.0,
                                     path_idx=0, pt_idx=idx_pt)
            if prev_node is not None:
                sdc_route_graph.add_edge(prev_node, curr_node, weight=2.0)
            prev_node = curr_node

        return UnifiedBatch(
            history_traj=np.array(hist_list, dtype=np.float32),
            future_traj=np.array(fut_list,  dtype=np.float32),
            map_polylines=map_polylines,
            agent_types=agent_types,
            sdc_route_graph=sdc_route_graph,
            scenario_id=scenario_id,
            mmap_dir=self.mmap_dir,
            camera_image=camera_image,
            has_image=has_image,
        ), scenario_id

    # ------------------------------------------------------------------
    # WOMD branch
    # ------------------------------------------------------------------
    def _get_waymo(self, idx):
        if len(self.waymo_ids) == 0:
            sc_id = "synthetic_waymo_0"
        else:
            sc_id = self.waymo_ids[idx % len(self.waymo_ids)]

        scenario = self.womd_parser.load_scenario(sc_id)
        if not scenario:
            scenario = self.womd_parser.scenarios.get("scenario_0", {})
            sc_id    = "scenario_0"

        tracks           = scenario["tracks"]
        map_polylines_raw = scenario["map_polylines"]

        # Ego pose
        sdc_state   = tracks[0]["history"][3]
        ego_x, ego_y = sdc_state[1], sdc_state[2]
        ego_heading  = sdc_state[5]

        hist_list, fut_list, agent_types = [], [], []

        # --- Agent histories / futures (scalar calls — N is small) ---
        for agent_id, data in tracks.items():
            hist_pts, fut_pts = [], []
            type_idx = 0.0 if data["type"] == "vehicle" else 1.0

            for h in data["history"]:
                x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(
                    h[1], h[2], h[3], h[4], h[5], ego_x, ego_y, ego_heading)
                hist_pts.append([x_n, y_n, vx_n, vy_n, h_n, type_idx])

            for f in data["future"]:
                x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(
                    f[0], f[1], f[2], f[3], f[4], ego_x, ego_y, ego_heading)
                fut_pts.append([x_n, y_n, vx_n, vy_n, h_n])

            hist_list.append(hist_pts)
            fut_list.append(fut_pts)
            agent_types.append(data["type"])

        # --- Map: vectorized batch transform (single matrix multiply) ---
        map_polylines = transform_polylines_batch(
            map_polylines_raw, ego_x, ego_y, ego_heading)

        sdc_route_graph = self.womd_parser.build_sdc_route_graph(sc_id)
        scenario_id     = sc_id

        return UnifiedBatch(
            history_traj=np.array(hist_list, dtype=np.float32),
            future_traj=np.array(fut_list,  dtype=np.float32),
            map_polylines=map_polylines,
            agent_types=agent_types,
            sdc_route_graph=sdc_route_graph,
            scenario_id=scenario_id,
            mmap_dir=self.mmap_dir,
        ), scenario_id


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    router = DatasetRouter(
        nuscenes_dir=os.path.join(os.path.dirname(__file__), "nuscenes"),
        waymo_dir=os.path.join(os.path.dirname(__file__), "waymo"),
        mode="nuscenes",
    )
    batch = router[0]
    print(f"Loaded batch for scenario {batch.scenario_id}")
    print(f"History shape: {batch.history_traj.shape}")
    print(f"Future shape:  {batch.future_traj.shape}")
    print(f"Map polylines: {len(batch.map_polylines)}")