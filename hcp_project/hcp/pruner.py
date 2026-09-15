import torch
import torch.nn as nn
import math
import time
try:
    from hcp.kff import KinematicFeasibilityFilter
    from hcp.srf import SpatialReachabilityFilter
    from hcp.scf import SocialCompatibilityFilter
except ModuleNotFoundError:
    from hcp_project.hcp.kff import KinematicFeasibilityFilter
    from hcp_project.hcp.srf import SpatialReachabilityFilter
    from hcp_project.hcp.scf import SocialCompatibilityFilter


def generate_kinematic_candidates(hist, T_fut=12, dt=0.5, K=6):
    """
    Generates K candidate future trajectories using ONLY the agent's own
    recent history (position, velocity, heading) — no ground truth involved
    anywhere, so this is exactly what a real deployed system could compute
    at inference time, before the true future is known.

    Each candidate extrapolates forward at constant speed under a different
    constant turn rate, covering straight-ahead, gentle/moderate left and
    right turns — a standard "anchor trajectory" bank, not a search over an
    enormous space, so it stays cheap.

    Args:
        hist: (T_hist, 6) tensor — [x, y, vx, vy, heading, _] per past step.
              Only the most recent step is used as the extrapolation origin.
        T_fut: number of future steps to generate.
        dt: seconds per step (nuScenes keyframes are 2Hz -> 0.5s).
        K: number of candidates to generate.
    Returns:
        (K, T_fut, 5) tensor, channel layout matching ground truth:
        [x, y, vx, vy, heading].
    """
    last = hist[-1]
    x0, y0 = last[0].item(), last[1].item()
    vx0, vy0 = last[2].item(), last[3].item()
    heading0 = last[4].item()
    speed = math.hypot(vx0, vy0)

    # Turn rates in rad/s: straight, gentle left/right, moderate left/right,
    # sharper turn — a spread that's plausible for a single 0.5s-sampled step
    # without being physically absurd.
    turn_rate_bank = [0.0, 0.12, -0.12, 0.25, -0.25, 0.40]
    turn_rates = turn_rate_bank[:K]
    while len(turn_rates) < K:  # if K > len(bank), repeat the straight option
        turn_rates.append(0.0)

    candidates = torch.zeros(K, T_fut, 5, dtype=hist.dtype, device=hist.device)
    for k, omega in enumerate(turn_rates):
        x, y, heading = x0, y0, heading0
        for t in range(T_fut):
            heading = heading + omega * dt
            x = x + speed * dt * math.cos(heading)
            y = y + speed * dt * math.sin(heading)
            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)
            candidates[k, t, 0] = x
            candidates[k, t, 1] = y
            candidates[k, t, 2] = vx
            candidates[k, t, 3] = vy
            candidates[k, t, 4] = heading
    return candidates


class HierarchicalCombinatorialPruner(nn.Module):
    """
    Hierarchical Combinatorial Pruning (HCP) module.
    Sequentially filters trajectory candidates using KFF, SRF, and SCF.
    
    Architecture:
      Dense candidates (N_agents × K_modes × T_steps) 
            ↓
       [KFF] Kinematic Feasibility Filter (Stage 1)
            ↓
       [SRF] Spatial Reachability Filter (Stage 2)
            ↓
       [SCF] Social Compatibility Filter (Stage 3)
            ↓
      Sparse candidates
      
    Complexity: O(Stage 1) + O(Stage 2) + O(Stage 3)
    """
    def __init__(self, collision_threshold=1.5, curvature_max=0.2, jerk_max=5.0, lat_acc_max=4.0):
        super().__init__()
        self.kff = KinematicFeasibilityFilter(kappa_max=curvature_max, j_max=jerk_max, a_lat_max=lat_acc_max)
        self.srf = SpatialReachabilityFilter(collision_threshold=collision_threshold)
        self.scf = SocialCompatibilityFilter(collision_threshold=collision_threshold)
        
    def forward(self, trajectories, history_traj, map_polylines, drivable_area=None):
        """
        Args:
            trajectories (Tensor): Shape (N, K, T, 5) representing dense trajectory candidates.
            history_traj (Tensor): Shape (N, T_hist, 6) representing agent histories.
            map_polylines (list): Road geometry representation.
            drivable_area (Polygon, optional): Shapely Polygon of drivable area.
        Returns:
            sparse_trajectories (list of Tensors): List of length N, each element containing the survived trajectories.
            stats (dict): Performance metrics of the pruning process (latency, pruning ratio, etc.)
        """
        N, K, T, _ = trajectories.shape
        start_time = time.perf_counter()
        
        # --- STAGE 1: Kinematic Feasibility Filter ---
        t0 = time.perf_counter()
        kff_mask = self.kff(trajectories) # (N, K)
        t1 = time.perf_counter()
        kff_time = t1 - t0
        
        # Apply KFF mask to trajectories for SRF (keep shape but set false items as zero or manage via index)
        # For simplicity and efficiency, filters can be evaluated in series using the masks.
        
        # --- STAGE 2: Spatial Reachability Filter ---
        t0 = time.perf_counter()
        srf_mask = self.srf(trajectories, map_polylines, drivable_area) # (N, K)
        t1 = time.perf_counter()
        srf_time = t1 - t0
        
        # --- STAGE 3: Social Compatibility Filter ---
        t0 = time.perf_counter()
        scf_mask = self.scf(trajectories, history_traj) # (N, K)
        t1 = time.perf_counter()
        scf_time = t1 - t0
        
        # Composite mask
        composite_mask = kff_mask & srf_mask & scf_mask # (N, K)
        
        # Ensure at least one trajectory survives per agent (if all pruned, fall back to best confidence mode 0)
        for i in range(N):
            if not torch.any(composite_mask[i]):
                composite_mask[i, 0] = True
                
        # Build list of sparse trajectories
        sparse_trajectories = []
        for i in range(N):
            survived_indices = torch.where(composite_mask[i])[0]
            sparse_trajectories.append(trajectories[i, survived_indices])
            
        total_time = time.perf_counter() - start_time
        
        # Calculate statistics
        total_candidates = N * K
        kff_survived = int(kff_mask.sum())
        srf_survived = int((kff_mask & srf_mask).sum())
        scf_survived = int(composite_mask.sum())
        
        pruning_ratio = 1.0 - (scf_survived / total_candidates)

        stats = {
            "total_time_ms": total_time * 1000.0,
            "kff_time_ms": kff_time * 1000.0,
            "srf_time_ms": srf_time * 1000.0,
            "scf_time_ms": scf_time * 1000.0,
            "raw_count": total_candidates,
            "kff_count": kff_survived,
            "srf_count": srf_survived,
            "scf_count": scf_survived,
            "pruning_ratio": pruning_ratio,
            # NOTE: this pruner stage's own wall-clock cost is measured above
            # (total_time_ms). What it does NOT measure is whether pruning
            # actually reduces the downstream transformer's compute — in the
            # current architecture the decoder always processes all K modes
            # regardless of this mask, so there is currently no real latency
            # reduction to report here. Earlier versions of this dict reported
            # hardcoded placeholder values for "latency_reduction_pct" and
            # "accuracy_retention_pct" that were never actually measured —
            # removed rather than continuing to report fabricated numbers.
        }
        
        return sparse_trajectories, composite_mask, stats

if __name__ == "__main__":
    pruner = HierarchicalCombinatorialPruner()
    
    # 2 agents, 64 candidates, 12 steps
    traj = torch.zeros((2, 64, 12, 5))
    hist = torch.zeros((2, 5, 6))
    
    # Ego (agent 0) at (0,0) and Front car (agent 1) at (10,0)
    hist[0, -1, :2] = torch.tensor([0.0, 0.0])
    hist[1, -1, :2] = torch.tensor([10.0, 0.0])
    
    # Generate some straight lines and some crazy lines
    for k in range(64):
        for t in range(12):
            if k < 20:
                # Straight (feasible)
                traj[0, k, t, :2] = torch.tensor([t * 1.5, 0.0])
                traj[1, k, t, :2] = torch.tensor([10.0 + t * 0.5, 0.0])
            elif k < 40:
                # Violates curvature (infeasible KFF)
                traj[0, k, t, :2] = torch.tensor([t * 1.5, 10.0 * torch.sin(torch.tensor(t * 1.5))])
                traj[1, k, t, :2] = torch.tensor([10.0 + t * 0.5, 5.0 * torch.sin(torch.tensor(t * 2.0))])
            else:
                # Collides (infeasible SCF)
                traj[0, k, t, :2] = torch.tensor([t * 2.5, 0.0]) # Ego moves fast and hits front car
                traj[1, k, t, :2] = torch.tensor([10.0 + t * 0.5, 0.0])
                
    map_polylines = [np.array([[-50.0, 0.0, 1.0], [50.0, 0.0, 1.0]])]
    
    sparse_trajs, mask, stats = pruner(traj, hist, map_polylines)
    
    print("Pruner evaluation stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")