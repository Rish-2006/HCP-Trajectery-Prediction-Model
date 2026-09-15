import torch
import torch.nn as nn
import torch.nn.functional as F


class MTRDecoder(nn.Module):
    """
    MTR Decoder.

    Uses intention point query anchors and a 3-layer MLP trajectory refinement
    head.  Optionally masks out queries using HCP-pruned candidate masks.

    Memory optimisation
    -------------------
    The original ``conf_logits.masked_fill(~hcp_mask, -1e9)`` created a
    temporary dense boolean tensor of size (B, N, n_modes) and then scattered
    -1e9 into a *copy* of conf_logits.  For large batches where hcp_mask itself
    is derived from a dense upstream computation this produced multi-GiB
    temporaries.

    The new implementation uses ``(~hcp_mask).nonzero()`` to collect only the
    indices of pruned entries (typically a small fraction of the total) and then
    performs a single **in-place** scatter — zero extra tensor allocation.

    Complexity: O(N * K * d_model + N * K * T_fut * 5) — unchanged.
      - N  = number of agents
      - K  = number of modes (intention points)
      - T_fut = future horizon (12)
    """

    def __init__(self, d_model: int = 256, n_modes: int = 6, T_fut: int = 12):
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.T_fut   = T_fut

        # Learned intention point anchors (x, y at t=T_fut)
        anchors = torch.tensor([
            [40.0,  0.0],   # Straight fast
            [15.0,  0.0],   # Straight slow
            [25.0,  15.0],  # Left turn
            [25.0, -15.0],  # Right turn
            [ 5.0,  15.0],  # Hard left
            [ 1.0,   0.0],  # Stop / slow
        ])
        self.register_buffer("intention_anchors", anchors)

        # Intention point projection → query embeddings
        self.query_proj = nn.Linear(2, d_model)

        # Decoder cross-attention: queries attend to map + agent context
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True)

        # 3-layer MLP trajectory refinement head
        # Output: T_fut * 5 (x, y, vx, vy, heading) + 1 (confidence logit)
        self.refinement_head = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, T_fut * 5 + 1),
        )

    # ------------------------------------------------------------------
    def forward(self, agent_embeds, map_embeds, hcp_mask=None):
        """
        Args:
            agent_embeds (Tensor): (B, N, d_model) — agent history embeddings
            map_embeds   (Tensor): (B, M, d_model) — fused map/agent embeddings
            hcp_mask     (Tensor, optional): (B, N, n_modes) bool.
                         True = keep, False = pruned by HCP.

        Returns:
            trajectories (Tensor): (B, N, n_modes, T_fut, 5)
            confidences  (Tensor): (B, N, n_modes)  — softmax probabilities
        """
        B, N, _ = agent_embeds.shape
        M = map_embeds.shape[1]

        # 1. Project intention point anchors → query embeddings
        query_embeds = self.query_proj(self.intention_anchors)  # (n_modes, d_model)

        # 2. Expand queries for all (B * N) instances
        queries = query_embeds.unsqueeze(0).expand(B * N, -1, -1)  # (B*N, n_modes, d_model)

        # 3. Build context tokens = map || agent, expand per instance
        context = torch.cat([map_embeds, agent_embeds], dim=1)          # (B, M+N, d_model)
        context_expanded = (context.unsqueeze(1)
                                   .expand(-1, N, -1, -1)
                                   .reshape(B * N, M + N, -1))           # (B*N, M+N, d_model)

        # 4. Cross-attention
        query_attn, _ = self.cross_attn(
            queries, context_expanded, context_expanded)                 # (B*N, n_modes, d_model)

        # 5. Combine with per-agent history context
        agent_embeds_expanded = (agent_embeds
                                 .unsqueeze(2)
                                 .expand(-1, -1, self.n_modes, -1)
                                 .reshape(B * N, self.n_modes, -1))       # (B*N, n_modes, d_model)

        combined = torch.cat([query_attn, agent_embeds_expanded], dim=-1) # (B*N, n_modes, 2*d_model)

        # 6. Refinement head → raw outputs
        outputs = self.refinement_head(combined)                          # (B*N, n_modes, T_fut*5+1)
        outputs = outputs.view(B, N, self.n_modes, -1)

        traj_out    = outputs[..., :self.T_fut * 5].view(B, N, self.n_modes, self.T_fut, 5)
        conf_logits = outputs[..., -1]                                    # (B, N, n_modes)

        # 7. Add intention anchor offsets (interpolated across timesteps)
        anchor_offsets = self.intention_anchors.view(1, 1, self.n_modes, 1, 2)
        t_factor = torch.linspace(0.0, 1.0, self.T_fut,
                                  device=agent_embeds.device).view(1, 1, 1, self.T_fut, 1)
        traj_out[..., :2] = traj_out[..., :2] + anchor_offsets * t_factor

        # 8. HCP pruning — SPARSE in-place masking (zero extra allocation)
        #    Instead of masked_fill which allocates a dense copy, we collect
        #    only the (typically sparse) pruned indices and scatter in-place.
        if hcp_mask is not None:
            pruned_indices = (~hcp_mask).nonzero(as_tuple=False)  # (num_pruned, 3)
            if pruned_indices.numel() > 0:
                # Use the active dtype's own minimum representable value instead of a
                # hardcoded -1e9 — under mixed-precision training conf_logits is cast
                # to float16 (max magnitude ~65504), and -1e9 overflows that, crashing
                # with "value cannot be converted to type at::Half without overflow".
                # finfo(dtype).min is always safe for whatever precision is active
                # (fp16, bf16, or fp32) and is still easily negative enough to drive
                # that mode's softmax probability to ~0.
                fill_value = torch.finfo(conf_logits.dtype).min
                conf_logits[
                    pruned_indices[:, 0],
                    pruned_indices[:, 1],
                    pruned_indices[:, 2],
                ] = fill_value

        confidences = F.softmax(conf_logits, dim=-1)  # (B, N, n_modes)
        return traj_out, confidences


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    decoder = MTRDecoder(d_model=256, n_modes=6)

    agent_emb = torch.randn(2, 5, 256)  # 2 scenes, 5 agents
    map_emb   = torch.randn(2, 30, 256) # 30 map tokens

    hcp_m = torch.ones((2, 5, 6), dtype=torch.bool)
    hcp_m[0, 0, 4:] = False  # prune modes 4 & 5 for agent 0 in scene 0

    trajs, confs = decoder(agent_emb, map_emb, hcp_mask=hcp_m)
    print("Decoder output:")
    print("Trajectories shape:", trajs.shape)   # (2, 5, 6, 12, 5)
    print("Confidences shape: ", confs.shape)   # (2, 5, 6)
    print("Pruned modes (should be ~0):", confs[0, 0, 4:].detach().numpy())