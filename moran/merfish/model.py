"""
DeepSets denoiser for MERFISH Moran spatial deconvolution.

Architecture:
  1. Per-cell encoder:  (x_norm, t_emb, noise) → h_i   [N, hidden]
  2. Global context:   mean(h) over real cells → MLP → c  [hidden]
  3. Per-cell decoder: (h_i, c) → x_hat_i              [N, G]

Input:
  x_t:  [B, N, G]  noisy expression (padded)
  t:    [B]         time
  mask: [B, N]      bool (True = real cell, False = padding)
  noise: [B, N, noise_dim] optional

Output: [B, N, G]  predicted clean expression x_hat_0
        (non-negative, enforced by Softplus)

Loss: symmetric Chamfer distance between prediction and target
      (permutation-invariant, handles cell ordering ambiguity)

MPS compatibility notes:
  - All operations are standard PyTorch, no CuPy or CUDA-specific code
  - float32 throughout (MPS does not support float64)
  - torch.multinomial used in sampler (MPS compatible)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MerfishDenoiser(nn.Module):
    """
    Permutation-equivariant denoiser for a population of N cells.

    G: number of genes (649 for Vizgen MERFISH)
    """

    def __init__(
        self,
        G: int = 649,
        hidden_dim: int = 512,
        time_dim: int = 64,
        noise_dim: int = 32,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        n_ctx_layers: int = 2,
        m_samples: int = 4,
        log_input: bool = True,
        use_coords: bool = True,
    ):
        super().__init__()
        self.G = G
        self.noise_dim = noise_dim
        self.m_samples = m_samples
        self.time_dim = time_dim
        self.log_input = log_input  # use log1p(x) for numerical stability
        self.use_coords = use_coords

        # Per-cell encoder
        enc_in = G + time_dim + noise_dim
        if self.use_coords:
            enc_in += 2  # +2 for (x,y) coords
        enc = []
        for i in range(n_enc_layers):
            enc.extend([
                nn.Linear(enc_in if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
        self.encoder = nn.Sequential(*enc)

        # Global context: mean-pool → MLP
        ctx = []
        for i in range(n_ctx_layers):
            ctx.extend([
                nn.Linear(hidden_dim if i == 0 else hidden_dim, hidden_dim),
                nn.SiLU(),
            ])
        self.context_net = nn.Sequential(*ctx)

        # Per-cell decoder: (h_i ‖ context) → x̂_0_i
        dec_in = hidden_dim * 2
        dec = []
        for i in range(n_dec_layers):
            dec.extend([
                nn.Linear(dec_in if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
        dec.append(nn.Linear(hidden_dim, G))
        self.decoder = nn.Sequential(*dec)

        # Softplus output: ensures non-negative predicted counts
        self.out_act = nn.Softplus()

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding. t: [B] → [B, time_dim]"""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float() * freqs          # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, time_dim]

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x_t:  [B, N, G]  noisy expression (padded with zeros)
        t:    [B]         time values in [0, 1]
        mask: [B, N]      True = real cell
        coords: [B, N, 2] spatial coordinates
        noise: [B, N, noise_dim] or None (sampled fresh)

        Returns: [B, N, G] predicted clean expression
        """
        B, N, G = x_t.shape
        device = x_t.device

        if noise is None:
            noise = torch.randn(B, N, self.noise_dim, device=device)

        # Normalise input counts (log1p for stability)
        if self.log_input:
            x_norm = torch.log1p(x_t.float())
        else:
            x_norm = x_t.float() / 100.0   # rough scale

        # Time embedding: [B, time_dim] → [B, N, time_dim]
        t_emb = self._time_embed(t).unsqueeze(1).expand(B, N, -1)

        # Per-cell encoding
        if self.use_coords:
            # Normalize coordinates roughly to [0, 1] scale (MERFISH coords are ~0-1000 µm)
            coords_norm = coords.float() / 1000.0
            h = self.encoder(torch.cat([x_norm, coords_norm, t_emb, noise], dim=-1))  # [B, N, H]
        else:
            h = self.encoder(torch.cat([x_norm, t_emb, noise], dim=-1))  # [B, N, H]

        # Masked mean-pool: only average over real cells
        mask_f = mask.float().unsqueeze(-1)          # [B, N, 1]
        h_sum = (h * mask_f).sum(dim=1)              # [B, H]
        n_real = mask_f.sum(dim=1).clamp(min=1)      # [B, 1]
        h_mean = h_sum / n_real                       # [B, H]

        # Global context
        ctx = self.context_net(h_mean)                # [B, H]
        ctx = ctx.unsqueeze(1).expand(B, N, -1)       # [B, N, H]

        # Per-cell decoding
        out = self.decoder(torch.cat([h, ctx], dim=-1))   # [B, N, G]
        out = self.out_act(out)                            # non-negative

        # Zero out padded cells (they don't contribute to loss)
        out = out * mask.float().unsqueeze(-1)
        return out

    def sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Single forward pass (no noise averaging). Returns [B, N, G]."""
        return self.forward(x_t, t, mask, coords=coords)

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        X_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combined loss:
          1. Chamfer distance between pred and target (permutation-invariant)
          2. Aggregation consistency: |Σ_i pred_i - X_0| (soft constraint)

        pred:   [B, N, G]
        target: [B, N, G]  (clean expression, padded)
        mask:   [B, N]     True = real cell
        X_0:    [B, G]     bulk expression (sum constraint)
        """
        B, N, G = pred.shape

        # --- 1. Chamfer distance (over real cells only) ---
        # MPS does not support torch.cdist backward; compute manually.
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 <a, b>
        p = pred.float()   # [B, N, G]
        t = target.float() # [B, N, G]
        p_sq = (p ** 2).sum(-1, keepdim=True)          # [B, N, 1]
        t_sq = (t ** 2).sum(-1, keepdim=True)          # [B, N, 1]
        cross = torch.bmm(p, t.transpose(1, 2))        # [B, N, N]
        dist_sq = p_sq + t_sq.transpose(1, 2) - 2 * cross
        dist = dist_sq.clamp(min=0)  # True L^2 squared Chamfer (no sqrt)

        # mask out padded rows/cols
        mask_f = mask.float()  # [B, N]
        row_mask = mask_f.unsqueeze(2)   # [B, N, 1]
        col_mask = mask_f.unsqueeze(1)   # [B, 1, N]
        large = 1e12  # squared domain needs larger mask
        dist_masked = dist + (1 - row_mask) * large + (1 - col_mask) * large

        # min over target per prediction
        min_p2t_vals, min_p2t_idx = dist_masked.min(dim=2)  # [B, N]
        min_pred_to_tgt = min_p2t_vals
        # min over prediction per target
        min_tgt_to_pred = dist_masked.min(dim=1)[0]  # [B, N]

        # Average only over real cells
        n_real = mask_f.sum(dim=1).clamp(min=1)      # [B]
        chamfer_p2t = (min_pred_to_tgt * mask_f).sum(dim=1) / n_real   # [B]
        chamfer_t2p = (min_tgt_to_pred * mask_f).sum(dim=1) / n_real   # [B]
        chamfer_loss = (chamfer_p2t + chamfer_t2p).mean() / 2.0

        # --- Option B: Zero-Match Penalty ---
        # Get the corresponding matched target cell profile for each predicted cell
        idx_expanded = min_p2t_idx.unsqueeze(-1).expand(-1, -1, G)
        matched_target = torch.gather(target, dim=1, index=idx_expanded)
        
        # Find genes where the biological target is exactly 0
        zero_mask = (matched_target == 0.0).float()
        
        # Mean prediction value specifically on structural zeros
        # mask_f ensures we only compute over valid cells
        zero_error = (pred * zero_mask * mask_f.unsqueeze(-1)).sum() / (mask_f.sum() * G + 1e-8)
        lambda_zero = 5.0  # Scalar parameter to heavily discourage smearing

        # --- 2. Aggregation consistency ---
        # pred_sum = Σ_i pred_i  [B, G]
        pred_sum = (pred * mask_f.unsqueeze(-1)).sum(dim=1)
        agg_loss = F.mse_loss(pred_sum, X_0.float()) / (G + 1e-6)

        return chamfer_loss + 0.1 * agg_loss + lambda_zero * zero_error


def ipf_rescale(
    x_pred: torch.Tensor,
    X_0: torch.Tensor,
    mask: torch.Tensor,
    n_iter: int = 30,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Iterative Proportional Fitting to enforce Σ_i x_i = X_0.

    x_pred: [B, N, G] float  model predictions
    X_0:    [B, G] float     bulk expression targets
    mask:   [B, N] bool

    Returns: [B, N, G] float rescaled so that masked sum = X_0
    """
    mask_f = mask.float().unsqueeze(-1)   # [B, N, 1]
    # Start from a small floor (eps) to avoid degenerate zeros
    y = (x_pred.clamp(min=0) + eps) * mask_f

    for _ in range(n_iter):
        y_sum = (y * mask_f).sum(dim=1, keepdim=True).clamp(min=eps)  # [B, 1, G]
        target = X_0.float().unsqueeze(1).clamp(min=0)                 # [B, 1, G]
        scale = target / y_sum
        # Guard against inf/nan scaling
        scale = scale.nan_to_num(nan=1.0, posinf=1.0, neginf=0.0)
        y = y * scale
        y = y * mask_f

    # Final NaN clean
    y = y.nan_to_num(0.0).clamp(min=0)
    return y


def randomised_round(
    y: torch.Tensor,
    X_0: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Stochastic Integer Projection (Symmetry Breaking).
    Unlike deterministic fractional rounding, this explicitly samples from
    the Multinomial distribution to preserve correct generative variance
    and break positional symmetries dynamically based on the network's normalized expectations.
    """
    B, N, G = y.shape
    result = torch.zeros_like(y, dtype=torch.long)

    for b in range(B):
        n_real = int(mask[b].sum().item())
        if n_real == 0:
            continue
            
        # Normalize the network's continuous predictions into categorical probabilities
        p = y[b, :n_real] / y[b, :n_real].sum(dim=0, keepdim=True).clamp(min=1e-8)  # [n_real, G]
        
        for g in range(G):
            total_g = int(X_0[b, g].item())
            if total_g > 0:
                # Stochastically allocate all molecules generating true expression variance
                samples = torch.distributions.Multinomial(total_count=total_g, probs=p[:, g]).sample()
                result[b, :n_real, g] = samples.long()

    return result
