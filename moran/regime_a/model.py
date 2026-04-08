"""
Regime A Denoiser: set-based network over N particle positions.

Input: N particle positions {x_t^(1), ..., x_t^(N)} + time t
Output: predicted clean positions {x_0^(1), ..., x_0^(N)}

Uses a permutation-equivariant architecture:
  1. Per-particle embedding (position + time)
  2. Set aggregation (mean pooling -> global context)
  3. Per-particle prediction conditioned on global context

This is a simplified DeepSets / PointNet architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ParticleDenoiser(nn.Module):
    """
    Permutation-equivariant denoiser for N particles on a grid.

    Architecture:
      per-particle: (x, y, t_emb) -> h_i via MLP
      global: mean(h_i) -> context via MLP
      per-particle: (h_i, context) -> (x_0, y_0) via MLP

    The global context lets each particle's prediction depend on the
    configuration of all other particles — essential for learning
    the reverse of the Moran resampling.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        time_dim: int = 32,
        n_layers: int = 3,
        g_max: int = 127,
    ):
        super().__init__()
        self.g_max = g_max
        self.time_embed = SinusoidalEmbedding(time_dim)

        # Per-particle encoder: (x, y, t_emb) -> h
        enc_layers = []
        in_dim = 2 + time_dim
        for _ in range(n_layers):
            enc_layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*enc_layers)

        # Global context: mean(h) -> context
        ctx_layers = [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                      nn.Linear(hidden_dim, hidden_dim)]
        self.context_net = nn.Sequential(*ctx_layers)

        # Per-particle decoder: (h, context) -> (x_0, y_0)
        dec_layers = []
        in_dim = 2 * hidden_dim  # h + context
        for _ in range(n_layers):
            dec_layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        dec_layers.append(nn.Linear(hidden_dim, 2))  # predict (x, y)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, N, 2] — N particle positions per batch element
        t:   [B] — time
        returns: [B, N, 2] — predicted clean positions
        """
        B, N, _ = x_t.shape

        # Time embedding broadcast to all particles
        t_emb = self.time_embed(t)  # [B, time_dim]
        t_emb = t_emb.unsqueeze(1).expand(B, N, -1)  # [B, N, time_dim]

        # Per-particle encoding
        inp = torch.cat([x_t, t_emb], dim=-1)  # [B, N, 2 + time_dim]
        h = self.encoder(inp)  # [B, N, hidden]

        # Global context via mean pooling
        ctx = self.context_net(h.mean(dim=1))  # [B, hidden]
        ctx = ctx.unsqueeze(1).expand(B, N, -1)  # [B, N, hidden]

        # Per-particle decoding
        dec_inp = torch.cat([h, ctx], dim=-1)  # [B, N, 2*hidden]
        out = self.decoder(dec_inp)  # [B, N, 2]

        return out
