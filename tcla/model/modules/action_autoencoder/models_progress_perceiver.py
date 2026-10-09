from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDepthwiseConv1D(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, (
            "kernel_size should be odd to keep sequence length."
        )
        padding = kernel_size // 2
        self.depthwise_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:

        x = h.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return x


class FiLM(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.Dropout(dropout),
        )

        last = self.to_gamma_beta[1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        h: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if cond is None:
            return h

        gamma_beta = self.to_gamma_beta(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = gamma[:, None, :]
        beta = beta[:, None, :]

        return h * (1.0 + gamma) + beta


class SpectralBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        conv_kernel_size: int = 3,
        dropout: float = 0.0,
        use_attention: bool = True,
    ):
        super().__init__()

        self.use_attention = use_attention

        self.norm_conv = nn.LayerNorm(hidden_dim)
        self.freq_conv = FrequencyDepthwiseConv1D(
            hidden_dim=hidden_dim,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.cross_dim_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

        if use_attention:
            self.norm_attn = nn.LayerNorm(hidden_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.norm_film = nn.LayerNorm(hidden_dim)
        self.film = FiLM(
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            dropout=dropout,
        )

    def forward(
        self,
        h: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        h = h + self.freq_conv(self.norm_conv(h))

        h = h + self.cross_dim_mlp(self.norm_mlp(h))

        if self.use_attention:
            x = self.norm_attn(h)
            attn_out, _ = self.attn(
                query=x,
                key=x,
                value=x,
                need_weights=False,
            )
            h = h + attn_out

        h = h + self.film(self.norm_film(h), cond)
        return h


class FrequencyProgressPerceiver(nn.Module):
    def __init__(
        self,
        action_dim: int,
        num_freqs: int,
        latent_action_dim: int,
        lang_dim: int,
        hidden_dim: int,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        conv_kernel_size: int = 3,
        dropout: float = 0.0,
        use_attention: bool = True,
        cond_dim: int = 128,
        eps: float = 1e-6,
        band_boundaries: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
    ):
        super().__init__()

        self.action_dim = action_dim
        self.num_freqs = num_freqs
        self.hidden_dim = hidden_dim
        self.latent_action_dim = latent_action_dim
        self.depth = depth
        self.lang_dim = lang_dim
        self.cond_dim = cond_dim
        self.eps = eps
        self.band_boundaries = band_boundaries

        self.value_proj = nn.Linear(action_dim, hidden_dim)

        self.freq_embed = nn.Embedding(num_freqs, hidden_dim)

        self.band_embed = nn.Embedding(3, hidden_dim)

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.task_proj = nn.Sequential(
            nn.Linear(lang_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_norm = nn.LayerNorm(cond_dim)

        self.blocks = nn.ModuleList(
            [
                SpectralBlock(
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    use_attention=use_attention,
                )
                for _ in range(depth)
            ]
        )

        self.final_token_norm = nn.LayerNorm(hidden_dim)

        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.pool_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.pool_norm = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_action_dim),
        )

        self.post_prediction_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def get_band_ids(
        self,
        K: int,
        device: torch.device,
    ) -> torch.Tensor:
        low_ratio, mid_ratio = self.band_boundaries

        low_end = int(round(K * low_ratio))
        mid_end = int(round(K * mid_ratio))

        low_end = max(1, min(low_end, K))
        mid_end = max(low_end, min(mid_end, K))

        band_ids = torch.zeros(K, dtype=torch.long, device=device)
        band_ids[low_end:mid_end] = 1
        band_ids[mid_end:] = 2

        return band_ids

    def embed_frequency_tokens(self, C: torch.Tensor) -> torch.Tensor:
        B, K, D = C.shape

        assert D == self.action_dim, f"Expected action_dim={self.action_dim}, got D={D}"
        assert K <= self.num_freqs, (
            f"Input K={K} exceeds configured num_freqs={self.num_freqs}."
        )

        h = self.value_proj(C)

        freq_ids = torch.arange(K, device=C.device)
        band_ids = self.get_band_ids(K, C.device)

        freq_emb = self.freq_embed(freq_ids)
        band_emb = self.band_embed(band_ids)

        h = h + freq_emb[None, :, :] + band_emb[None, :, :]
        h = self.input_norm(h)
        h = self.input_dropout(h)

        return h

    def _encode_pooled(
        self,
        freq_executed_actions: torch.Tensor,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:

        B, K, D = freq_executed_actions.shape
        assert D == self.action_dim, f"Expected action_dim={self.action_dim}, got D={D}"

        h = self.embed_frequency_tokens(freq_executed_actions)
        cond = self.cond_norm(self.task_proj(task_embedding))

        for block in self.blocks:
            h = block(h=h, cond=cond)

        h = self.final_token_norm(h)
        query = self.query.expand(B, -1, -1)
        z_h, _ = self.pool_attn(
            query=query,
            key=h,
            value=h,
            need_weights=False,
            average_attn_weights=True,
        )
        return self.pool_norm(z_h)

    def get_phase_latent(
        self,
        traj_a_freq: torch.Tensor,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:

        z_h = self._encode_pooled(traj_a_freq, task_embedding)
        return self.output_proj(z_h).squeeze(1)

    def predict_progress(
        self,
        traj_a_freq: torch.Tensor,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:

        z_h = self._encode_pooled(traj_a_freq, task_embedding)
        return self.post_prediction_mlp(z_h.squeeze(1)).squeeze(-1)

    def compute_loss(
        self,
        traj_a_freq: torch.Tensor,
        traj_a_len: torch.Tensor,
        total_length: torch.Tensor,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:
        score = self.predict_progress(traj_a_freq, task_embedding)
        target = traj_a_len.to(dtype=score.dtype) / total_length.to(
            dtype=score.dtype
        ).clamp_min(1.0)
        return F.mse_loss(score, target)

    def forward(
        self,
        freq_executed_actions: torch.Tensor,
        task_embedding: torch.Tensor,
        return_attention: bool = False,
    ):
        assert freq_executed_actions.dim() == 3, (
            f"freq_executed_actions should be [B, K, action_dim], got {freq_executed_actions.shape}"
        )

        B, K, _ = freq_executed_actions.shape
        h = self.embed_frequency_tokens(freq_executed_actions)
        cond = self.cond_norm(self.task_proj(task_embedding))

        for block in self.blocks:
            h = block(h=h, cond=cond)

        h = self.final_token_norm(h)
        query = self.query.expand(B, -1, -1)
        z_h, pool_attn = self.pool_attn(
            query=query,
            key=h,
            value=h,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        z_h = self.pool_norm(z_h)
        z = self.output_proj(z_h)

        if return_attention:
            return z, pool_attn

        return z
