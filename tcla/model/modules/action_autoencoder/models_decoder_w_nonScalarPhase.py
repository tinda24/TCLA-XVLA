from typing import Optional
import math

import torch
from torch import nn

from tcla.model.modules.action_autoencoder.shared_networks import init_weights


def build_sinusoidal_pos_emb(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )

    pe = torch.zeros(length, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)

    if dim % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)


class Transformer_SequenceLang_w_nonScalarPhase(nn.Module):
    def __init__(
        self,
        action_dim: int,
        lang_dim: int,
        chunk_size: int,
        lang_len: int,
        latent_action_dim: int,
        latent_action_len: int,
        is_relative_action: bool = False,
        qpos_dim_for_relative_action: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_memory_type_emb: bool = True,
        use_learned_latent_pos: bool = True,
        use_learned_lang_pos: bool = True,
        causal_tgt_mask: bool = False,
        phase_noise_std: float = 0.02,
        phase_drop_prob: float = 0.10,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.lang_len = lang_len
        self.latent_action_dim = latent_action_dim
        self.latent_action_len = latent_action_len
        self.hidden_dim = hidden_dim
        self.causal_tgt_mask = causal_tgt_mask
        self.is_relative_action = is_relative_action

        self.phase_noise_std = phase_noise_std
        self.phase_drop_prob = phase_drop_prob

        self.latent_proj = nn.Linear(latent_action_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.phase_proj = nn.Linear(latent_action_dim, hidden_dim)

        if not is_relative_action:
            action_query_pos = build_sinusoidal_pos_emb(chunk_size, hidden_dim)
        else:
            action_query_pos = build_sinusoidal_pos_emb(chunk_size + 1, hidden_dim)
            self.qpos_proj = nn.Linear(hidden_dim, qpos_dim_for_relative_action)
        self.register_buffer("action_query_pos", action_query_pos, persistent=False)

        if use_learned_latent_pos:
            self.latent_pos_emb = nn.Parameter(
                torch.empty(1, latent_action_len, hidden_dim)
            )
        else:
            latent_pos = build_sinusoidal_pos_emb(latent_action_len, hidden_dim)
            self.register_buffer("latent_pos_emb", latent_pos, persistent=False)

        if use_learned_lang_pos:
            self.lang_pos_emb = nn.Parameter(torch.empty(1, lang_len, hidden_dim))
        else:
            lang_pos = build_sinusoidal_pos_emb(lang_len, hidden_dim)
            self.register_buffer("lang_pos_emb", lang_pos, persistent=False)

        self.use_memory_type_emb = use_memory_type_emb
        if use_memory_type_emb:
            self.memory_type_emb = nn.Parameter(torch.empty(3, hidden_dim))
        else:
            self.memory_type_emb = None

        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        if isinstance(self.latent_pos_emb, nn.Parameter):
            nn.init.trunc_normal_(self.latent_pos_emb, std=0.02)

        if isinstance(self.lang_pos_emb, nn.Parameter):
            nn.init.trunc_normal_(self.lang_pos_emb, std=0.02)

        if self.memory_type_emb is not None:
            nn.init.trunc_normal_(self.memory_type_emb, std=0.02)

        self.apply(init_weights)

    def _make_causal_mask(
        self,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _prepare_phase(
        self,
        phase: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        if phase is None:
            return torch.zeros(
                batch_size,
                self.latent_action_dim,
                device=device,
                dtype=dtype,
            )

        phase = phase.to(device=device, dtype=dtype)

        if phase.dim() == 3:
            if phase.shape[1] != 1:
                raise ValueError(
                    f"phase with 3 dims expects (B, 1, latent_action_dim), got {tuple(phase.shape)}."
                )
            phase = phase.squeeze(1)

        if phase.dim() == 1:
            if phase.shape[0] == batch_size:
                raise ValueError(
                    f"Scalar phase (B,) is not supported; expected "
                    f"(B, {self.latent_action_dim}) or (B, 1, {self.latent_action_dim}), "
                    f"got {tuple(phase.shape)}."
                )
            phase = phase.unsqueeze(0)

        if phase.dim() != 2:
            raise ValueError(
                f"phase must be (B, latent_action_dim) or (B, 1, latent_action_dim), "
                f"got {tuple(phase.shape)}."
            )

        if phase.shape[0] != batch_size:
            raise ValueError(
                f"phase batch size {phase.shape[0]} does not match batch {batch_size}."
            )

        if phase.shape[1] != self.latent_action_dim:
            raise ValueError(
                f"phase last dim must be latent_action_dim={self.latent_action_dim}, "
                f"got {phase.shape[1]}."
            )

        if self.training and self.phase_noise_std > 0:
            phase = phase + torch.randn_like(phase) * self.phase_noise_std

        if self.training and self.phase_drop_prob > 0:
            drop_mask = torch.rand(batch_size, 1, device=device) < self.phase_drop_prob
            phase = torch.where(drop_mask, torch.zeros_like(phase), phase)

        return phase

    def _prepare_lang(self, lang: torch.Tensor) -> torch.Tensor:
        if lang.dim() == 2:
            lang = lang[:, None, :]

        if lang.dim() != 3:
            raise ValueError(
                f"lang should be (B, lang_dim) or (B, lang_len, lang_dim), "
                f"but got shape {tuple(lang.shape)}."
            )

        return lang

    def _build_memory(
        self,
        latent_actions: torch.Tensor,
        lang: torch.Tensor,
        phase_vec: torch.Tensor,
    ) -> torch.Tensor:

        latent_tokens = self.latent_proj(latent_actions)

        z_len = latent_tokens.shape[1]
        if z_len > self.latent_pos_emb.shape[1]:
            raise ValueError(
                f"Input latent_action_len={z_len} exceeds initialized "
                f"latent_action_len={self.latent_pos_emb.shape[1]}."
            )

        latent_tokens = latent_tokens + self.latent_pos_emb[:, :z_len, :]

        if self.memory_type_emb is not None:
            latent_tokens = latent_tokens + self.memory_type_emb[0][None, None, :]

        lang = self._prepare_lang(lang)
        lang_tokens = self.lang_proj(lang)

        l_len = lang_tokens.shape[1]
        if l_len > self.lang_pos_emb.shape[1]:
            raise ValueError(
                f"Input lang_len={l_len} exceeds initialized "
                f"lang_len={self.lang_pos_emb.shape[1]}."
            )

        lang_tokens = lang_tokens + self.lang_pos_emb[:, :l_len, :]

        if self.memory_type_emb is not None:
            lang_tokens = lang_tokens + self.memory_type_emb[1][None, None, :]

        phase_tokens = self.phase_proj(phase_vec).unsqueeze(1)

        if self.memory_type_emb is not None:
            phase_tokens = phase_tokens + self.memory_type_emb[2][None, None, :]

        return torch.cat([latent_tokens, lang_tokens, phase_tokens], dim=1)

    def forward(
        self,
        latent_actions: torch.Tensor,
        lang: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = latent_actions.shape[0]
        device = latent_actions.device
        dtype = latent_actions.dtype

        phase_vec = self._prepare_phase(
            phase=phase,
            batch_size=B,
            device=device,
            dtype=dtype,
        )

        memory = self._build_memory(latent_actions, lang, phase_vec)
        memory = memory.to(dtype=dtype)

        x = self.action_query_pos.to(device=device, dtype=dtype)
        x = x.expand(B, -1, -1)

        tgt_mask = None
        if self.causal_tgt_mask:
            tgt_mask = self._make_causal_mask(
                length=self.chunk_size,
                device=device,
            )

        for layer in self.decoder_layers:
            x = layer(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
            )

        x = self.final_norm(x)

        if not self.is_relative_action:
            actions = self.out_proj(x)
            return actions
        else:
            qpos = x[:, 0, :]
            qpos = self.qpos_proj(qpos)
            actions = x[:, 1:, :]
            actions = self.out_proj(actions)
            return qpos, actions
