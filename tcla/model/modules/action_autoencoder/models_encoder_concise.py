from typing import Optional

import torch
from torch import nn
from einops.layers.torch import Rearrange
from tcla.model.modules.action_autoencoder.shared_networks import init_weights


class CausalConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation, stride, no_pad=False
    ):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if no_pad:
            self.padding = 0
        else:
            self.padding = dilation * (kernel_size - 1)
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            stride=stride,
        )

    def forward(self, x):
        x = self.conv(x)
        last_n = (2 * self.padding - self.kernel_size) // self.stride + 1
        if last_n > 0:
            return x[:, :, :-last_n]
        else:
            return x


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size,
        stride,
        n_groups=4,
        causal=True,
        no_pad=False,
    ):
        super().__init__()
        if causal:
            conv = CausalConv1d(
                inp_channels,
                out_channels,
                kernel_size,
                dilation=1,
                stride=stride,
                no_pad=no_pad,
            )
        else:
            conv = nn.Conv1d(
                inp_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
            )

        self.block = nn.Sequential(
            conv,
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size=[5, 3],
        stride=[2, 2],
        n_groups=8,
        causal=True,
        residual=False,
        pooling_layers=[],
    ):
        super().__init__()
        self.pooling_layers = pooling_layers
        self.blocks = nn.ModuleList()
        for i in range(len(kernel_size)):
            block = Conv1dBlock(
                inp_channels if i == 0 else out_channels,
                out_channels,
                kernel_size[i],
                stride[i],
                n_groups=n_groups,
                causal=causal,
            )
            self.blocks.append(block)
        if residual:
            if out_channels == inp_channels and stride[0] == 1:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv1d(
                    inp_channels, out_channels, kernel_size=1, stride=sum(stride)
                )
        if pooling_layers:
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, input_dict):
        x = input_dict
        x = torch.transpose(x, 1, 2)
        out = x
        layer_num = 0
        for block in self.blocks:
            out = block(out)
            if hasattr(self, "pooling"):
                if layer_num in self.pooling_layers:
                    out = self.pooling(out)
            layer_num += 1
        if hasattr(self, "residual_conv"):
            out = out + self.residual_conv(x)
        return torch.transpose(out, 1, 2)


class Transformer_CNN(nn.Module):
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
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.is_relative_action = is_relative_action
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        if is_relative_action:
            self.qpos_proj = nn.Linear(qpos_dim_for_relative_action, hidden_dim)

        max_len = lang_len + chunk_size + latent_action_len
        self.pos_emb = nn.Parameter(torch.empty(1, max_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.action_conv_block = ResidualTemporalBlock(
            hidden_dim,
            hidden_dim,
            kernel_size=[5, 3, 3],
            stride=[2, 2, 1],
            causal=True,
            residual=False,
            pooling_layers=[],
        )
        self.action_conv_block_out_proj = nn.Linear(hidden_dim, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.out_proj = nn.Linear(hidden_dim, latent_action_dim)
        self.apply(init_weights)

        self.latent_action_len = latent_action_len

    def forward(
        self,
        actions: torch.Tensor,
        lang: torch.Tensor,
        qpos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        lang_tokens = self.lang_proj(lang)

        action_tokens = self.action_proj(actions)
        action_tokens = self.action_conv_block(action_tokens)
        action_tokens = self.action_conv_block_out_proj(action_tokens)

        assert action_tokens.shape[1] == self.latent_action_len, (
            "action_tokens.shape[1] != latent_action_len"
        )

        if self.is_relative_action:
            assert qpos is not None, "qpos is required when is_relative_action is True"
            qpos_tokens = self.qpos_proj(qpos)
            tokens = torch.cat([lang_tokens, qpos_tokens, action_tokens], dim=1)
        else:
            if lang.ndim == 2:
                lang_tokens = lang_tokens.unsqueeze(1)
            tokens = torch.cat([lang_tokens, action_tokens], dim=1)

        tokens = tokens + self.pos_emb[:, : tokens.shape[1]]

        encoded = self.encoder(tokens)
        return self.out_proj(encoded[:, -self.latent_action_len :, :])
