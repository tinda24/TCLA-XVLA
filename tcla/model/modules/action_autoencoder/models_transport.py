import math
import torch
import torch.nn as nn
import collections.abc
from functools import partial
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.squeeze(-1)

        device = x.device
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )

        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FlowNetLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.norm = nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        )

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=dropout,
        )

        self.cond_modulator = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim),
        )

        self.dim = dim
        self._init_weights()

    def _init_weights(self):

        nn.init.constant_(self.cond_modulator[-1].weight, 0)
        nn.init.constant_(self.cond_modulator[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        cond_emb: torch.Tensor,
    ) -> torch.Tensor:
        B = x.shape[0]

        features = self.cond_modulator(cond_emb).view(B, 3, self.dim)
        gamma, scale, shift = features.unbind(1)

        if x.dim() == 3:
            gamma = gamma[:, None, :]
            scale = scale[:, None, :]
            shift = shift[:, None, :]

        x_norm = self.norm(x)
        x_norm = x_norm * (1.0 + scale) + shift

        x = x + self.mlp(x_norm) * gamma

        return x


class LatentActionFlowNet_w_nonScalarPhase(nn.Module):
    def __init__(
        self,
        latent_action_dim: int,
        hidden_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_embed_dim: int = 256,
        phase_noise_std: float = 0.0,
        phase_drop_prob: float = 0.0,
    ):
        super().__init__()

        self.latent_action_dim = latent_action_dim
        self.hidden_dim = hidden_dim
        self.phase_noise_std = phase_noise_std
        self.phase_drop_prob = phase_drop_prob

        self.input_proj = nn.Linear(latent_action_dim, hidden_dim)

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )

        self.phase_encoder = nn.Sequential(
            nn.Linear(latent_action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [
                FlowNetLayer(
                    dim=hidden_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_action_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

        for layer in self.layers:
            layer._init_weights()

    def _prepare_phase(
        self,
        phase: torch.Tensor | None,
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
                    f"phase with 3 dims expects shape (B, 1, latent_action_dim), got {tuple(phase.shape)}."
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
                f"phase must be (B, latent_action_dim) or (B, 1, latent_action_dim), got {tuple(phase.shape)}."
            )

        if phase.shape[0] != batch_size:
            raise ValueError(
                f"phase batch size {phase.shape[0]} does not match x batch size {batch_size}."
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
            phase = torch.where(
                drop_mask,
                torch.zeros_like(phase),
                phase,
            )

        return phase

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        x = self.input_proj(x)

        time_emb = self.time_embed(t)

        phase_vec = self._prepare_phase(
            phase=phase,
            batch_size=B,
            device=device,
            dtype=dtype,
        )

        phase_emb = self.phase_encoder(phase_vec)

        cond_emb = time_emb + phase_emb

        for block in self.layers:
            x = block(x, cond_emb)

        x = self.norm(x)
        x = self.out_proj(x)

        return x

    def compute_loss(
        self,
        target: torch.Tensor,
        start: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        sigma: float = 0.0,
        loss_type: str = "mse",
        return_pred: bool = False,
    ):
        if start is None:
            x0 = torch.randn_like(target)
        else:
            x0 = start.to(device=target.device, dtype=target.dtype)

        x1 = target

        if x0.shape != x1.shape:
            raise ValueError(
                f"start and target should have the same shape, "
                f"but got start={tuple(x0.shape)}, target={tuple(x1.shape)}."
            )

        B = x1.shape[0]

        t = torch.rand(
            B,
            device=x1.device,
            dtype=x1.dtype,
        )

        t_view = t.view((B,) + (1,) * (x1.ndim - 1))

        xt = (1.0 - t_view) * x0 + t_view * x1

        if sigma > 0:
            xt = xt + sigma * torch.randn_like(xt)

        ut = x1 - x0

        vt = self(
            x=xt,
            t=t,
            phase=phase,
        )

        if vt.shape != ut.shape:
            raise ValueError(
                f"model output and target flow should have the same shape, "
                f"but got vt={tuple(vt.shape)}, ut={tuple(ut.shape)}."
            )

        if loss_type == "mse":
            loss = torch.mean((vt - ut) ** 2)
        elif loss_type == "l1":
            loss = torch.mean(torch.abs(vt - ut))
        else:
            raise ValueError(f"Unsupported loss_type={loss_type}. Use 'mse' or 'l1'.")

        metrics = {
            "loss": loss.item(),
            "flow_loss": loss.item(),
            "t_mean": t.mean().item(),
            "t_min": t.min().item(),
            "t_max": t.max().item(),
        }

        if return_pred:
            pred_dict = {
                "x0": x0,
                "x1": x1,
                "t": t,
                "xt": xt,
                "ut": ut,
                "vt": vt,
            }
            return loss, metrics, pred_dict

        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        shape: tuple | torch.Size | None = None,
        device: torch.device | str | None = None,
        phase: torch.Tensor | None = None,
        num_steps: int | None = None,
        start: torch.Tensor | None = None,
        return_traces: bool = False,
        dtype: torch.dtype | None = None,
    ):
        if num_steps is None:
            num_steps = getattr(self, "num_sampling_steps", 6)

        if num_steps <= 0:
            raise ValueError(f"num_steps should be positive, but got {num_steps}.")

        if start is None:
            if shape is None:
                raise ValueError("shape must be provided when start is None.")

            if device is None:
                device = next(self.parameters()).device

            if dtype is None:
                dtype = torch.float32

            x = torch.randn(
                shape,
                device=device,
                dtype=dtype,
            )
        else:
            x = start
            if device is not None:
                x = x.to(device=device)
            if dtype is not None:
                x = x.to(dtype=dtype)

        B = x.shape[0]
        dt = 1.0 / float(num_steps)

        if return_traces:
            traj_history = [x.detach().clone().cpu()]
            vel_history = [torch.zeros_like(x).detach().clone().cpu()]

        for step in range(num_steps):
            t = torch.full(
                (B,),
                fill_value=step / float(num_steps),
                device=x.device,
                dtype=x.dtype,
            )

            vt = self(
                x=x,
                t=t,
                phase=phase,
            )

            if vt.shape != x.shape:
                raise ValueError(
                    f"model output and current sample should have the same shape, "
                    f"but got vt={tuple(vt.shape)}, x={tuple(x.shape)}."
                )

            x = x + vt * dt

            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(vt.detach().clone().cpu())

        if return_traces:
            return x, (traj_history, vel_history)

        return x
