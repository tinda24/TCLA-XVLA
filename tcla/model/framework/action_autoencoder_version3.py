import copy
import torch
from torch import nn
import torch.nn.functional as F
from tcla.model.tools import FRAMEWORK_REGISTRY
from tcla.model.modules.action_autoencoder.factory import (
    build_encoder,
    build_decoder,
    build_dynamics_model,
    build_progress_model,
)
from tcla.dataloader.frequency_encoding import trajectory_to_frequency


@FRAMEWORK_REGISTRY.register("ActionAutoencoder_version3")
class ActionAutoencoder(nn.Module):
    def __init__(
        self,
        encoder_type,
        decoder_type,
        dynamics_model_type,
        progress_model_type,
        is_relative_action,
        qpos_dim_for_relative_action,
        qpos_key,
        action_dim,
        chunk_size,
        latent_action_dim,
        downsample_rate,
        lang_len,
        lang_dim,
        num_freqs,
        fft_type: str = "dct",
        action_key="action",
        lang_key="task_emb",
        phase_key="phase_label",
        dynamics_loss_coef=1.0,
        recon_loss_coef=1.0,
        progress_loss_coef=1.0,
        use_ema_target_encoder=True,
        ema_momentum_start=0.996,
        ema_momentum_end=1.0,
        device="cuda",
    ):
        super().__init__()

        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_action_dim = latent_action_dim
        self.downsample_rate = downsample_rate
        self.lang_len = lang_len
        self.lang_dim = lang_dim
        self.num_freqs = num_freqs
        self.fft_type = str(fft_type)
        self.sequence_length = chunk_size
        self.action_latent_tokens = chunk_size // downsample_rate
        self.latent_dim = latent_action_dim

        self.action_key = action_key
        self.lang_key = lang_key
        self.phase_key = phase_key
        self.qpos_key = qpos_key

        self.dynamics_loss_coef = dynamics_loss_coef
        self.recon_loss_coef = recon_loss_coef
        self.progress_loss_coef = progress_loss_coef
        self.use_ema_target_encoder = use_ema_target_encoder
        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end

        self.is_relative_action = is_relative_action
        self.qpos_dim_for_relative_action = qpos_dim_for_relative_action
        self.qpos_key = qpos_key

        self.encoder = build_encoder(
            encoder_type,
            action_dim,
            chunk_size,
            lang_len,
            lang_dim,
            latent_action_dim,
            downsample_rate,
            is_relative_action,
            qpos_dim_for_relative_action,
        )
        self.decoder = build_decoder(
            decoder_type,
            action_dim,
            chunk_size,
            lang_len,
            lang_dim,
            latent_action_dim,
            downsample_rate,
            is_relative_action,
            qpos_dim_for_relative_action,
            num_freqs,
        )
        self.dynamics_model = build_dynamics_model(
            dynamics_model_type,
            latent_action_dim,
            latent_action_len=self.action_latent_tokens,
        )
        self.progress_model = build_progress_model(
            progress_model_type, num_freqs, action_dim, latent_action_dim, lang_dim
        )
        self.initial_traj_a_freq = nn.Parameter(
            torch.zeros(num_freqs, action_dim, dtype=torch.float32)
        )

        if self.use_ema_target_encoder:
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_encoder.requires_grad_(False)
        else:
            self.target_encoder = None

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.dynamics_model_type = dynamics_model_type
        self.progress_model_type = progress_model_type

        self.to(device)

    def _decoder_uses_progress(self) -> bool:
        return True

    def _dynamics_uses_vector_phase(self) -> bool:
        return True

    def decode_latent(
        self,
        latent: torch.Tensor,
        lang: torch.Tensor,
        phase: torch.Tensor | None = None,
        raw_freq_action: torch.Tensor | None = None,
        traj_a_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._decoder_uses_progress():
            return self.decoder(latent, lang)

        phase_latent = self._resolve_vector_phase(
            phase, raw_freq_action, lang=lang, traj_a_len=traj_a_len
        )
        return self.decoder(latent_actions=latent, lang=lang, phase=phase_latent)

    def _resolve_vector_phase(
        self,
        phase: torch.Tensor | None,
        raw_freq_action: torch.Tensor | None,
        lang: torch.Tensor | None = None,
        traj_a_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if phase is not None:
            return phase
        return self.phase_from_freq(
            raw_freq_action, task_emb=lang, traj_a_len=traj_a_len
        )

    def _prepare_traj_a_freq(self, encoded_trajectory: torch.Tensor) -> torch.Tensor:
        if encoded_trajectory.ndim == 2:
            return encoded_trajectory.unsqueeze(0)
        if encoded_trajectory.ndim == 3:
            return encoded_trajectory
        raise ValueError(
            f"traj_a_freq must be [num_freqs, action_dim] or [B, num_freqs, action_dim], "
            f"got {tuple(encoded_trajectory.shape)}."
        )

    def resolve_traj_a_freq(
        self,
        encoded_trajectory: torch.Tensor | None,
        *,
        traj_a_len: torch.Tensor | None = None,
        batch_size: int | None = None,
        device=None,
        dtype=None,
    ) -> torch.Tensor:

        if device is None:
            device = self.initial_traj_a_freq.device
        if dtype is None:
            dtype = self.initial_traj_a_freq.dtype
        initial = self.initial_traj_a_freq.to(device=device, dtype=dtype)
        if encoded_trajectory is None:
            if batch_size is None:
                return initial.unsqueeze(0)
            return initial.unsqueeze(0).expand(int(batch_size), -1, -1)

        freq = self._prepare_traj_a_freq(encoded_trajectory).to(
            device=device, dtype=dtype
        )
        if batch_size is not None and freq.shape[0] == 1 and int(batch_size) > 1:
            freq = freq.expand(int(batch_size), -1, -1)
        if traj_a_len is not None:
            first_step = traj_a_len.to(device=device).reshape(-1) <= 0
            if first_step.any():
                if freq.shape[0] != first_step.shape[0]:
                    raise ValueError(
                        f"traj_a_freq batch ({freq.shape[0]}) and traj_a_len batch ({first_step.shape[0]}) mismatch."
                    )
                freq = freq.clone()
                freq[first_step] = initial
        return freq

    def phase_from_freq(
        self,
        encoded_trajectory: torch.Tensor | None,
        *,
        task_emb: torch.Tensor | None = None,
        traj_a_len: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:

        dtype = next(self.progress_model.parameters()).dtype
        freq = self.resolve_traj_a_freq(
            encoded_trajectory,
            traj_a_len=traj_a_len,
            batch_size=batch_size,
            device=self.initial_traj_a_freq.device,
            dtype=dtype,
        )

        if hasattr(self.progress_model, "get_phase_latent"):
            if task_emb is None:
                raise ValueError(
                    "task_emb is required for the perceiver progress model."
                )
            task_embedding = task_emb
            if task_embedding.ndim == 3 and task_embedding.shape[1] == 1:
                task_embedding = task_embedding.squeeze(1)
            elif task_embedding.ndim == 1:
                task_embedding = task_embedding.unsqueeze(0)
            return self.progress_model.get_phase_latent(freq, task_embedding)
        return self.progress_model(freq)

    def phase_from_batch(self, data: dict) -> torch.Tensor:

        traj_a_len = data.get("traj_a_len")
        return self.phase_from_freq(
            data.get("traj_a_freq"),
            task_emb=data.get(self.lang_key),
            traj_a_len=traj_a_len,
        )

    def progress_loss_from_data(self, data: dict) -> torch.Tensor:
        if float(self.progress_loss_coef) == 0.0:
            return (
                data["traj_a_len"]
                .to(device=self.initial_traj_a_freq.device, dtype=torch.float32)
                .sum()
                * 0.0
            )
        traj_a_freq = self.resolve_traj_a_freq(
            data.get("traj_a_freq"),
            traj_a_len=data.get("traj_a_len"),
            batch_size=data["traj_a_len"].shape[0],
            device=self.initial_traj_a_freq.device,
            dtype=next(self.progress_model.parameters()).dtype,
        )
        return self.progress_model.compute_loss(
            traj_a_freq,
            data["traj_a_len"],
            data["total_length"],
            data[self.lang_key],
        )

    def forward(self, batch):

        data = self.prepare_data(batch)
        actions = data[self.action_key]
        next_actions = data.get(f"next_{self.action_key}")
        task_emb = data.get(self.lang_key)
        if self.is_relative_action:
            qpos = data.get(self.qpos_key)
            next_qpos = data.get(f"next_{self.qpos_key}")

        progress_loss = self.progress_loss_from_data(data)
        needs_phase = (
            self._decoder_uses_progress() or self._dynamics_uses_vector_phase()
        )

        phase = self.phase_from_batch(data) if needs_phase else None

        decoder_phase = phase if self._decoder_uses_progress() else None

        latent = (
            self.encoder(actions, task_emb)
            if not self.is_relative_action
            else self.encoder(actions, task_emb, qpos)
        )
        target_latent = (
            self.encode_ema_target(next_actions, task_emb)
            if not self.is_relative_action
            else self.encode_ema_target(next_actions, task_emb, next_qpos)
        )
        if self._dynamics_uses_vector_phase():
            if phase is None:
                raise ValueError("Vector-phase dynamics requires a non-null phase.")
            if phase.dim() != 2 or phase.shape[-1] != self.latent_action_dim:
                raise ValueError(
                    f"direct_flow_w_nonscalarphase expects phase (B, {self.latent_action_dim}), "
                    f"got {tuple(phase.shape)}."
                )

        pred_loss, pred_metrics = self.dynamics_model.compute_loss(
            latent, target_latent, phase
        )

        recon_actions = self.decode_latent(
            latent,
            task_emb,
            phase=decoder_phase,
            raw_freq_action=data.get("traj_a_freq"),
            traj_a_len=data.get("traj_a_len"),
        )
        recon_loss = F.mse_loss(recon_actions, actions)

        total_loss = (
            self.recon_loss_coef * recon_loss.mean()
            + self.dynamics_loss_coef * pred_loss.mean()
            + self.progress_loss_coef * progress_loss.mean()
        )

        metrics = {
            "total_loss": total_loss,
            "pred_loss": pred_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "progress_loss": progress_loss.detach(),
        }
        return metrics

    def get_ema_momentum(self, train_step, num_train_steps):
        if num_train_steps <= 1:
            return self.ema_momentum_end
        progress = min(max(train_step / (num_train_steps - 1), 0.0), 1.0)
        return self.ema_momentum_start + progress * (
            self.ema_momentum_end - self.ema_momentum_start
        )

    @torch.no_grad()
    def update_target_encoder(self, train_step, num_train_steps):
        if not self.use_ema_target_encoder:
            return 0.0

        momentum = self.get_ema_momentum(train_step, num_train_steps)
        for online_param, target_param in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.mul_(momentum).add_(
                online_param.data, alpha=1.0 - momentum
            )

        for online_buffer, target_buffer in zip(
            self.encoder.buffers(), self.target_encoder.buffers()
        ):
            target_buffer.copy_(online_buffer)

        return momentum

    @torch.no_grad()
    def encode_ema_target(self, action, lang, qpos=None):
        target_encoder = (
            self.target_encoder if self.use_ema_target_encoder else self.encoder
        )
        if self.use_ema_target_encoder:
            target_encoder.eval()
        if not self.is_relative_action:
            return target_encoder(action, lang)
        else:
            return target_encoder(action, lang, qpos)

    def prepare_data(self, examples):
        keys = examples[0].keys()
        return {
            key: self._stack_values([example[key] for example in examples])
            for key in keys
        }

    def _stack_values(self, values):
        first = values[0]
        if torch.is_tensor(first):
            return torch.stack(values, dim=0).to(self.device)
        if isinstance(first, str):
            return list(values)
        return torch.as_tensor(values, device=self.device)

    @torch.no_grad()
    def predict_progress_from_trajectory(
        self,
        trajectory: torch.Tensor,
        task_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            if trajectory.ndim == 3:
                if task_emb is None:
                    raise ValueError("task_emb is required when trajectory is batched.")
                return torch.stack(
                    [
                        self.predict_progress_from_trajectory(
                            trajectory[i], task_emb[i]
                        ).squeeze(0)
                        for i in range(trajectory.shape[0])
                    ],
                    dim=0,
                )
            if task_emb is None:
                raise ValueError(
                    "task_emb is required for version3 perceiver progress prediction."
                )
            freq = self.trajectory_to_frequency(trajectory).unsqueeze(0)
            task_batch = task_emb.unsqueeze(0) if task_emb.ndim == 1 else task_emb
            return self.progress_model.predict_progress(freq, task_batch).squeeze(0)
        finally:
            if was_training:
                self.train()

    def trajectory_to_frequency(self, trajectory: torch.Tensor) -> torch.Tensor:
        return trajectory_to_frequency(
            trajectory,
            num_freqs=self.num_freqs,
            fft_type=self.fft_type,
        )

    @property
    def device(self):
        return next(self.parameters()).device
