import torch
from torch import nn

from models.latent_action_ae import FrozenTCLAActionAE


class TrainableTCLAActionAE(nn.Module):
    _prepare_lang = staticmethod(FrozenTCLAActionAE._prepare_lang)
    _prepare_task_embedding = staticmethod(FrozenTCLAActionAE._prepare_task_embedding)
    decode = FrozenTCLAActionAE.decode

    def __init__(self, config_yaml, ckpt_path, device="cuda"):
        super().__init__()
        frozen = FrozenTCLAActionAE(config_yaml, ckpt_path, device)
        self.model = frozen.model
        for name in (
            "action_dim",
            "chunk_size",
            "latent_action_dim",
            "downsample_rate",
            "latent_action_tokens",
            "num_freqs",
            "fft_type",
        ):
            setattr(self, name, getattr(frozen, name))
        self.model.requires_grad_(True)
        if self.model.target_encoder is not None:
            self.model.target_encoder.requires_grad_(False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.model.target_encoder is not None:
            self.model.target_encoder.eval()
        return self

    def encode(self, raw_action_norm, task_emb):
        lang = self._prepare_lang(task_emb).to(raw_action_norm.device, torch.float32)
        with torch.autocast(device_type=raw_action_norm.device.type, enabled=False):
            return self.model.encoder(raw_action_norm.float(), lang)

    @torch.no_grad()
    def encode_target(self, raw_action_norm, task_emb):
        encoder = (
            self.model.target_encoder
            if self.model.target_encoder is not None
            else self.model.encoder
        )
        lang = self._prepare_lang(task_emb).to(raw_action_norm.device, torch.float32)
        with torch.autocast(device_type=raw_action_norm.device.type, enabled=False):
            return encoder(raw_action_norm.float(), lang)

    def progress_loss_from_data(self, traj_a_freq, traj_a_len, total_length, task_emb):
        with torch.autocast(device_type=traj_a_freq.device.type, enabled=False):
            return self.model.progress_loss_from_data(
                {
                    "traj_a_freq": traj_a_freq.float(),
                    "traj_a_len": traj_a_len,
                    "total_length": total_length,
                    self.model.lang_key: self._prepare_task_embedding(task_emb).float(),
                }
            )

    def dynamics_loss(self, latent, next_latent, task_emb, traj_a_freq, traj_a_len):
        with torch.autocast(device_type=latent.device.type, enabled=False):
            phase = self.model.phase_from_freq(
                traj_a_freq.float(),
                traj_a_len=traj_a_len,
                batch_size=len(latent),
                task_emb=self._prepare_task_embedding(task_emb).float(),
            )
            loss, _ = self.model.dynamics_model.compute_loss(
                latent.float(), next_latent.float(), phase
            )
        return loss


def load_trainable_tcla_action_ae(config_yaml, ckpt_path, device="cuda"):
    return TrainableTCLAActionAE(config_yaml, ckpt_path, device)
