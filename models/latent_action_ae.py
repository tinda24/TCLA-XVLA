import torch
from omegaconf import OmegaConf

from tcla.model.framework.action_autoencoder_version3 import ActionAutoencoder


class FrozenTCLAActionAE:
    def __init__(self, config_yaml, ckpt_path, device="cuda"):
        cfg = OmegaConf.load(config_yaml)
        model_cfg = OmegaConf.to_container(cfg.framework, resolve=True)
        if model_cfg.pop("name") != "ActionAutoencoder_version3":
            raise ValueError("XVLA-TCLA requires ActionAutoencoder_version3")
        if model_cfg.get("is_relative_action", False):
            raise ValueError(
                "Policy integration requires an AE trained with absolute actions"
            )
        model_cfg["device"] = str(device)
        self.model = ActionAutoencoder(**model_cfg)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        # AE numerics remain fp32 while the VLA may use mixed precision.
        self.model.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
        self.action_dim = self.model.action_dim
        self.chunk_size = self.model.chunk_size
        self.latent_action_dim = self.model.latent_action_dim
        self.downsample_rate = self.model.downsample_rate
        self.latent_action_tokens = self.model.action_latent_tokens
        self.num_freqs = self.model.num_freqs
        self.fft_type = self.model.fft_type

    @staticmethod
    def _prepare_lang(task_emb):
        return task_emb.unsqueeze(1) if task_emb.ndim == 2 else task_emb

    @staticmethod
    def _prepare_task_embedding(task_emb):
        return task_emb[:, 0] if task_emb.ndim == 3 else task_emb

    @torch.no_grad()
    def encode(self, raw_action_norm, task_emb):
        lang = self._prepare_lang(task_emb).to(raw_action_norm.device, torch.float32)
        with torch.autocast(device_type=raw_action_norm.device.type, enabled=False):
            return self.model.encoder(raw_action_norm.float(), lang)

    def decode(self, latent, task_emb, traj_a_freq=None, traj_a_len=None):
        lang = self._prepare_lang(task_emb).to(latent.device, torch.float32)
        task_embedding = self._prepare_task_embedding(task_emb).to(
            latent.device, torch.float32
        )
        freq = traj_a_freq.float() if traj_a_freq is not None else None
        with torch.autocast(device_type=latent.device.type, enabled=False):
            phase = self.model.phase_from_freq(
                freq,
                traj_a_len=traj_a_len,
                batch_size=len(latent),
                task_emb=task_embedding,
            )
            decoded = self.model.decode_latent(
                latent.float(),
                lang,
                phase=phase,
                raw_freq_action=freq,
                traj_a_len=traj_a_len,
            )
        return decoded[-1] if isinstance(decoded, tuple) else decoded


def load_frozen_tcla_action_ae(config_yaml, ckpt_path, device="cuda"):
    return FrozenTCLAActionAE(config_yaml, ckpt_path, device)
