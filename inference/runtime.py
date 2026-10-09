import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor


def array_value(value, dtype=None):
    return np.asarray(value, dtype=dtype)


class PolicyRuntime:
    def __init__(self, checkpoint, device="auto", steps=10, execute_steps=None):
        checkpoint = Path(checkpoint)
        self.metadata = json.loads(
            (checkpoint / "policy.json").read_text(encoding="utf-8")
        )
        self.mode = self.metadata["mode"]
        self.action_dim = int(self.metadata["action_dim"])
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.steps = int(steps)
        self.execute_steps = int(execute_steps or self.metadata["chunk_size"])
        if self.steps <= 0 or not 0 < self.execute_steps <= self.metadata["chunk_size"]:
            raise ValueError(
                "steps must be positive; execute_steps must fit in the action chunk"
            )
        self.model = (
            XVLA.from_pretrained(checkpoint)
            .to(device=self.device, dtype=torch.float32)
            .eval()
        )
        self.processor = XVLAProcessor.from_pretrained(checkpoint)
        self.processor.num_views = len(self.metadata["camera_names"])
        self.action_ae = None
        if self.mode != "raw":
            from models.latent_action_ae import FrozenTCLAActionAE
            from policy_data.normalization import ActionAENormalizer
            from policy_data.task_embedding import TaskEmbEncoder

            self.action_ae = FrozenTCLAActionAE(
                checkpoint / "action_ae_config.yaml",
                checkpoint / "action_ae.pt",
                self.device,
            )
            self.normalizer = ActionAENormalizer.from_path(
                checkpoint / "dataset_statistics.json"
            )
            qpos = json.loads(
                (checkpoint / "qpos_stats.json").read_text(encoding="utf-8")
            )
            self.qpos_mean = np.asarray(qpos["qpos_mean"], dtype=np.float32)
            self.qpos_std = np.asarray(qpos["qpos_std"], dtype=np.float32)
            self.task_embeddings = json.loads(
                (checkpoint / "task_embeddings.json").read_text(encoding="utf-8")
            )
            self.task_encoder = TaskEmbEncoder(checkpoint / "action_ae_config.yaml")
            if (
                self.model.num_actions != self.action_ae.latent_action_tokens
                or self.model.action_space.dim_action
                != self.action_ae.latent_action_dim
            ):
                raise ValueError(
                    "Policy and bundled AE have incompatible latent dimensions"
                )
        self.reset()

    def reset(self, task_name=None, task_emb=None, language_instruction=None):
        self.history = []
        self.action_chunk = None
        self.chunk_index = 0
        self.task_name = task_name
        self.language_instruction = language_instruction
        self.task_emb = None
        if self.action_ae is not None and (
            task_name is not None or task_emb is not None
        ):
            if task_emb is None:
                task_emb = self.task_embeddings.get(task_name)
                if task_emb is None:
                    task_emb = self.task_encoder.encode(task_name)
            task_emb = array_value(task_emb, np.float32).reshape(-1)
            if (
                task_emb.shape != (self.action_ae.model.lang_dim,)
                or not np.isfinite(task_emb).all()
            ):
                raise ValueError("task_emb does not match the AE language dimension")
            self.task_emb = torch.as_tensor(task_emb, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def _query_chunk(self, payload):
        qpos = array_value(payload["qpos"], np.float32)
        if qpos.shape != (self.action_dim,) or not np.isfinite(qpos).all():
            raise ValueError(
                f"qpos must be a finite vector of length {self.action_dim}"
            )
        images = []
        for i, name in enumerate(self.metadata["camera_names"]):
            value = array_value(payload[f"image{i}"])
            if value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError(f"image{i} ({name}) must have RGB shape [H, W, 3]")
            images.append(Image.fromarray(value.astype(np.uint8)))
        task_name = self.task_name or payload.get("task_name")
        instruction = payload.get("language_instruction") or self.language_instruction
        if not instruction:
            if not task_name:
                raise ValueError("Provide task_name or language_instruction")
            if self.metadata["dataset"] == "franka":
                from tcla.dataloader.franka import task_prompt_for

                instruction = task_prompt_for(task_name)
            else:
                instruction = task_name
        if self.action_ae is not None:
            if self.task_emb is None:
                self.reset(task_name, payload.get("task_emb"), instruction)
            if self.task_emb is None:
                raise ValueError(
                    "Latent inference requires a task embedding or a known task name"
                )
            qpos = (qpos - self.qpos_mean) / self.qpos_std
            if len(qpos) > self.action_ae.latent_action_dim:
                raise ValueError(
                    "AE latent width must accommodate all proprioception channels"
                )
            qpos = np.pad(qpos, (0, self.action_ae.latent_action_dim - len(qpos)))
        encoded = self.processor(images, instruction)
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        predicted = self.model.generate_actions(
            **inputs,
            proprio=torch.as_tensor(qpos, device=self.device).unsqueeze(0),
            domain_id=torch.tensor([self.metadata["domain_id"]], device=self.device),
            steps=self.steps,
        )
        if self.action_ae is not None:
            from policy_data.normalization import encode_trajectory_to_frequency

            freq = None
            if self.history:
                freq = encode_trajectory_to_frequency(
                    np.stack(self.history),
                    num_freqs=self.action_ae.num_freqs,
                    fft_type=self.action_ae.fft_type,
                )
                freq = torch.as_tensor(
                    self.normalizer.normalize_traj_a_freq(freq), device=self.device
                ).unsqueeze(0)
            decoded = self.action_ae.decode(
                predicted,
                self.task_emb,
                traj_a_freq=freq,
                traj_a_len=torch.tensor([len(self.history)], device=self.device),
            )
            actions = self.normalizer.unnormalize_action(decoded[0].cpu().numpy())
        else:
            actions = predicted[0].cpu().numpy()
        if (
            actions.shape != (self.metadata["chunk_size"], self.action_dim)
            or not np.isfinite(actions).all()
        ):
            raise RuntimeError("Policy produced invalid actions")
        return actions[: self.execute_steps].astype(np.float32)

    def act(self, payload):
        if self.action_chunk is None or self.chunk_index >= len(self.action_chunk):
            self.action_chunk = self._query_chunk(payload)
            self.chunk_index = 0
        action = self.action_chunk[self.chunk_index].copy()
        self.chunk_index += 1
        if self.action_ae is not None:
            self.history.append(self.normalizer.normalize_action(action))
        return action
