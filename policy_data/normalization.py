import json

import numpy as np
import torch

from tcla.dataloader.frequency_encoding import trajectory_to_frequency


class ActionAENormalizer:
    def __init__(self, stats):
        self.stats = {
            key: np.asarray(value, dtype=np.float32) for key, value in stats.items()
        }
        for key in ("action_mean", "action_std", "traj_a_freq_mean", "traj_a_freq_std"):
            if key not in self.stats:
                raise ValueError(f"AE v3 statistics are missing {key}")
        if np.any(self.stats["action_std"] <= 0) or np.any(
            self.stats["traj_a_freq_std"] <= 0
        ):
            raise ValueError("Normalization standard deviations must be positive")

    @classmethod
    def from_path(cls, path):
        with open(path, encoding="utf-8") as file:
            return cls(json.load(file))

    def normalize_action(self, action):
        return (
            np.asarray(action, dtype=np.float32) - self.stats["action_mean"]
        ) / self.stats["action_std"]

    def unnormalize_action(self, action):
        return (
            np.asarray(action, dtype=np.float32) * self.stats["action_std"]
            + self.stats["action_mean"]
        )

    def unnormalize_action_torch(self, action):
        mean = torch.as_tensor(
            self.stats["action_mean"], device=action.device, dtype=action.dtype
        )
        std = torch.as_tensor(
            self.stats["action_std"], device=action.device, dtype=action.dtype
        )
        return action * std + mean

    def normalize_traj_a_freq(self, freq):
        return (freq - self.stats["traj_a_freq_mean"]) / self.stats["traj_a_freq_std"]


def encode_trajectory_to_frequency(trajectory, *, num_freqs, fft_type="dct"):
    tensor = torch.as_tensor(trajectory, dtype=torch.float32)
    return (
        trajectory_to_frequency(tensor, num_freqs=num_freqs, fft_type=fft_type)
        .cpu()
        .numpy()
    )
