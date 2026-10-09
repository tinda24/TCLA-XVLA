import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from tcla.dataloader.frequency_encoding import trajectory_to_frequency


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class RoboTwinAEWFreqNormDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        num_freqs: int = 256,
        fft_type: str = "dct",
        task_name: Union[str, Sequence[str]] = "adjust_bottle",
        n_obs_steps: int = 8,
        n_action_steps: int = 8,
        num_episodes: Optional[int] = 50,
        val_episode_index: int = 0,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        super().__init__()
        self.path = Path(path)
        self.task_names = [task_name] if isinstance(task_name, str) else list(task_name)
        if not self.task_names:
            raise ValueError("task_name must contain at least one task.")
        self.task_name = task_name
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_episodes = num_episodes
        self.rank = rank
        self.world_size = world_size
        self.seed = seed + rank
        self.num_freqs = num_freqs
        self.fft_type = str(fft_type)
        self.val_episode_index = int(val_episode_index)
        (
            self.train_episode_paths,
            self.val_episode_path_by_task,
            self.train_first_episode_path_by_task,
        ) = self._load_episode_paths()
        self._stats = self._compute_action_stats(self.all_train_episode_paths)
        self._traj_a_freq_sum: Optional[np.ndarray] = None
        self._traj_a_freq_sum_sq: Optional[np.ndarray] = None
        self._traj_a_freq_count: int = 0
        self._traj_a_freq_mean: Optional[np.ndarray] = None
        self._traj_a_freq_std: Optional[np.ndarray] = None
        self._rng = random.Random(self.seed)
        self._episodes = self._load_all_episodes(self.train_episode_paths)
        self._finalize_traj_a_freq_stats()
        self._stats["traj_a_freq_mean"] = self._traj_a_freq_mean
        self._stats["traj_a_freq_std"] = self._traj_a_freq_std
        self.val_episode_by_task = self._load_val_episodes()
        self.len_episodes = len(self._episodes)
        print(
            f"Loaded {self.len_episodes} train RoboTwin episodes from "
            f"{self.task_names} on rank {self.rank}; "
            f"val episode_0 per task: {len(self.val_episode_by_task)}"
        )

    @staticmethod
    def _episode_index(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    def _load_episode_paths(self):
        train_paths = []
        val_path_by_task = {}
        train_first_by_task = {}

        for task_name in self.task_names:
            task_dir = self.path / task_name
            if not task_dir.exists():
                raise FileNotFoundError(
                    f"RoboTwin task directory does not exist: {task_dir}"
                )
            task_paths = sorted(
                task_dir.glob("episode_*.hdf5"),
                key=self._episode_index,
            )
            if self.num_episodes is not None:
                task_paths = task_paths[: self.num_episodes]

            val_path = None
            task_train_paths = []
            for path in task_paths:
                if self._episode_index(path) == self.val_episode_index:
                    val_path = path
                else:
                    task_train_paths.append(path)

            if val_path is None:
                raise FileNotFoundError(
                    f"Task {task_name} has no episode_{self.val_episode_index}.hdf5 for validation."
                )
            if not task_train_paths:
                raise RuntimeError(
                    f"Task {task_name} has no training episodes after holding out "
                    f"episode_{self.val_episode_index}.hdf5."
                )

            val_path_by_task[task_name] = val_path
            train_first_by_task[task_name] = task_train_paths[0]
            train_paths.extend(task_train_paths)

        self.all_train_episode_paths = train_paths
        train_paths = train_paths[self.rank :: self.world_size]
        if not train_paths:
            raise RuntimeError("No RoboTwin train episodes are assigned to this rank.")
        return train_paths, val_path_by_task, train_first_by_task

    def _compute_action_stats(self, episode_paths) -> Dict[str, np.ndarray]:
        all_actions = []

        for path in episode_paths:
            with h5py.File(path, "r") as f:
                all_actions.append(f["action"][:].astype(np.float32))

        all_actions = np.concatenate(all_actions, axis=0)
        action_std = np.std(all_actions, axis=0).astype(np.float32)

        return {
            "action_mean": np.mean(all_actions, axis=0).astype(np.float32),
            "action_std": np.clip(action_std, 1e-5, None),
        }

    def _accumulate_traj_a_freq_stats(self, traj_a_freq_by_prefix: np.ndarray) -> None:

        batch = traj_a_freq_by_prefix.astype(np.float64)
        if self._traj_a_freq_sum is None:
            self._traj_a_freq_sum = np.sum(batch, axis=0)
            self._traj_a_freq_sum_sq = np.sum(batch**2, axis=0)
        else:
            self._traj_a_freq_sum += np.sum(batch, axis=0)
            self._traj_a_freq_sum_sq += np.sum(batch**2, axis=0)
        self._traj_a_freq_count += int(batch.shape[0])

    def _finalize_traj_a_freq_stats(self) -> None:
        # All ranks must use the same action and frequency coordinate system.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.distributed.get_backend() == "nccl"
                else torch.device("cpu")
            )
            shape = self._traj_a_freq_sum.shape
            merged = torch.as_tensor(
                np.concatenate(
                    [
                        self._traj_a_freq_sum.ravel(),
                        self._traj_a_freq_sum_sq.ravel(),
                        [self._traj_a_freq_count],
                    ]
                ),
                dtype=torch.float64,
                device=device,
            )
            torch.distributed.all_reduce(merged)
            merged = merged.cpu().numpy()
            size = int(np.prod(shape))
            self._traj_a_freq_sum = merged[:size].reshape(shape)
            self._traj_a_freq_sum_sq = merged[size : 2 * size].reshape(shape)
            self._traj_a_freq_count = int(merged[-1])
        if (
            self._traj_a_freq_count <= 0
            or self._traj_a_freq_sum is None
            or self._traj_a_freq_sum_sq is None
        ):
            raise RuntimeError(
                "Failed to compute traj_a_freq statistics from training episodes."
            )

        mean = self._traj_a_freq_sum / self._traj_a_freq_count
        variance = self._traj_a_freq_sum_sq / self._traj_a_freq_count - mean**2
        std = np.sqrt(np.maximum(variance, 0.0))
        self._traj_a_freq_mean = mean.astype(np.float32)
        self._traj_a_freq_std = np.clip(std.astype(np.float32), 1e-5, None)

    @property
    def stats(self):
        return self._stats

    def _preprocess_action(self, x: np.ndarray) -> np.ndarray:
        return (x - self.stats["action_mean"]) / self.stats["action_std"]

    def _preprocess_traj_a_freq(self, x: np.ndarray) -> np.ndarray:
        if self._traj_a_freq_mean is None or self._traj_a_freq_std is None:
            raise RuntimeError("traj_a_freq statistics are not initialized.")
        return (x - self._traj_a_freq_mean) / self._traj_a_freq_std

    @staticmethod
    def _read_hdf5_string(value):
        if hasattr(value, "__getitem__"):
            value = value[()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _precompute_traj_a_freq_by_prefix(self, action: np.ndarray) -> np.ndarray:

        action_norm = self._preprocess_action(action)
        episode_len = action_norm.shape[0]
        cached = []
        for prefix_len in range(1, episode_len + 1):
            prefix = torch.from_numpy(action_norm[:prefix_len])
            cached.append(
                trajectory_to_frequency(
                    prefix, num_freqs=self.num_freqs, fft_type=self.fft_type
                ).numpy()
            )
        return np.stack(cached, axis=0)

    def _attach_traj_a_freq_cache(
        self, episode: dict, *, accumulate_stats: bool
    ) -> dict:
        episode["traj_a_freq_by_prefix"] = self._precompute_traj_a_freq_by_prefix(
            episode["action"]
        )
        if accumulate_stats:
            self._accumulate_traj_a_freq_stats(episode["traj_a_freq_by_prefix"])
        return episode

    def _load_episode(self, path: Path, *, accumulate_stats: bool = False):
        episode = {}
        with h5py.File(path, "r") as f:
            task_name = (
                self._read_hdf5_string(f["task_name"])
                if "task_name" in f
                else path.parent.name
            )
            episode["qpos"] = f["observations/qpos"][:].astype(np.float32)
            episode["action"] = f["action"][:].astype(np.float32)
            if "task_emb" in f:
                episode["task_emb"] = f["task_emb"][:].astype(np.float32)
            else:
                raise ValueError(f"Task {task_name} has no task_emb.")
            episode["task_name"] = task_name
            episode["__path__"] = str(path)
        return self._attach_traj_a_freq_cache(
            episode, accumulate_stats=accumulate_stats
        )

    def _load_all_episodes(self, episode_paths):
        episodes = []
        for path in tqdm(
            episode_paths,
            desc=f"Loading RoboTwin train episodes rank{self.rank} (incl. traj_a_freq)",
        ):
            episodes.append(self._load_episode(path, accumulate_stats=True))
        if not episodes:
            raise RuntimeError("No RoboTwin train episodes loaded into memory.")
        return episodes

    def _load_val_episodes(self):
        val_episodes = {}
        for task_name, path in self.val_episode_path_by_task.items():
            episode = self._load_episode(path)
            val_episodes[task_name] = episode
        return val_episodes

    def _sample_start(self, episode_len: int):
        return self._rng.randint(0, episode_len - 1)

    @staticmethod
    def _slice_with_padding(array, start, length):
        if start >= array.shape[0]:
            return np.repeat(array[-1:], length, axis=0)
        if start + length <= 0:
            return np.repeat(array[:1], length, axis=0)
        buffer_start = max(start, 0)
        buffer_end = min(start + length, array.shape[0])
        sample_start = buffer_start - start
        sample_end = sample_start + (buffer_end - buffer_start)

        out = np.empty((length, *array.shape[1:]), dtype=array.dtype)
        out[sample_start:sample_end] = array[buffer_start:buffer_end]
        if sample_start > 0:
            out[:sample_start] = array[0]
        if sample_end < length:
            out[sample_end:] = array[-1]
        return out

    def _sample(self):
        episode = self._rng.choice(self._episodes)
        episode_len = episode["action"].shape[0]
        start = self._sample_start(episode_len)
        action_length = self.n_obs_steps - 1 + self.n_action_steps

        action_sequence = episode["action"]

        action = self._preprocess_action(
            self._slice_with_padding(action_sequence, start, action_length)
        )
        next_action = self._preprocess_action(
            self._slice_with_padding(
                action_sequence, start + action_length, action_length
            )
        )

        prefix_len_a = start
        freq_index = max(prefix_len_a - 1, 0)

        sample = {
            "action": torch.from_numpy(action),
            "next_action": torch.from_numpy(next_action),
            "task_emb": torch.from_numpy(episode["task_emb"]),
            "task_name": episode["task_name"],
            "traj_a_freq": torch.from_numpy(
                self._preprocess_traj_a_freq(
                    episode["traj_a_freq_by_prefix"][freq_index]
                )
            ),
            "traj_a_len": torch.tensor(prefix_len_a, dtype=torch.long),
            "total_length": torch.tensor(episode_len, dtype=torch.long),
        }
        return sample

    def __iter__(self) -> Iterable[dict]:
        worker = torch.utils.data.get_worker_info()
        self._rng = random.Random(self.seed + (worker.id * 1000003 if worker else 0))
        while True:
            yield self._sample()

    def __len__(self):
        total = 0
        for path in self.train_episode_paths:
            with h5py.File(path, "r") as f:
                total += int(f["action"].shape[0])
        return total


def collate_fn(batch):
    return batch


def make_robotwin_action_encoder_loader(
    dataset: RoboTwinAEWFreqNormDataset,
    batch_size: int,
    num_workers: int = 0,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
