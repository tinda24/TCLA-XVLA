import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def collect_episode_paths(data_root, task_names=None, num_episodes=50):
    root = Path(data_root)
    tasks = (
        list(task_names)
        if task_names
        else sorted(path.name for path in root.iterdir() if path.is_dir())
    )
    paths = []
    for task in tasks:
        episodes = sorted(
            (root / task).glob("episode_*.hdf5"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if not episodes:
            raise FileNotFoundError(f"No episodes for task {task} in {root}")
        paths.extend(episodes[:num_episodes])
    if not paths:
        raise ValueError(f"No episodes found in {root}")
    return paths


def compute_qpos_stats(paths):
    values = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            values.append(handle["observations/qpos"][:].astype(np.float32))
    values = np.concatenate(values)
    return values.mean(axis=0), values.std(axis=0).clip(1e-2, 10.0)


def slice_with_padding(array, start, length):
    return array[np.arange(start, start + length).clip(0, len(array) - 1)]


class PolicyDataset(IterableDataset):
    def __init__(self, args, *, chunk_size, action_ae=None, rank=0, world_size=1):
        self.platform, self.mode = args.dataset, args.mode
        if self.mode not in {"raw", "latent"}:
            raise ValueError("mode must be raw or latent")
        self.camera_names = list(args.camera_names)
        self.chunk_size, self.domain_id, self.action_dim = (
            chunk_size,
            args.domain_id,
            args.action_dim,
        )
        self.seed = args.seed + rank * 1000003
        self.rng = random.Random(self.seed)
        self.action_normalizer = None
        all_paths = collect_episode_paths(
            args.data_root, args.task_names, args.num_episodes
        )
        self.task_names = sorted({path.parent.name for path in all_paths})
        self.episode_paths = all_paths[rank::world_size]
        if not self.episode_paths:
            raise ValueError("Each distributed rank needs at least one episode")
        self.task_embeddings = {}
        ambiguous_tasks = set()
        if self.mode != "raw":
            from policy_data.normalization import ActionAENormalizer
            from policy_data.task_embedding import TaskEmbEncoder

            if action_ae.action_dim != self.action_dim:
                raise ValueError(
                    "AE action dimension does not match the selected platform"
                )
            if action_ae.latent_action_dim < self.action_dim:
                raise ValueError(
                    "AE latent width must accommodate all proprioception channels"
                )
            self.num_freqs, self.fft_type, self.lang_dim = (
                action_ae.num_freqs,
                action_ae.fft_type,
                action_ae.model.lang_dim,
            )
            self.action_normalizer = ActionAENormalizer.from_path(args.action_ae_stats)
            self.task_encoder = TaskEmbEncoder(args.action_ae_config)
            self.qpos_mean, self.qpos_std = compute_qpos_stats(all_paths)
            # Each rank exports a complete task cache although episodes are sharded.
            for path in all_paths:
                task = path.parent.name
                if task in ambiguous_tasks or (
                    self.platform == "franka" and task in self.task_embeddings
                ):
                    continue
                if self.platform == "franka":
                    embedding = self.task_encoder.encode(task)
                else:
                    with h5py.File(path, "r") as handle:
                        embedding = np.asarray(
                            handle["task_emb"][()], dtype=np.float32
                        ).reshape(-1)
                if embedding.shape != (self.lang_dim,):
                    raise ValueError(
                        f"Invalid task embedding shape for {task}: {embedding.shape}"
                    )
                if task in self.task_embeddings and not np.array_equal(
                    self.task_embeddings[task], embedding
                ):
                    del self.task_embeddings[task]
                    ambiguous_tasks.add(task)
                    continue
                self.task_embeddings[task] = embedding
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.episodes = [self._load_episode(path) for path in self.episode_paths]

    def _load_episode(self, path):
        with h5py.File(path, "r") as handle:
            action = handle["action"][:].astype(np.float32)
            qpos = handle["observations/qpos"][:].astype(np.float32)
            images = {
                name: handle[f"observations/images/{name}"][:]
                for name in self.camera_names
            }
            instruction = (
                handle["task_name"][()] if "task_name" in handle else path.parent.name
            )
            instruction = (
                instruction.decode("utf-8")
                if isinstance(instruction, bytes)
                else str(instruction)
            )
            task_emb = None
            if self.mode != "raw" and self.platform == "robotwin":
                task_emb = np.asarray(handle["task_emb"][()], dtype=np.float32).reshape(
                    -1
                )
        if (
            action.ndim != 2
            or action.shape[1] != self.action_dim
            or qpos.shape != action.shape
        ):
            raise ValueError(
                f"Expected aligned [T, {self.action_dim}] actions and qpos: {path}"
            )
        needed = self.chunk_size
        if not len(action) or (self.platform == "franka" and len(action) < needed):
            raise ValueError(f"Episode too short for {needed} action steps: {path}")
        if any(len(image) != len(action) for image in images.values()):
            raise ValueError(f"Image and action lengths differ: {path}")
        episode = {
            "task": path.parent.name,
            "instruction": instruction,
            "action": action,
            "qpos": qpos,
            "images": images,
        }
        if self.mode != "raw":
            from policy_data.normalization import encode_trajectory_to_frequency

            episode["action_norm"] = self.action_normalizer.normalize_action(action)
            episode["task_emb"] = (
                task_emb
                if task_emb is not None
                else self.task_embeddings[path.parent.name]
            )
            episode["freq"] = np.stack(
                [
                    self.action_normalizer.normalize_traj_a_freq(
                        encode_trajectory_to_frequency(
                            episode["action_norm"][:end],
                            num_freqs=self.num_freqs,
                            fft_type=self.fft_type,
                        )
                    )
                    for end in range(1, len(action) + 1)
                ]
            ).astype(np.float32)
        return episode

    def _sample(self):
        episode = self.rng.choice(self.episodes)
        length = len(episode["action"])
        last = length - 1
        if self.platform == "franka":
            last = length - self.chunk_size
        start = self.rng.randint(0, last)
        task = episode["task"]
        if self.platform == "franka":
            from tcla.dataloader.franka import task_prompt_for

            instruction = task_prompt_for(task)
        else:
            instruction = episode["instruction"]
        images = [
            self.image_transform(Image.fromarray(episode["images"][name][start]))
            for name in self.camera_names
        ]
        sample = {
            "image_input": torch.stack(images),
            "image_mask": torch.ones(len(images), dtype=torch.bool),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "language_instruction": instruction,
            "raw_qpos": torch.from_numpy(episode["qpos"][start]),
            "raw_action": torch.from_numpy(
                slice_with_padding(episode["action"], start, self.chunk_size)
            ),
        }
        if self.mode != "raw":
            sample.update(
                qpos=torch.from_numpy(
                    (episode["qpos"][start] - self.qpos_mean) / self.qpos_std
                ),
                raw_action_norm=torch.from_numpy(
                    slice_with_padding(episode["action_norm"], start, self.chunk_size)
                ),
                task_emb=torch.from_numpy(episode["task_emb"]),
                traj_a_freq=torch.from_numpy(episode["freq"][max(start - 1, 0)]),
                traj_a_len=torch.tensor(start, dtype=torch.long),
            )
        return sample

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        self.rng = random.Random(self.seed + (worker.id * 1000033 if worker else 0))
        while True:
            yield self._sample()


def create_dataloader(args, accelerator, action_ae=None):
    dataset = PolicyDataset(
        args,
        chunk_size=action_ae.chunk_size if action_ae else args.chunk_size,
        action_ae=action_ae,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=accelerator.device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
