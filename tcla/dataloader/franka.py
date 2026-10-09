from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence

import h5py
import numpy as np

from tcla.dataloader.robotwin_ae_w_freqnorm import collate_fn as ae_collate_fn
from tcla.dataloader.robotwin_ae_w_freqnorm import RoboTwinAEWFreqNormDataset
from tcla.utils.text_embedding import encode_texts_np
from tcla.configuration import load_paths

os_env = __import__("os")
os_env.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


CAMERA_HDF5_KEYS: Dict[str, str] = {
    "cam_front": "front_cam",
    "cam_high": "front_cam",
    "front_cam": "front_cam",
    "cam_right_wrist": "wrist_cam",
    "cam_left_wrist": "wrist_cam",
    "cam_wrist": "wrist_cam",
    "wrist_cam": "wrist_cam",
}

DEFAULT_FRANKA_TASKS = (
    "260830_bowl_separate",
    "260830_bowl_stack",
    "260830_button",
    "260830_light",
    "260830_screw",
)

TASK_PROMPTS: Dict[str, str] = {
    "260830_bowl_separate": "separate the bowls",
    "260830_bowl_stack": "stack the bowls",
    "260830_button": "press the button",
    "260830_light": "turn on the light",
    "260830_screw": "tighten the screw",
}

DEFAULT_TASK_EMB_SOURCE = "minilm"
TASK_EMB_SOURCES = ("minilm", "hash")


def _resolve_camera_key(camera_name: str) -> str:
    if camera_name not in CAMERA_HDF5_KEYS:
        raise KeyError(
            f"Unsupported camera '{camera_name}'. Supported: {sorted(CAMERA_HDF5_KEYS)}"
        )
    return CAMERA_HDF5_KEYS[camera_name]


def _task_name_embedding(task_name: str, dim: int = 384) -> np.ndarray:
    digest = hashlib.sha256(task_name.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(dim).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-6)


def task_prompt_for(task_name: str) -> str:
    if task_name in TASK_PROMPTS:
        return TASK_PROMPTS[task_name]
    parts = str(task_name).split("_", 1)
    name = parts[1] if len(parts) == 2 and parts[0].isdigit() else str(task_name)
    return name.replace("_", " ")


def _resolve_task_name(path: Path) -> str:

    return path.parent.name


def _full_window_start(
    rng, episode_len: int, window_len: int, n_windows: int = 1
) -> int:
    needed = int(n_windows) * int(window_len)
    max_start = int(episode_len) - needed
    if max_start < 0:
        raise ValueError(
            f"Episode length {episode_len} is shorter than {n_windows} window(s) "
            f"of {window_len} (need {needed})."
        )
    return rng.randint(0, max_start)


def _load_franka_episode(
    path: Path,
    camera_names: Sequence[str],
    load_images: bool = True,
    task_emb: Optional[np.ndarray] = None,
) -> dict:
    episode: dict = {}
    with h5py.File(path, "r") as f:
        task_name = _resolve_task_name(path)
        episode["qpos"] = f["observations/qpos"][:].astype(np.float32)
        episode["action"] = f["action"][:].astype(np.float32)

        if load_images:
            image_group = f.get("observations/images")
            available = [] if image_group is None else list(image_group.keys())
            images = {}
            for cam in camera_names:
                h5_key = _resolve_camera_key(cam)
                if h5_key not in available:
                    raise KeyError(
                        f"Missing camera dataset '{h5_key}' in {path} "
                        f"(logical '{cam}'). Available: {available}"
                    )
                images[cam] = f[f"observations/images/{h5_key}"][:]
            episode["images"] = images

        if task_emb is not None:
            episode["task_emb"] = np.asarray(task_emb, dtype=np.float32).reshape(-1)
        elif "task_emb" in f:
            stored = f["task_emb"][:].astype(np.float32)
            if stored.ndim > 1:
                stored = stored.reshape(-1)
            episode["task_emb"] = stored
        else:
            episode["task_emb"] = _task_name_embedding(task_name)

        episode["task_name"] = task_name
        episode["__path__"] = str(path)
    return episode


class _FrankaTaskEmbMixin:
    def __init__(self, *args, **kwargs):
        self.task_emb_source = str(
            kwargs.pop("task_emb_source", DEFAULT_TASK_EMB_SOURCE)
        ).lower()
        if self.task_emb_source not in TASK_EMB_SOURCES:
            raise ValueError(
                f"Unsupported task_emb_source={self.task_emb_source!r}. "
                f"Use one of {TASK_EMB_SOURCES}."
            )
        self.sentence_encoder_name_or_path = str(
            kwargs.pop("sentence_encoder_name_or_path", None)
            or load_paths().paths.language_model
        )
        self.sentence_encoder_device = str(kwargs.pop("sentence_encoder_device", "cpu"))
        self.lang_dim = int(kwargs.pop("lang_dim", 384))
        self._task_emb_cache: Dict[str, np.ndarray] = {}
        print(
            f"[franka] task_emb_source={self.task_emb_source} "
            f"encoder={self.sentence_encoder_name_or_path}"
        )
        super().__init__(*args, **kwargs)

    def _compute_task_emb(self, task_name: str) -> np.ndarray:
        if task_name not in self._task_emb_cache:
            if self.task_emb_source == "hash":
                emb = _task_name_embedding(task_name, dim=self.lang_dim)
            else:
                emb = encode_texts_np(
                    [task_prompt_for(task_name)],
                    name_or_path=self.sentence_encoder_name_or_path,
                    dim=self.lang_dim,
                    device=self.sentence_encoder_device,
                )[0]
            self._task_emb_cache[task_name] = emb
        return self._task_emb_cache[task_name]


class FrankaAEWFreqNormDataset(_FrankaTaskEmbMixin, RoboTwinAEWFreqNormDataset):
    def _load_episode(self, path: Path, *, accumulate_stats: bool = False):
        episode = _load_franka_episode(
            path,
            camera_names=(),
            load_images=False,
            task_emb=self._compute_task_emb(_resolve_task_name(path)),
        )
        return self._attach_traj_a_freq_cache(
            episode, accumulate_stats=accumulate_stats
        )

    def _sample_start(self, episode_len: int) -> int:
        action_length = int(self.n_obs_steps) - 1 + int(self.n_action_steps)
        return _full_window_start(self._rng, episode_len, action_length, n_windows=2)


__all__ = [
    "FrankaAEWFreqNormDataset",
    "ae_collate_fn",
    "DEFAULT_FRANKA_TASKS",
    "TASK_PROMPTS",
    "DEFAULT_TASK_EMB_SOURCE",
    "task_prompt_for",
]
