from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from tcla.dataloader.frequency_encoding import trajectory_to_frequency


def normalize_actions_version3(actions: np.ndarray, stats: dict) -> np.ndarray:
    if "action_mean" in stats and "action_std" in stats:
        mean = np.asarray(stats["action_mean"], dtype=np.float32)
        std = np.asarray(stats["action_std"], dtype=np.float32)
        return (actions - mean) / std
    action_min = np.asarray(stats["action_min"], dtype=np.float32)
    action_max = np.asarray(stats["action_max"], dtype=np.float32)
    return 2.0 * (actions - action_min) / (action_max - action_min + 1e-5) - 1.0


def normalize_traj_a_freq_version3(freq: np.ndarray, stats: dict) -> np.ndarray:
    if "traj_a_freq_mean" not in stats or "traj_a_freq_std" not in stats:
        return freq.astype(np.float32)
    mean = np.asarray(stats["traj_a_freq_mean"], dtype=np.float32)
    std = np.asarray(stats["traj_a_freq_std"], dtype=np.float32)
    return ((freq - mean) / std).astype(np.float32)


class ActionAETrainingEvaluatorVersion3:
    def __init__(
        self,
        cfg,
        dataset,
        run_dir: Path,
        device: torch.device,
        batch_size: int = 256,
        eval_loss_max_windows: Optional[int] = 128,
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.run_dir = Path(run_dir)
        self.device = device
        self.batch_size = int(batch_size)
        self.eval_loss_max_windows = eval_loss_max_windows
        self.stats = dict(getattr(dataset, "stats", {}))
        self.chunk_size = int(cfg.framework.chunk_size)
        self.num_freqs = int(cfg.framework.num_freqs)
        self.lang_dim = int(cfg.framework.lang_dim)
        self.fft_type = str(getattr(cfg.framework, "fft_type", "dct"))

    def _encode_prefix_freq(self, normalized_prefix: np.ndarray) -> np.ndarray:
        freq = trajectory_to_frequency(
            torch.from_numpy(normalized_prefix.astype(np.float32)),
            num_freqs=self.num_freqs,
            fft_type=self.fft_type,
        ).numpy()
        return normalize_traj_a_freq_version3(freq, self.stats)

    def _traj_a_freq_for_window(self, episode: dict, prefix_len: int) -> torch.Tensor:
        freq_index = max(prefix_len - 1, 0)
        if "traj_a_freq_by_prefix" in episode:
            raw_freq = episode["traj_a_freq_by_prefix"][freq_index]
            if hasattr(self.dataset, "_preprocess_traj_a_freq"):
                freq = self.dataset._preprocess_traj_a_freq(raw_freq)
            else:
                freq = normalize_traj_a_freq_version3(raw_freq, self.stats)
            return torch.from_numpy(freq.astype(np.float32))

        action_seq = episode["action"]
        prefix_a = self.dataset._preprocess_action(action_seq[: max(prefix_len, 1)])
        raw_freq = self._encode_prefix_freq(prefix_a)
        return torch.from_numpy(raw_freq)

    def _episode_windows(self, episode: dict) -> List[dict]:
        action_seq = episode["action"]
        episode_len = int(action_seq.shape[0])
        action_length = self.chunk_size
        starts = np.arange(episode_len, dtype=np.int64)
        if (
            self.eval_loss_max_windows is not None
            and episode_len > self.eval_loss_max_windows
        ):
            starts = np.linspace(
                0, episode_len - 1, self.eval_loss_max_windows, dtype=np.int64
            )

        windows = []
        for start in starts:
            start = int(start)
            action = self.dataset._preprocess_action(
                self.dataset._slice_with_padding(action_seq, start, action_length)
            )
            next_action = self.dataset._preprocess_action(
                self.dataset._slice_with_padding(
                    action_seq, start + action_length, action_length
                )
            )
            prefix_len = start
            window = {
                "action": torch.from_numpy(action),
                "next_action": torch.from_numpy(next_action),
                "task_emb": torch.from_numpy(episode["task_emb"]),
                "traj_a_freq": self._traj_a_freq_for_window(episode, prefix_len),
                "traj_a_len": torch.tensor(prefix_len, dtype=torch.long),
                "total_length": torch.tensor(episode_len, dtype=torch.long),
            }
            windows.append(window)
        return windows

    @torch.no_grad()
    def _eval_loss_on_episode(self, model, episode: dict) -> Dict[str, float]:
        model.eval()
        windows = self._episode_windows(episode)
        if not windows:
            return {}

        metrics_sum: Dict[str, float] = {}
        count = 0
        num_windows = len(windows)
        for start in range(0, len(windows), self.batch_size):
            batch_windows = windows[start : start + self.batch_size]
            batch = [
                {
                    "action": item["action"],
                    "next_action": item["next_action"],
                    "task_emb": item["task_emb"],
                    "traj_a_freq": item["traj_a_freq"],
                    "traj_a_len": item["traj_a_len"],
                    "total_length": item["total_length"],
                }
                for item in batch_windows
            ]
            examples = batch
            data = model.prepare_data(examples)
            actions = data[model.action_key]
            next_actions = data[f"next_{model.action_key}"]
            task_emb = data[model.lang_key]
            needs_phase = (
                model._decoder_uses_progress() or model._dynamics_uses_vector_phase()
            )
            phase = model.phase_from_batch(data) if needs_phase else None
            decoder_phase = phase if model._decoder_uses_progress() else None
            freq_action_context = data["traj_a_freq"]
            traj_a_len = data["traj_a_len"]

            latent = model.encoder(actions, task_emb)
            target_latent = model.encode_ema_target(next_actions, task_emb)
            pred_loss, _ = model.dynamics_model.compute_loss(
                latent, target_latent, phase
            )
            progress_loss = model.progress_loss_from_data(data)
            recon_actions = model.decode_latent(
                latent,
                task_emb,
                phase=decoder_phase,
                raw_freq_action=freq_action_context,
                traj_a_len=traj_a_len,
            )
            recon_loss = F.mse_loss(recon_actions, actions)

            step_metrics = {
                "recon_loss": float(recon_loss.detach().cpu()),
                "pred_loss": float(pred_loss.detach().cpu()),
                "progress_loss": float(progress_loss.detach().cpu()),
                "total_loss": float(
                    (
                        model.recon_loss_coef * recon_loss
                        + model.dynamics_loss_coef * pred_loss
                        + getattr(model, "progress_loss_coef", 1.0) * progress_loss
                    )
                    .detach()
                    .cpu()
                ),
            }
            for key, value in step_metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
            count += 1

        metrics = {key: value / max(count, 1) for key, value in metrics_sum.items()}
        metrics["num_windows"] = float(num_windows)
        return metrics

    def _append_eval_loss_csv(self, step: int, rows: List[Dict[str, float]]):
        csv_path = self.run_dir / "progress_eval" / "eval_loss.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "step",
            "task",
            "recon_loss",
            "pred_loss",
            "progress_loss",
            "total_loss",
            "num_windows",
        ]

        file_exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "step": int(step),
                        "task": row["task"],
                        "recon_loss": row.get("recon_loss", ""),
                        "pred_loss": row.get("pred_loss", ""),
                        "progress_loss": row.get("progress_loss", ""),
                        "total_loss": row.get("total_loss", ""),
                        "num_windows": row.get("num_windows", ""),
                    }
                )

    def run_train_mode_eval(self, model, step: int):
        if not hasattr(self.dataset, "val_episode_by_task"):
            return

        model.eval()
        loss_rows = []
        for task_name, episode in self.dataset.val_episode_by_task.items():
            metrics = self._eval_loss_on_episode(model, episode)
            if metrics:
                loss_rows.append({"task": task_name, **metrics})

        if loss_rows:
            self._append_eval_loss_csv(step, loss_rows)
