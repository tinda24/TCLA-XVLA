import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf


METRICS_CSV_PREFERRED_ORDER = [
    "step",
    "action_loss",
    "flow_loss",
    "l1_loss",
    "latent_l1_loss",
    "decoded_action_l1_loss",
    "kl_loss",
    "total_loss",
    "pred_loss",
    "recon_loss",
    "learning_rate",
    "epoch",
]


class ActionSpaceTrainerUtils:
    def save_initial_configs(self):
        if not self.accelerator.is_main_process:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, self.run_dir / "config.full.yaml")
        OmegaConf.save(self.cfg, self.run_dir / "config.yaml")

    def save_dataset_statistics(self):
        if not self.accelerator.is_main_process:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stats = getattr(self.dataset, "stats", {})
        with open(self.run_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats, f, indent=2, default=json_default)

    def print_module_parameters(self, model):
        if not self.accelerator.is_main_process:
            return
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.accelerator.print("___________________________________________")
        self.accelerator.print(f"{type(model).__name__} 模型参数量")
        self.accelerator.print(
            f"总参数量: {format_num_params(total_params)} | "
            f"可训练参数量: {format_num_params(trainable_params)}"
        )
        self.accelerator.print("-------------------------------------------")
        for name, module in model.named_children():
            module_total = sum(p.numel() for p in module.parameters())
            module_trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            self.accelerator.print(
                f"{name}: 总参数量 {format_num_params(module_total)} | "
                f"可训练 {format_num_params(module_trainable)}"
            )
        self.accelerator.print("___________________________________________")
        self.save_model_parameters_json(model)

    def save_model_parameters_json(self, model):
        if not self.accelerator.is_main_process:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        modules = {}
        for name, module in model.named_children():
            module_total = sum(p.numel() for p in module.parameters())
            module_trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            modules[name] = {
                "total_params": int(module_total),
                "trainable_params": int(module_trainable),
                "total_params_M": round(module_total / 1e6, 4),
                "trainable_params_M": round(module_trainable / 1e6, 4),
            }
        payload = {
            "model_class": type(model).__name__,
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "total_params_M": round(total_params / 1e6, 4),
            "trainable_params_M": round(trainable_params / 1e6, 4),
            "modules": modules,
        }
        out_path = self.run_dir / "model_parameters.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.accelerator.print(f"saved model parameters: {out_path}")

    @staticmethod
    def _metrics_csv_fieldnames(records):
        keys = set()
        for record in records:
            keys.update(record.keys())
        ordered = [key for key in METRICS_CSV_PREFERRED_ORDER if key in keys]
        ordered.extend(sorted(keys - set(ordered)))
        return ordered

    @property
    def metrics_csv_path(self) -> Path:
        custom = (
            self.cfg.trainer.get("metrics_csv_path", None)
            if hasattr(self, "cfg")
            else None
        )
        if custom:
            path = Path(str(custom))
            if not path.is_absolute():
                path = self.run_dir / path
            return path
        return self.run_dir / "metrics.csv"

    @property
    def metrics_jsonl_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    def _read_metrics_jsonl(self):
        metrics_path = self.metrics_jsonl_path
        if not metrics_path.exists():
            return []
        records = []
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _write_metrics_csv(self, records):
        if not records:
            return self.metrics_csv_path
        csv_path = self.metrics_csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = self._metrics_csv_fieldnames(records)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        return csv_path

    def _append_metrics_csv_row(self, record):
        csv_path = self.metrics_csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        existing_rows = []
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    existing_rows = list(reader)
        all_rows = existing_rows + [record]
        return self._write_metrics_csv(all_rows)

    def record_metrics(self, step, metrics):

        if not self.accelerator.is_main_process:
            return
        record = {"step": int(step), **metrics}
        metrics_path = self.metrics_jsonl_path
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=json_default) + "\n")
        csv_path = self._append_metrics_csv_row(record)
        return csv_path

    def export_metrics_csv(self):
        if not self.accelerator.is_main_process:
            return None
        records = self._read_metrics_jsonl()
        if not records:
            return None
        return self._write_metrics_csv(records)

    @staticmethod
    def _metrics_to_float(outputs):
        metrics = {}
        for key in ["total_loss", "pred_loss", "recon_loss", "progress_loss"]:
            if key not in outputs:
                continue
            value = outputs[key]
            metrics[key] = (
                float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            )
        return metrics

    def log_metrics(self, step, metrics, *, record=True):
        if record:
            self.record_metrics(step, metrics)
        if not self.accelerator.is_main_process:
            return
        items = " | ".join(f"{key}: {value:.6f}" for key, value in metrics.items())
        self.accelerator.print(f"step: {step} | {items} | csv: {self.metrics_csv_path}")
        self.plot_metrics()

    def plot_metrics(self):
        records = self._read_metrics_jsonl()
        if not records:
            return
        if len(records) < 2:
            return

        imgs_dir = self.run_dir / "imgs"
        imgs_dir.mkdir(parents=True, exist_ok=True)
        steps = [record["step"] for record in records]

        self._plot_metric_group(
            imgs_dir / "loss_curve.png",
            steps,
            records,
            ["total_loss", "pred_loss", "recon_loss"],
            "Training Loss",
            "Loss",
        )

    @staticmethod
    def _plot_metric_group(path, steps, records, keys, title, ylabel, log_scale=False):
        plt.figure(figsize=(10, 6))
        for key in keys:
            values = [record[key] for record in records if key in record]
            if len(values) != len(steps):
                continue
            plt.plot(steps, values, label=key)
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        if log_scale:
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def write_summary(self, step):
        if not self.accelerator.is_main_process:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "summary.jsonl", "a") as f:
            f.write(json.dumps({"steps": int(step)}) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(run_root_dir, run_name, accelerator, resume_dir=None):
    if resume_dir:
        run_dir = Path(resume_dir)
        if accelerator.is_main_process:
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()
        return run_dir

    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    if accelerator.num_processes > 1:
        from accelerate.utils import broadcast_object_list

        shared = [run_id]
        broadcast_object_list(shared, from_process=0)
        run_id = shared[0]
    run_dir = Path(run_root_dir) / run_id
    if accelerator.is_main_process:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    return run_dir


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def format_num_params(num_params):
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    if num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


TrainerUtils = ActionSpaceTrainerUtils
