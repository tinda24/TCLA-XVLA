import argparse
import importlib
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tcla.configuration import load_training_config

import tcla.model.framework.action_autoencoder_version3  # noqa: F401
from tcla.model.tools import FRAMEWORK_REGISTRY
from tcla.training.trainer_action_space_version3 import ActionSpaceTrainer
from tcla.training.trainer_utils import make_run_dir, set_seed


DATASET_REGISTRY = {
    "robotwin_ae_w_freqnorm": (
        "tcla.dataloader.robotwin_ae_w_freqnorm",
        "RoboTwinAEWFreqNormDataset",
        "collate_fn",
    ),
    "franka_ae_w_freqnorm": (
        "tcla.dataloader.franka",
        "FrankaAEWFreqNormDataset",
        "ae_collate_fn",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    args, overrides = parser.parse_known_args()
    return args, overrides


def normalize_overrides(overrides):
    dotlist = []
    index = 0
    while index < len(overrides):
        item = overrides[index].lstrip("-")
        if "=" in item:
            dotlist.append(item)
            index += 1
        else:
            if index + 1 >= len(overrides):
                raise ValueError(f"Missing value for override: {overrides[index]}")
            dotlist.append(f"{item}={overrides[index + 1]}")
            index += 2
    return dotlist


def load_config(config_yaml, overrides):
    dotlist = normalize_overrides(overrides)
    cfg = load_training_config(config_yaml, dotlist)
    OmegaConf.resolve(cfg)
    return cfg


def build_dataset(cfg, accelerator):
    dataset_cfg = dict(cfg.datasets.action_data.dataset)
    dataset_cfg["rank"] = accelerator.process_index
    dataset_cfg["world_size"] = accelerator.num_processes

    dataset_name = cfg.datasets.action_data.dataset_py
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unsupported dataset_py: {dataset_name}. Available: {available}"
        )
    module_name, dataset_cls_name, collate_fn_name = DATASET_REGISTRY[dataset_name]
    dataset_module = importlib.import_module(module_name)
    dataset_cls = getattr(dataset_module, dataset_cls_name)
    collate_fn = getattr(dataset_module, collate_fn_name)
    return dataset_cls(**dataset_cfg), collate_fn


def build_model(cfg, device):
    model_cfg = dict(cfg.framework)
    framework_name = model_cfg.pop("name")

    model_cfg["device"] = str(device)
    return FRAMEWORK_REGISTRY[framework_name](**model_cfg)


def build_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.trainer.learning_rate.base),
        betas=tuple(cfg.trainer.optimizer.betas),
        eps=float(cfg.trainer.optimizer.eps),
        weight_decay=float(cfg.trainer.optimizer.weight_decay),
    )


def main():
    args, overrides = parse_args()
    cfg = load_config(args.config_yaml, overrides)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=cfg.accelerator.get("mixed_precision", "no"),
        gradient_accumulation_steps=int(
            cfg.accelerator.get("gradient_accumulation_steps", 1)
        ),
        kwargs_handlers=[ddp_kwargs],
    )
    accelerator.print(accelerator.state)

    set_seed(int(cfg.seed) + accelerator.process_index)
    resume_checkpoint = (
        cfg.trainer.get("resume_checkpoint", None)
        if cfg.trainer.get("is_resume", False)
        else None
    )
    resume_dir = str(Path(resume_checkpoint).parents[1]) if resume_checkpoint else None
    run_dir = make_run_dir(
        cfg.run_root_dir, cfg.run_name, accelerator, resume_dir=resume_dir
    )
    if accelerator.is_main_process:
        accelerator.print(f"workspace: {run_dir}")

    dataset, collate_fn = build_dataset(cfg, accelerator)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.datasets.action_data.per_device_batch_size),
        num_workers=int(cfg.datasets.action_data.num_workers),
        pin_memory=accelerator.device.type == "cuda",
        collate_fn=collate_fn,
    )

    model = build_model(cfg, accelerator.device)
    optimizer = build_optimizer(model, cfg)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    trainer = ActionSpaceTrainer(
        cfg=cfg,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        run_dir=run_dir,
        dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
