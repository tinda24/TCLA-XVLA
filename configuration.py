import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parent


def absolute_path(value):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())


def load_paths():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs/paths.yaml")
    override = os.environ.get("XVLA_TCLA_PATHS_CONFIG")
    local = (
        Path(absolute_path(override))
        if override
        else PROJECT_ROOT / "configs/paths.local.yaml"
    )
    if override or local.is_file():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(local))
    for group in ("data", "output", "action_ae"):
        for name, value in cfg.paths[group].items():
            cfg.paths[group][name] = absolute_path(value)
    for name in ("policy_checkpoint", "robotwin", "eval_output"):
        cfg.paths[name] = absolute_path(cfg.paths[name])
    for name in ("pretrained_model", "language_model"):
        value = str(cfg.paths[name])
        if (
            value.startswith((".", "~"))
            or Path(value).is_absolute()
            or (PROJECT_ROOT / value).exists()
        ):
            cfg.paths[name] = absolute_path(value)
    return cfg


def load_training_config(path, dotlist=()):
    return OmegaConf.merge(
        load_paths(),
        OmegaConf.load(absolute_path(path)),
        OmegaConf.from_dotlist(dotlist),
    )


def policy_config(path, mode, overrides=()):
    cfg = OmegaConf.merge(
        load_paths(),
        OmegaConf.load(PROJECT_ROOT / "configs/policy/base.yaml"),
        OmegaConf.load(absolute_path(path)),
        {"mode": mode},
        OmegaConf.from_dotlist(overrides),
    )
    if cfg.dataset not in {"robotwin", "franka"}:
        raise ValueError("dataset must be robotwin or franka")
    if cfg.mode not in {"raw", "latent", "latent_joint"}:
        raise ValueError("mode must be raw, latent or latent_joint")
    OmegaConf.resolve(cfg)
    return cfg


def policy_args(cfg):
    values = dict(OmegaConf.to_container(cfg.training, resolve=True))
    values.update(OmegaConf.to_container(cfg.data, resolve=True))
    values.update(
        dataset=cfg.dataset,
        mode=cfg.mode,
        models=cfg.models,
        output_dir=absolute_path(cfg.output_dir),
    )
    values["data_root"] = absolute_path(values["data_root"])
    values["action_dim"] = 14 if cfg.dataset == "robotwin" else 8
    values["action_mode"] = "joint" if cfg.dataset == "robotwin" else "joint8"
    values["max_action_dim"] = values["action_dim"]
    if cfg.mode == "raw":
        values["chunk_size"] = int(cfg.raw.chunk_size)
    else:
        values.update(OmegaConf.to_container(cfg.latent, resolve=True))
        ae_dir = Path(absolute_path(cfg.action_ae.run_dir))
        values.update(
            action_ae_config=str(ae_dir / "config.yaml"),
            action_ae_ckpt=str(
                ae_dir / "checkpoints" / f"steps_{cfg.action_ae.step}_pytorch_model.pt"
            ),
            action_ae_stats=str(ae_dir / "dataset_statistics.json"),
        )
        if cfg.mode == "latent_joint":
            values.update(OmegaConf.to_container(cfg.joint_ae, resolve=True))
    for name in (
        "batch_size",
        "iters",
        "save_interval",
        "log_interval",
        "num_episodes",
    ):
        if int(values[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(values["num_workers"]) < 0:
        raise ValueError("num_workers must be nonnegative")
    return SimpleNamespace(**values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key")
    args = parser.parse_args()
    cfg = load_paths()
    print(
        OmegaConf.select(cfg, args.key, throw_on_missing=True)
        if args.key
        else OmegaConf.to_yaml(cfg)
    )
