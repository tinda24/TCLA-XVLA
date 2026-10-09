import argparse
import importlib

from omegaconf import OmegaConf

from configuration import policy_args, policy_config


def main():
    parser = argparse.ArgumentParser(
        description="Train XVLA-TCLA on RoboTwin or Franka."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", choices=["raw", "latent", "latent_joint"], required=True
    )
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument(
        "overrides", nargs="*", help="YAML overrides, e.g. training.iters=50000"
    )
    args = parser.parse_args()
    cfg = policy_config(args.config, args.mode, args.overrides)
    resolved = policy_args(cfg)
    if args.print_config:
        print(OmegaConf.to_yaml(OmegaConf.create(vars(resolved))))
        return
    modules = {
        "raw": "train_raw",
        "latent": "train_latent",
        "latent_joint": "train_latent_joint_ae",
    }
    importlib.import_module(modules[cfg.mode]).main(resolved)


if __name__ == "__main__":
    main()
