import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def checkpoint_dirname(step, batch_size, world_size, num_freqs):
    bins = "none" if num_freqs is None else str(int(num_freqs))
    return f"ckpt-{int(step)}-bs{int(batch_size)}x{int(world_size)}-bins-{bins}"


def save_policy_checkpoint(
    model, processor, args, dataset, step, action_ae=None, world_size=1
):
    num_freqs = None if action_ae is None else int(action_ae.num_freqs)
    destination = Path(args.output_dir) / checkpoint_dirname(
        step, args.batch_size, world_size, num_freqs
    )
    destination.mkdir(parents=True, exist_ok=True)
    model.config._name_or_path = ""
    model.save_pretrained(destination, safe_serialization=True)
    processor.save_pretrained(destination)
    for filename in (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        path = destination / filename
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key in (
                "_name_or_path",
                "name_or_path",
                "tokenizer_file",
                "special_tokens_map_file",
            ):
                payload.pop(key, None)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "global_step": step,
        "dataset": args.dataset,
        "mode": args.mode,
        "domain_id": args.domain_id,
        "camera_names": dataset.camera_names,
        "action_dim": args.action_dim,
        "chunk_size": dataset.chunk_size,
        "task_names": dataset.task_names,
        "batch_size": int(args.batch_size),
        "world_size": int(world_size),
        "global_batch_size": int(args.batch_size) * int(world_size),
        "num_freqs": num_freqs,
    }
    if action_ae is not None:
        cfg = OmegaConf.load(args.action_ae_config)
        framework = OmegaConf.to_container(cfg.framework, resolve=True)
        framework.pop("device", None)
        source = dataset.task_encoder.source
        portable_config = OmegaConf.create(
            {
                "framework": framework,
                "datasets": {
                    "action_data": {
                        "dataset": {
                            "task_emb_source": source,
                            "lang_dim": action_ae.model.lang_dim,
                        }
                    }
                },
            }
        )
        OmegaConf.save(portable_config, destination / "action_ae_config.yaml")
        torch.save(
            {"model": action_ae.model.state_dict()}, destination / "action_ae.pt"
        )
        statistics = {
            key: value.tolist()
            for key, value in dataset.action_normalizer.stats.items()
        }
        (destination / "dataset_statistics.json").write_text(
            json.dumps(statistics), encoding="utf-8"
        )
        qpos = {
            "qpos_mean": dataset.qpos_mean.tolist(),
            "qpos_std": dataset.qpos_std.tolist(),
        }
        (destination / "qpos_stats.json").write_text(json.dumps(qpos), encoding="utf-8")
        embeddings = {
            name: vector.tolist() for name, vector in dataset.task_embeddings.items()
        }
        (destination / "task_embeddings.json").write_text(
            json.dumps(embeddings), encoding="utf-8"
        )
    (destination / "policy.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
