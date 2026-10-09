from __future__ import annotations

import os
import time
from pathlib import Path

import psutil
import torch
from accelerate import Accelerator

from policy_data.episodes import create_dataloader
from training.checkpoints import save_policy_checkpoint
from models.configuration_xvla import XVLAConfig
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor
from training.utils import build_optimizer, get_logger, set_seed, update_group_lrs


def move_tensor_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def reinit_joint_action_head(model: XVLA) -> None:
    tr = model.transformer
    torch.nn.init.xavier_uniform_(tr.action_encoder.fc.weight)
    torch.nn.init.zeros_(tr.action_encoder.bias.weight)
    torch.nn.init.xavier_uniform_(tr.action_decoder.fc.weight)
    torch.nn.init.zeros_(tr.action_decoder.bias.weight)


def main(args):
    output_dir = Path(args.output_dir)
    accelerator = Accelerator(log_with="tensorboard", project_dir=output_dir)
    accelerator.init_trackers("XVLA-Raw-Training")
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)

    set_seed(args.seed + accelerator.process_index)
    device = accelerator.device

    config = XVLAConfig.from_pretrained(args.models)
    config.action_mode = str(args.action_mode).lower()
    config.num_actions = int(args.chunk_size)
    config.real_action_dim = int(args.action_dim)
    config.max_action_dim = int(args.action_dim)
    model = XVLA.from_pretrained(
        args.models, config=config, ignore_mismatched_sizes=True
    )
    reinit_joint_action_head(model)
    processor = XVLAProcessor.from_pretrained(args.models)

    train_dataloader = create_dataloader(args, accelerator)

    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_soft=args.learning_coef,
    )
    model, optim = accelerator.prepare(model, optim)
    model.train()

    global_step, t0 = 0, time.time()
    logger.info(
        "Start %s raw training for %s iterations | world_size=%s | action_mode=%s dim=%s",
        args.dataset,
        args.iters,
        accelerator.num_processes,
        config.action_mode,
        args.action_dim,
    )

    for batch in train_dataloader:
        batch = move_tensor_batch(batch, device)
        lang = processor.encode_language(batch["language_instruction"])
        proprio = batch["raw_qpos"]
        inputs = {
            "input_ids": lang["input_ids"].to(device, non_blocking=True),
            "image_input": batch["image_input"],
            "image_mask": batch["image_mask"],
            "domain_id": batch["domain_id"],
            "proprio": proprio,
            "action": batch["raw_action"],
        }

        update_group_lrs(optim, global_step, args)
        loss_dict = model(**inputs)
        loss = sum(loss_dict.values())
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)
            if accelerator.is_main_process:
                dt = (time.time() - t0) / max(1, args.log_interval)
                t0 = time.time()
                cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                logger.info(
                    "[%s/%s] loss=%.4f lr_core=%.2e (%.2fs/it) CPU=%.2fGB GPU=%.2fGB",
                    global_step,
                    args.iters,
                    logs["loss_total"],
                    logs["lr_transformer_core"],
                    dt,
                    cpu_mem,
                    gpu_mem,
                )

        global_step += 1
        if accelerator.is_main_process and (
            global_step == args.iters or global_step % args.save_interval == 0
        ):
            save_policy_checkpoint(
                accelerator.unwrap_model(model),
                processor,
                args,
                train_dataloader.dataset,
                global_step,
            )
        if global_step >= args.iters:
            break

    accelerator.end_training()
