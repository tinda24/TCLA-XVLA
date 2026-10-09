from __future__ import annotations

import os
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW

from policy_data.episodes import create_dataloader
from training.checkpoints import save_policy_checkpoint
from models.joint_latent_action_ae import (
    load_trainable_tcla_action_ae,
)
from models.configuration_xvla import XVLAConfig
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor
from training.utils import get_logger, set_seed
from train_latent import _move_tensor_batch, _pad_or_trim, reinit_latent_action_head


class JointLatentTrainModule(nn.Module):
    def __init__(
        self,
        xvla: XVLA,
        action_ae: nn.Module,
        latent_loss_coef: float,
        decoded_loss_coef: float,
        action_ae_reconstruction_coef: float,
        action_ae_progress_coef: float,
        action_ae_dynamics_coef: float,
        action_normalizer=None,
    ):
        super().__init__()
        self.xvla = xvla
        self.action_ae = action_ae
        self._action_normalizer = action_normalizer
        self.latent_loss_coef = float(latent_loss_coef)
        self.decoded_loss_coef = float(decoded_loss_coef)
        self.action_ae_reconstruction_coef = float(action_ae_reconstruction_coef)
        self.action_ae_progress_coef = float(action_ae_progress_coef)
        self.action_ae_dynamics_coef = float(action_ae_dynamics_coef)

    def forward(
        self,
        input_ids: torch.Tensor,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
        domain_id: torch.Tensor,
        proprio: torch.Tensor,
        raw_action_norm: torch.Tensor,
        task_emb: torch.Tensor,
        traj_a_freq: torch.Tensor,
        traj_a_len: torch.Tensor,
        total_length: torch.Tensor,
        next_raw_action_norm: torch.Tensor,
        raw_action: torch.Tensor | None = None,
    ):
        latent_target = self.action_ae.encode(raw_action_norm, task_emb)

        enc = self.xvla.forward_vlm(input_ids, image_input, image_mask)
        batch = input_ids.shape[0]
        t = (
            torch.rand(1, device=input_ids.device)
            + torch.arange(batch, device=input_ids.device) / batch
        ) % (1 - 1e-5)
        action_noisy = torch.randn_like(latent_target) * t.view(
            -1, 1, 1
        ) + latent_target.detach() * (1 - t.view(-1, 1, 1))
        proprio_m, action_noisy_m = self.xvla.action_space.preprocess(
            proprio, action_noisy
        )
        pred_latent = self.xvla.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )

        latent_loss = (
            F.mse_loss(pred_latent.float(), latent_target.detach())
            * self.latent_loss_coef
        )
        train_losses = {"latent_action_loss": latent_loss}
        metrics = {}

        decoded = self.action_ae.decode(
            pred_latent,
            task_emb,
            traj_a_freq=traj_a_freq,
            traj_a_len=traj_a_len,
        )
        decoded_norm_mse = F.mse_loss(decoded.float(), raw_action_norm.float())
        if self.decoded_loss_coef > 0:
            train_losses["decoded_action_loss"] = (
                decoded_norm_mse * self.decoded_loss_coef
            )
        else:
            metrics["decoded_action_loss"] = decoded_norm_mse.detach()

        reconstructed = self.action_ae.decode(
            latent_target,
            task_emb,
            traj_a_freq=traj_a_freq,
            traj_a_len=traj_a_len,
        )
        ae_recon = F.mse_loss(reconstructed.float(), raw_action_norm.float())
        train_losses["action_ae_reconstruction_loss"] = (
            ae_recon * self.action_ae_reconstruction_coef
        )

        ae_progress = self.action_ae.progress_loss_from_data(
            traj_a_freq,
            traj_a_len,
            total_length,
            task_emb,
        )
        train_losses["action_ae_progress_loss"] = (
            ae_progress * self.action_ae_progress_coef
        )

        with torch.no_grad():
            next_latent = self.action_ae.encode_target(next_raw_action_norm, task_emb)
        ae_dynamics = self.action_ae.dynamics_loss(
            latent_target,
            next_latent,
            task_emb,
            traj_a_freq=traj_a_freq,
            traj_a_len=traj_a_len,
        )
        train_losses["action_ae_dynamics_loss"] = (
            ae_dynamics * self.action_ae_dynamics_coef
        )

        if raw_action is not None and self._action_normalizer is not None:
            decoded_raw = self._action_normalizer.unnormalize_action_torch(
                decoded.detach().float()
            )
            metrics["decoded_raw_action_loss"] = F.mse_loss(
                decoded_raw.float(),
                raw_action.float(),
            ).detach()

        loss = sum(train_losses.values())
        return loss, {**train_losses, **metrics}


def build_joint_optimizer(module: JointLatentTrainModule, args):
    model = module.xvla
    vlm_params = list(model.vlm.parameters())
    soft_prompt_params = list(model.transformer.soft_prompt_hub.parameters())
    action_params = list(model.transformer.action_decoder.parameters()) + list(
        model.transformer.action_encoder.parameters()
    )
    exclude = set(map(id, vlm_params + soft_prompt_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    ae_params = [p for p in module.action_ae.parameters() if p.requires_grad]
    return AdamW(
        [
            {
                "name": "vlm",
                "params": vlm_params,
                "lr": 0.0,
                "weight_decay": args.weight_decay,
            },
            {
                "name": "transformer_core",
                "params": transformer_core_params,
                "lr": 0.0,
                "weight_decay": args.weight_decay,
            },
            {
                "name": "soft_prompts",
                "params": soft_prompt_params,
                "lr": args.learning_rate * args.learning_coef,
                "weight_decay": args.weight_decay,
            },
            {
                "name": "action_heads",
                "params": action_params,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "name": "action_ae",
                "params": ae_params,
                "lr": args.action_ae_learning_rate,
                "weight_decay": args.weight_decay,
            },
        ],
        betas=tuple(args.betas),
    )


def update_joint_group_lrs(optim, step, args):
    from training.utils import linear_warmup_cosine, set_group_lr

    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
        "action_ae": args.action_ae_learning_rate,
    }

    def schedule(step_idx, base_lr):
        return linear_warmup_cosine(
            step_idx,
            args.freeze_steps,
            args.warmup_steps,
            args.iters,
            base_lr,
            args.min_lr_ratio,
        )

    if step < args.freeze_steps:
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "soft_prompts", base["soft_prompts"])
        set_group_lr(optim, "action_heads", base["action_heads"])
        set_group_lr(optim, "action_ae", base["action_ae"])
    else:
        for name, base_lr in base.items():
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


def main(args):
    output_dir = Path(args.output_dir)
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=output_dir,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    accelerator.init_trackers("XVLA-Joint-Latent-Training")
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)

    set_seed(args.seed + accelerator.process_index)
    device = accelerator.device
    action_ae = load_trainable_tcla_action_ae(
        config_yaml=args.action_ae_config,
        ckpt_path=args.action_ae_ckpt,
        device=device,
    )
    logger.info(
        "Loaded trainable ActionAE: chunk=%s latent_tokens=%s latent_dim=%s raw_dim=%s",
        action_ae.chunk_size,
        action_ae.latent_action_tokens,
        action_ae.latent_action_dim,
        action_ae.action_dim,
    )

    config = XVLAConfig.from_pretrained(args.models)
    config.action_mode = "auto"
    config.real_action_dim = action_ae.latent_action_dim
    config.max_action_dim = action_ae.latent_action_dim
    config.num_actions = action_ae.latent_action_tokens

    model = XVLA.from_pretrained(
        args.models, config=config, ignore_mismatched_sizes=True
    )
    reinit_latent_action_head(model)
    processor = XVLAProcessor.from_pretrained(args.models)

    train_dataloader = create_dataloader(args, accelerator, action_ae)
    train_dataset = train_dataloader.dataset
    train_module = JointLatentTrainModule(
        xvla=model,
        action_ae=action_ae,
        latent_loss_coef=args.latent_loss_coef,
        decoded_loss_coef=args.decoded_loss_coef,
        action_ae_reconstruction_coef=args.action_ae_reconstruction_coef,
        action_ae_progress_coef=args.action_ae_progress_coef,
        action_ae_dynamics_coef=args.action_ae_dynamics_coef,
        action_normalizer=train_dataset.action_normalizer,
    )

    optim = build_joint_optimizer(train_module, args)
    train_module, optim = accelerator.prepare(train_module, optim)
    train_module.train()

    global_step, t0 = 0, time.time()
    logger.info(
        "Start %s joint latent training for %s iterations | world_size=%s",
        args.dataset,
        args.iters,
        accelerator.num_processes,
    )

    for batch in train_dataloader:
        batch = _move_tensor_batch(batch, device)
        lang = processor.encode_language(batch["language_instruction"])
        input_ids = lang["input_ids"].to(device, non_blocking=True)

        proprio = _pad_or_trim(batch["qpos"], action_ae.latent_action_dim)
        loss, loss_dict = train_module(
            input_ids=input_ids,
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
            domain_id=batch["domain_id"],
            proprio=proprio,
            raw_action_norm=batch["raw_action_norm"],
            task_emb=batch["task_emb"],
            traj_a_freq=batch["traj_a_freq"],
            traj_a_len=batch["traj_a_len"],
            total_length=batch["total_length"],
            next_raw_action_norm=batch["next_raw_action_norm"],
            raw_action=batch["raw_action"],
        )

        update_joint_group_lrs(optim, global_step, args)
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(train_module.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()
        if not accelerator.optimizer_step_was_skipped:
            accelerator.unwrap_model(
                train_module
            ).action_ae.model.update_target_encoder(global_step, args.iters)

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
                    "[%s/%s] loss=%.4f latent=%.4f decoded=%.4f ae_recon=%.4f ae_prog=%.4f ae_dyn=%.4f "
                    "lr_core=%.2e lr_ae=%.2e (%.2fs/it) CPU=%.2fGB GPU=%.2fGB",
                    global_step,
                    args.iters,
                    logs["loss_total"],
                    logs.get("latent_action_loss", 0.0),
                    logs.get("decoded_action_loss", 0.0),
                    logs.get("action_ae_reconstruction_loss", 0.0),
                    logs.get("action_ae_progress_loss", 0.0),
                    logs.get("action_ae_dynamics_loss", 0.0),
                    logs.get("lr_transformer_core", 0.0),
                    logs.get("lr_action_ae", 0.0),
                    dt,
                    cpu_mem,
                    gpu_mem,
                )

        global_step += 1
        if accelerator.is_main_process and (
            global_step == args.iters or global_step % args.save_interval == 0
        ):
            unwrapped = accelerator.unwrap_model(train_module)
            save_policy_checkpoint(
                unwrapped.xvla,
                processor,
                args,
                train_dataset,
                global_step,
                unwrapped.action_ae,
            )
        if global_step >= args.iters:
            break

    accelerator.end_training()
