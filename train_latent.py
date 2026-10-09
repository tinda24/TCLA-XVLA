from __future__ import annotations

import os
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

from policy_data.episodes import create_dataloader
from training.checkpoints import save_policy_checkpoint
from models.latent_action_ae import load_frozen_tcla_action_ae
from models.configuration_xvla import XVLAConfig
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor
from training.utils import build_optimizer, get_logger, set_seed, update_group_lrs


def reinit_latent_action_head(model: XVLA) -> None:

    tr = model.transformer
    nn.init.xavier_uniform_(tr.action_encoder.fc.weight)
    nn.init.zeros_(tr.action_encoder.bias.weight)
    nn.init.xavier_uniform_(tr.action_decoder.fc.weight)
    nn.init.zeros_(tr.action_decoder.bias.weight)


def _move_tensor_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def _pad_or_trim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.size(-1) == dim:
        return x
    if x.size(-1) > dim:
        return x[..., :dim]
    pad = x.new_zeros(*x.shape[:-1], dim - x.size(-1))
    return torch.cat([x, pad], dim=-1)


class LatentTrainModule(nn.Module):
    def __init__(
        self,
        xvla: XVLA,
        action_ae,
        latent_loss_coef: float,
        decoded_loss_coef: float,
        action_normalizer=None,
    ):
        super().__init__()
        self.xvla = xvla

        self._action_ae = action_ae
        self._action_normalizer = action_normalizer
        self.latent_loss_coef = float(latent_loss_coef)
        self.decoded_loss_coef = float(decoded_loss_coef)

    def _unnormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        if self._action_normalizer is None:
            return action_norm
        return self._action_normalizer.unnormalize_action_torch(action_norm)

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
        raw_action: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            latent_target = self._action_ae.encode(raw_action_norm, task_emb)

        enc = self.xvla.forward_vlm(input_ids, image_input, image_mask)
        batch = input_ids.shape[0]
        t = (
            torch.rand(1, device=input_ids.device)
            + torch.arange(batch, device=input_ids.device) / batch
        ) % (1 - 1e-5)
        action_noisy = torch.randn_like(latent_target) * t.view(
            -1, 1, 1
        ) + latent_target * (1 - t).view(-1, 1, 1)
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
            F.mse_loss(pred_latent.float(), latent_target.float())
            * self.latent_loss_coef
        )
        train_losses = {"latent_action_loss": latent_loss}
        metrics = {}

        decode_ctx = (
            torch.enable_grad() if self.decoded_loss_coef > 0 else torch.no_grad()
        )
        with decode_ctx:
            decoded = self._action_ae.decode(
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

        if raw_action is not None:
            decoded_raw = self._unnormalize_action(decoded.detach().float())
            metrics["decoded_raw_action_loss"] = F.mse_loss(
                decoded_raw.float(),
                raw_action.float(),
            ).detach()

        loss = sum(train_losses.values())
        return loss, {**train_losses, **metrics}


def main(args):
    output_dir = Path(args.output_dir)
    accelerator = Accelerator(log_with="tensorboard", project_dir=output_dir)
    accelerator.init_trackers("XVLA-Latent-Training")
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)

    set_seed(args.seed + accelerator.process_index)
    device = accelerator.device
    action_ae = load_frozen_tcla_action_ae(
        config_yaml=args.action_ae_config,
        ckpt_path=args.action_ae_ckpt,
        device=device,
    )
    logger.info(
        "Loaded frozen ActionAE: chunk=%s latent_tokens=%s latent_dim=%s raw_dim=%s",
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
    train_module = LatentTrainModule(
        xvla=model,
        action_ae=action_ae,
        latent_loss_coef=args.latent_loss_coef,
        decoded_loss_coef=args.decoded_loss_coef,
        action_normalizer=train_dataset.action_normalizer,
    )

    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_soft=args.learning_coef,
    )
    train_module, optim = accelerator.prepare(train_module, optim)
    train_module.train()

    global_step, t0 = 0, time.time()
    logger.info(
        "Start %s latent training for %s iterations | world_size=%s",
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
            raw_action=batch["raw_action"],
        )

        update_group_lrs(optim, global_step, args)
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(train_module.parameters(), args.max_grad_norm)
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
                    "[%s/%s] loss=%.4f latent=%.4f decoded=%.4f decoded_raw=%.4f lr_core=%.2e (%.2fs/it) CPU=%.2fGB GPU=%.2fGB",
                    global_step,
                    args.iters,
                    logs["loss_total"],
                    logs.get("latent_action_loss", 0.0),
                    logs.get("decoded_action_loss", 0.0),
                    logs.get("decoded_raw_action_loss", 0.0),
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
                accelerator.unwrap_model(train_module).xvla,
                processor,
                args,
                train_dataset,
                global_step,
                action_ae,
            )
        if global_step >= args.iters:
            break

    accelerator.end_training()
