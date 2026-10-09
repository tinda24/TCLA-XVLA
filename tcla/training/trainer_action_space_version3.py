from pathlib import Path

import torch

from tcla.training.action_ae_evaluator_version3 import ActionAETrainingEvaluatorVersion3
from tcla.training.trainer_utils import TrainerUtils


class ActionSpaceTrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        run_dir,
        dataset,
    ):
        self.cfg, self.model, self.dataloader = cfg, model, dataloader
        self.optimizer, self.lr_scheduler, self.accelerator = (
            optimizer,
            lr_scheduler,
            accelerator,
        )
        self.run_dir, self.dataset = Path(run_dir), dataset
        self.completed_steps = 0
        self.train_mode_eval_every_steps = int(
            cfg.trainer.get("train_mode_eval_every_steps", 0)
        )
        self.evaluator = None
        if accelerator.is_main_process and self.train_mode_eval_every_steps > 0:
            self.evaluator = ActionAETrainingEvaluatorVersion3(
                cfg=cfg,
                dataset=dataset,
                run_dir=self.run_dir,
                device=accelerator.device,
                dpi=int(cfg.trainer.get("eval_plot_dpi", 150)),
                batch_size=int(cfg.trainer.get("eval_batch_size", 256)),
                tsne_perplexity=float(cfg.trainer.get("eval_tsne_perplexity", 30.0)),
                eval_loss_max_windows=cfg.trainer.get("eval_loss_max_windows", None),
            )
        self.print_module_parameters(accelerator.unwrap_model(model))
        self.load_training_state_if_needed()
        self.save_initial_configs()
        self.save_dataset_statistics()

    def train(self):
        max_steps = int(self.cfg.trainer.max_train_steps)
        log_every = int(self.cfg.trainer.log_every_steps)
        save_every = int(self.cfg.trainer.save_every_steps)
        if max_steps < self.completed_steps or max_steps <= 0 or log_every <= 0:
            raise ValueError("Invalid training steps or logging interval")
        self.model.train()
        iterator = iter(self.dataloader)
        self.optimizer.zero_grad(set_to_none=True)
        last_saved = None
        while self.completed_steps < max_steps:
            with self.accelerator.accumulate(self.model):
                outputs = self.model(next(iterator))
                self.accelerator.backward(outputs["total_loss"])
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            if (
                not self.accelerator.sync_gradients
                or self.accelerator.optimizer_step_was_skipped
            ):
                continue
            model = self.accelerator.unwrap_model(self.model)
            momentum = model.update_target_encoder(self.completed_steps, max_steps)
            self.completed_steps += 1
            step = self.completed_steps
            if step % log_every == 0 or step == max_steps:
                metrics = self._metrics_to_float(outputs)
                metrics.update(
                    ema_momentum=float(momentum),
                    learning_rate=float(self.lr_scheduler.get_last_lr()[0]),
                )
                self.log_metrics(step, metrics)
            if save_every > 0 and step % save_every == 0:
                self.accelerator.wait_for_everyone()
                self.save_checkpoint(step)
                last_saved = step
            if (
                self.train_mode_eval_every_steps > 0
                and step % self.train_mode_eval_every_steps == 0
            ):
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.evaluator.run_train_mode_eval(model, step)
                self.model.train()
                self.accelerator.wait_for_everyone()
        self.accelerator.wait_for_everyone()
        if last_saved != self.completed_steps:
            self.save_checkpoint(self.completed_steps)

    def load_training_state_if_needed(self):
        if not self.cfg.trainer.get("is_resume", False):
            return
        path = self.cfg.trainer.get("resume_checkpoint")
        if not path:
            raise ValueError("trainer.resume_checkpoint is required to resume")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.accelerator.unwrap_model(self.model).load_state_dict(
            state["model"], strict=True
        )
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.completed_steps = int(state["completed_steps"])

    def save_checkpoint(self, step):
        if not self.accelerator.is_main_process:
            return
        path = self.run_dir / "checkpoints" / f"steps_{step}_pytorch_model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.accelerator.save(
            {
                "completed_steps": int(step),
                "model": self.accelerator.unwrap_model(self.model).state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            },
            path,
        )
        self.write_summary(step)
        self.accelerator.print(f"Saved AE checkpoint: {path}")
