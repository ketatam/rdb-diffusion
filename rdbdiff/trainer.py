import math
import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

import wandb
from rdbdiff.model import GRDM
from rdbdiff.third_party.tab_ddpm import update_ema


class GRDMTrainer:
    def __init__(
        self,
        diffusion_model: GRDM,
        train_dataloader,
        lr,
        weight_decay,
        steps,
        save_dir: str,
        device=torch.device("cuda"),
        train_batch_size: int = 4096,
        n_total_target_nodes: int = None,
        max_epochs: int = None,
        target_node_types: list[str] = None,
    ):
        self.diffusion_model = diffusion_model
        # Exponential Moving Average (EMA) model is a copy of the denoising model
        # that is not trained but updated as an EMA of the trained model.
        self.ema_model = deepcopy(self.diffusion_model._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        # data iterator that yields batches indefinitely from the datalaoder
        self.train_iter = get_iter_from_dataloader(train_dataloader)

        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.diffusion_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # number of steps per epoch is determined by the dataloader size
        self.steps_per_epoch = len(train_dataloader)
        # it should hold that steps_per_epoch * train_batch_size >= n_total_target_nodes
        assert self.steps_per_epoch == math.ceil(
            n_total_target_nodes / train_batch_size
        )
        if max_epochs is not None:
            # limit the total number of steps to steps_per_epoch * max_epochs
            self.steps = min(steps, self.steps_per_epoch * max_epochs)
        else:
            self.steps = steps

        self.save_dir = save_dir
        self.device = device
        self.train_batch_size = train_batch_size
        self.n_total_target_nodes = n_total_target_nodes
        self.target_node_types = target_node_types

        self.loss_history = pd.DataFrame(columns=["step", "loss"])

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_training_step(self, batch):
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        loss, node_type_to_loss = self.diffusion_model.relational_gaussian_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss, node_type_to_loss

    def run_training_loop(self):
        training_start_time = time.time()
        self.diffusion_model.train()

        step = 0
        epoch = 0

        node_type_to_curr_count = {node_type: 0 for node_type in self.target_node_types}
        node_type_to_curr_loss = {
            node_type: 0.0 for node_type in self.target_node_types
        }

        best_epoch = 0
        best_loss = float("inf")

        epoch_start_time = time.time()
        while step < self.steps:
            batch = next(self.train_iter)

            batch_loss, batch_node_type_to_loss = self._run_training_step(batch)
            self._anneal_lr(step)

            for node_type, loss_node_type in batch_node_type_to_loss.items():
                node_type_to_curr_count[node_type] += loss_node_type.size(0)
                node_type_to_curr_loss[node_type] += loss_node_type.sum().item()

            wandb.log({f"step": step})
            wandb.log({f"batch_loss": batch_loss.item()})

            if (step + 1) % self.steps_per_epoch == 0:
                # end of epoch
                epoch_end_time = time.time()
                epoch_total_time = epoch_end_time - epoch_start_time

                assert (
                    sum(node_type_to_curr_count.values()) == self.n_total_target_nodes
                )
                epoch_loss = 0.0
                for node_type in self.target_node_types:
                    epoch_loss += (
                        node_type_to_curr_loss[node_type]
                        / node_type_to_curr_count[node_type]
                    )
                epoch_loss = np.around(epoch_loss, 4)

                print(
                    f"Epoch: {epoch}, Step: {(step + 1)}/{self.steps}, Loss: {epoch_loss}"
                )
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1,
                    epoch_loss,
                ]
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss

                wandb.log({f"epoch": epoch})
                wandb.log({f"epoch_loss": epoch_loss})
                wandb.log({f"epoch_time": epoch_total_time})
                wandb.log(
                    {f"average_step_time": epoch_total_time / self.steps_per_epoch}
                )
                wandb.log({f"best_epoch": best_epoch})
                wandb.log({f"best_loss": best_loss})

                node_type_to_curr_count = {
                    node_type: 0 for node_type in self.target_node_types
                }
                node_type_to_curr_loss = {
                    node_type: 0.0 for node_type in self.target_node_types
                }
                epoch += 1
                epoch_start_time = time.time()

            update_ema(
                self.ema_model.parameters(),
                self.diffusion_model._denoise_fn.parameters(),
            )
            step += 1

        # end of training
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        wandb.log({f"total_training_time": total_training_time})

    def save_model_and_loss_history(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.loss_history.to_csv(os.path.join(self.save_dir, "loss.csv"), index=False)
        torch.save(
            self.diffusion_model._denoise_fn.state_dict(),
            os.path.join(self.save_dir, "model.pt"),
        )
        torch.save(
            self.ema_model.state_dict(), os.path.join(self.save_dir, "model_ema.pt")
        )

    def is_model_already_trained(self):
        return os.path.exists(os.path.join(self.save_dir, "model_ema.pt"))

    def load_model(self):
        self.diffusion_model._denoise_fn.load_state_dict(
            torch.load(os.path.join(self.save_dir, "model.pt"), weights_only=True)
        )
        self.ema_model.load_state_dict(
            torch.load(os.path.join(self.save_dir, "model_ema.pt"), weights_only=True)
        )


def get_iter_from_dataloader(dataloader):
    while True:
        yield from dataloader
