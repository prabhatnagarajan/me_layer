# MIT License
# Copyright (c) 2025 Brett Daley and Prabhat Nagarajan
# Copyright (c) 2026 Prabhat Nagarajan
#
# Portions derived from https://github.com/prabhatnagarajan/reg-duel-q:
# `semi_gradient_mse`, `RegularizedDuelingQLearning`

import pfrl
from pfrl.action_value import ActionValue
from me_layer.agents import DQN
from pfrl.replay_buffer import batch_experiences
import torch
from pfrl.utils.contexts import evaluating
from typing import Any, Dict, Optional, Tuple, Sequence, List, Callable
from pfrl.explorer import Explorer
from logging import Logger, getLogger
from pfrl.utils.batch_states import batch_states

from me_layer.q_networks import RDQNetwork

import torch.nn as nn


def semi_gradient_mse(targets, predictions, batch_accumulator):
    # Make sure all inputs have consistent, 1-dim shapes to avoid bugs
    targets = targets.squeeze(axis=-1)
    predictions = predictions.squeeze(axis=-1)
    assert targets.ndim == 1
    assert predictions.ndim == 1
    assert targets.shape == predictions.shape
    return (
        nn.functional.mse_loss(
            predictions, targets.detach(), reduction=batch_accumulator
        )
        / 2.0
    )


class RegularizedDuelingQLearning(DQN):
    def __init__(
        self,
        q_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,  # type: ignore  # somehow mypy complains
        replay_buffer: pfrl.replay_buffer.AbstractReplayBuffer,
        gamma: float,
        explorer: Explorer,
        gpu: Optional[int] = None,
        replay_start_size: int = 50000,
        minibatch_size: int = 32,
        update_interval: int = 1,
        target_update_interval: int = 10000,
        clip_delta: bool = True,
        phi: Callable[[Any], Any] = lambda x: x,
        target_update_method: str = "hard",
        soft_update_tau: float = 1e-2,
        n_times_update: int = 1,
        batch_accumulator: str = "mean",
        episodic_update_len: Optional[int] = None,
        logger: Logger = getLogger(__name__),
        batch_states: Callable[
            [Sequence[Any], torch.device, Callable[[Any], Any]], Any
        ] = batch_states,
        recurrent: bool = False,
        max_grad_norm: Optional[float] = None,
        reset_optimizer: bool = False,
        outdir: Optional[str] = None,
        beta=1e-3,
    ):
        self.beta = beta
        super().__init__(
            q_function,
            optimizer,
            replay_buffer,
            gamma,
            explorer,
            gpu,
            replay_start_size,
            minibatch_size,
            update_interval,
            target_update_interval,
            clip_delta,
            phi,
            target_update_method,
            soft_update_tau,
            n_times_update,
            batch_accumulator,
            episodic_update_len,
            logger,
            batch_states,
            recurrent,
            max_grad_norm,
            reset_optimizer,
            outdir,
        )
        assert self.batch_accumulator == "mean"

    def measure_action_gap(self) -> float:
        """Measure the action gap of Q-values.

        Returns:
            float: Action gap value.
        """
        with torch.no_grad(), evaluating(self.model):
            batch = self.replay_buffer.sample(self.stat_batch_size)
            batch_obs = batch_experiences(
                batch,
                device=self.device,
                phi=self.phi,
                gamma=self.gamma,
                batch_states=self.batch_states,
            )["state"]
            assert not self.recurrent
            action_values = self.model(batch_obs)[0].q_values
            top2 = torch.topk(action_values, 2, dim=-1).values
            max_values = torch.max(top2, dim=-1).values
            min_values = torch.min(top2, dim=-1).values
        action_gap = torch.mean(max_values - min_values).item()
        return action_gap

    def update(
        self, experiences: List[List[Dict[str, Any]]], errors_out: Optional[list] = None
    ) -> None:  # TODO: check
        exp_batch = batch_experiences(
            experiences,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states,
        )
        if self.do_measure_interference:
            interference_batch = self.replay_buffer.sample(self.stat_batch_size)
            interference_exp_batch = batch_experiences(
                interference_batch,
                device=self.device,
                phi=self.phi,
                gamma=self.gamma,
                batch_states=self.batch_states,
            )
            with torch.no_grad(), evaluating(self.model):
                pre_update_holdout_loss = self.compute_unaccumulated_loss(
                    interference_exp_batch
                )
        loss = self._compute_loss(exp_batch, errors_out=errors_out)
        self.loss_record.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            pfrl.utils.clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_t += 1
        if self.do_measure_churn:
            churn = self.measure_churn()
            self.churn_record.append(churn)
        if self.do_measure_interference:
            interference, backward_transfer = self.measure_interference(
                interference_exp_batch, pre_update_holdout_loss
            )
            self.interference_record.append(interference)
            self.backward_transfer_record.append(backward_transfer)

    def _compute_y_and_t(
        self, exp_batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = exp_batch["reward"].shape[0]
        batch_actions = exp_batch["action"]
        # Compute Q-values for current states
        batch_state = exp_batch["state"]
        qout, _ = self.model.forward(batch_state, return_adv=False)
        qout = torch.reshape(qout.evaluate_actions(batch_actions), (batch_size, 1))
        q_target = self._compute_target_values(exp_batch)

        return qout, q_target

    def _compute_loss(
        self, exp_batch: Dict[str, Any], errors_out: Optional[list] = None
    ) -> torch.Tensor:
        assert isinstance(self.model, RDQNetwork)
        qout, vout, advout = self.model.forward(exp_batch["state"], return_adv=True)
        q = qout.evaluate_actions(exp_batch["action"])

        self.q_record.extend(q.detach().cpu().numpy().ravel())

        assert errors_out is None
        assert "weights" not in exp_batch

        q_target = self._compute_target_values(exp_batch)
        l2_penalty = self._l2_penalty(qout, vout, advout)
        q_loss = semi_gradient_mse(q_target, q, self.batch_accumulator)
        return q_loss + l2_penalty

    def _compute_target_values(self, exp_batch: Dict[str, Any]) -> torch.Tensor:
        next_qout = self.target_model.forward_q(exp_batch["next_state"])
        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]
        return batch_rewards + discount * (1.0 - batch_terminal) * next_qout.max

    def _l2_penalty(self, qout, vout, advout):
        v = vout.q_values.squeeze(dim=-1)
        sum_squared_adv = torch.sum(torch.square(advout.q_values), dim=-1)
        assert v.ndim == 1
        assert sum_squared_adv.ndim == 1
        assert v.shape == sum_squared_adv.shape
        return self.beta * torch.mean(0.5 * (torch.square(v) + sum_squared_adv))

    def _evaluate_model_and_update_recurrent_states(
        self, batch_obs: Sequence[Any]
    ) -> ActionValue:
        batch_xs = self.batch_states(batch_obs, self.device, self.phi)
        assert not self.recurrent
        batch_av = self.model.forward_q(batch_xs)
        return batch_av
