# SPDX-License-Identifier: Apache-2.0

"""Sequence-level entropy estimates from a prompt's binary-reward rollouts."""

import math
from collections.abc import Sequence
from typing import Any

import torch

from areal.utils import stats_tracker
from areal.utils.rollout_metrics import CONDITIONAL_ENTROPY_KEYS, Metrics


def _tensor_response(
    rollout: dict[str, Any], reward: float | None
) -> tuple[float, float] | None:
    """Read a single response from a token-aligned native or cached trajectory."""
    if reward is None:
        rewards = rollout.get("original_rewards", rollout.get("rewards"))
        if not isinstance(rewards, torch.Tensor) or rewards.numel() != 1:
            return None
        reward = float(rewards)
    logp = rollout.get("logprobs")
    mask = rollout.get("loss_mask")
    if not isinstance(logp, torch.Tensor) or logp.ndim != 2 or logp.size(0) != 1:
        return None
    if not isinstance(mask, torch.Tensor) or mask.shape != logp.shape:
        return None
    mask = mask.bool()
    if "attention_mask" in rollout:
        mask = mask & rollout["attention_mask"].bool()
    if "turn_ids" in rollout and ((rollout["turn_ids"] > 0) & mask).any():
        return None
    nll = -logp.detach().masked_fill(~mask, 0).sum(dtype=torch.float64)
    return reward, float(nll)


def _single_response(rollout: dict[str, Any]) -> tuple[float, float] | None:
    if "input_ids" in rollout:
        return _tensor_response(rollout, rollout.get("rollout_reward"))
    if len(rollout) != 1:
        return None
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    interaction = next(iter(rollout.values()))
    if not isinstance(interaction, InteractionWithTokenLogpReward) or interaction.parent:
        return None
    references = (
        interaction.original_rollout_reward,
        interaction.rollout_reward,
        interaction.original_reward,
    )
    reward = next((value for value in references if value is not None), None)
    label = reward if reward is not None else interaction.reward
    if label is None:
        return None
    response = interaction.model_response
    if response is not None:
        if response.output_len != len(response.output_logprobs):
            return None
        return label, -sum(response.output_logprobs)
    if not interaction.has_tensor_data:
        return None
    # The v1 proxy can send only cached tensors, including the original reward.
    cached = interaction.to_tensor_dict()
    return _tensor_response(cached, reward if "original_rewards" in cached else label)


def binary_conditional_entropies(rollouts: Sequence[dict[str, Any] | None]) -> Metrics:
    """Compute one contribution per scored reply, in nats, for a single prompt.

    For class size K in a group of G, the finite-sample estimate of log P(class)
    is H[K-1] - H[G-1], with H[0] = 0. Missing rollout slots are unobserved.
    """
    surprisals: tuple[list[float], list[float]] = ([], [])
    for rollout in rollouts:
        if not rollout:
            continue
        sample = _single_response(rollout)
        if sample is None:
            return {}
        reward, nll = sample
        if reward not in (0, 1) or not math.isfinite(nll):
            return {}
        surprisals[int(reward)].append(nll)

    group_size = sum(map(len, surprisals))
    metrics: Metrics = {}
    for key, values in zip(CONDITIONAL_ENTROPY_KEYS, surprisals, strict=True):
        if values:
            correction = -math.fsum(1 / j for j in range(len(values), group_size))
            metrics[key] = [nll + correction for nll in values]
    return metrics


def record_conditional_entropies(metrics: Metrics) -> None:
    """Give every response one observation in the consumed training batch."""
    tracker = stats_tracker.get("rollout")
    for key, values in metrics.items():
        for value in values:
            tracker.scalar(**{key: value})
