# SPDX-License-Identifier: Apache-2.0

"""Success-conditioned sequence KL for RL and on-policy distillation."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from areal.utils import stats_tracker
from areal.utils.data import TrajBatchMeta
from areal.utils.stats_tracker import ReduceType


@dataclass
class ConditionalKL:
    signal: torch.Tensor  # Detached policy-gradient coefficient per physical row.
    value: torch.Tensor  # Plug-in divergence per prompt.
    available: torch.Tensor  # Prompts with at least two successful responses.

    def record(self, name: str) -> None:
        stats_tracker.stat_compact(self.available, ReduceType.AVG, **{name: self.value})


def response_prediction_mask(data: Mapping[str, Any]) -> torch.Tensor:
    """Positions predicting generated tokens, excluding padding and wraparound."""
    mask = (data["loss_mask"].bool() & data["attention_mask"].bool()).roll(-1, dims=-1)
    mask[:, -1] = False
    return mask


@torch.no_grad()
def conditional_kl(
    actor_logp: torch.Tensor,
    target_logp: torch.Tensor,
    loss_mask: torch.Tensor,
    raw_rewards: torch.Tensor,
    meta: TrajBatchMeta,
    *,
    require_outcomes: bool = False,
) -> ConditionalKL:
    """Estimate CKL on a batch of prompt groups with next-token-aligned scores.

    For positive samples d = log p(y) - log q(y). The training signal is
    (d - mean(d)) / N. The scalar KL additionally needs the relative success
    probability q(success) / p(success), estimated by mean(exp(-d)).
    """
    if actor_logp.shape != target_logp.shape or actor_logp.shape != loss_mask.shape:
        raise ValueError(
            "CKL log probabilities and loss mask must have matching shapes"
        )
    rows, _ = actor_logp.shape
    if raw_rewards.shape != (rows,) or sum(meta.traj_group_sizes) != rows:
        raise ValueError("CKL rewards and group metadata must match the batch rows")
    mask = loss_mask.bool()
    score_dtype = torch.promote_types(
        torch.promote_types(actor_logp.dtype, target_logp.dtype), torch.float32
    )
    row_d = (
        torch.where(mask, actor_logp.to(score_dtype) - target_logp.to(score_dtype), 0.0)
        .sum(-1)
        .double()
    )
    groups = meta.rollout_groups or [None] * meta.n_trajs
    row_counts = []
    references = []
    for rows_in_prompt, group in zip(meta.traj_group_sizes, groups, strict=True):
        if group is None:
            row_counts.extend([1] * rows_in_prompt)
            references.extend([None] * rows_in_prompt)
        else:
            row_counts.extend(group.row_counts)
            originals = group.original_rewards or group.rewards
            references.extend(
                raw if raw is not None else ref
                for raw, ref in zip(originals, group.rewards, strict=True)
            )

    # Each sampled response can span multiple training rows.
    device = row_d.device
    lengths = torch.tensor(row_counts, dtype=torch.long, device=device)
    response = torch.arange(len(lengths), device=device).repeat_interleave(
        lengths, output_size=rows
    )
    starts = lengths.cumsum(0) - lengths
    d = row_d.new_zeros(len(lengths)).scatter_add_(0, response, row_d)
    active = torch.zeros_like(d).scatter_add_(0, response, mask.any(-1).double()) > 0
    reference = d.new_tensor([float("nan") if r is None else r for r in references])
    positive = torch.where(reference.isnan(), raw_rewards[starts], reference) > 0
    # Without an explicit rollout outcome, all its rows must agree on success.
    ambiguous = reference.isnan()[response] & (
        ~raw_rewards.isfinite() | ((raw_rewards > 0) != positive[response])
    )
    if require_outcomes:
        torch._assert_async(
            ~ambiguous.any(),
            "CKL needs an explicit rollout_reward for ambiguous row outcomes",
        )
    positive &= active

    group_sizes = torch.tensor(
        meta.logical_group_sizes, dtype=torch.long, device=device
    )
    prompt = torch.arange(len(group_sizes), device=device).repeat_interleave(
        group_sizes, output_size=len(lengths)
    )
    zeros = d.new_zeros(len(group_sizes))
    count = zeros.scatter_add(0, prompt, positive.double())
    available = (count >= 2) & (
        zeros.scatter_add(0, prompt[response], ambiguous.double()) == 0
    )
    divisor = count.clamp_min(1)
    mean = zeros.scatter_add(0, prompt, torch.where(positive, d, 0.0)) / divisor
    centered = torch.where(positive, d - mean[prompt], 0.0)
    signal = torch.where(available[prompt], centered / divisor[prompt], 0.0)

    # Stable log(mean(exp(-centered))) without exponentiating sequence ratios.
    neg = (-centered).masked_fill(~positive, -torch.inf)
    peak = zeros.scatter_reduce(0, prompt, neg, reduce="amax", include_self=True)
    exp_sum = zeros.scatter_add(0, prompt, (neg - peak[prompt]).exp())
    value = peak + (exp_sum.clamp_min(1.0) / divisor).log()
    return ConditionalKL(signal[response], value.clamp_min(0.0), available)


def estimate_batch_ckl(
    data: Mapping[str, Any],
    target_logp: torch.Tensor,
    meta: TrajBatchMeta,
    *,
    require_outcomes: bool = False,
) -> ConditionalKL:
    """Score the raw rollout before reward shaping or next-token realignment."""
    logp = data.get("prox_logp")
    if logp is None:
        logp = data["logprobs"].roll(-1, dims=-1)
    return conditional_kl(
        logp,
        target_logp,
        response_prediction_mask(data),
        data.get("original_rewards", data["rewards"]),
        meta,
        require_outcomes=require_outcomes,
    )
