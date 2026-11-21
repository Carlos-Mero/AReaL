# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import torch

from areal.utils import stats_tracker
from areal.utils.conditional_kl import estimate_batch_ckl
from areal.utils.data import concat_batch

MOPD_CONTRIBUTIONS_KEY = "_mopd_teacher_contributions"


def _record_teacher_ckl(
    trajectory: dict[str, Any], contributions: dict[str, dict[str, Any]]
) -> None:
    """Average the individual teacher CKLs within this routed prompt."""
    if (
        not {"input_ids", "attention_mask", "loss_mask", "logprobs", "rewards"}
        <= trajectory.keys()
    ):
        return
    _, meta = concat_batch([trajectory])
    if meta.logical_group_sizes[0] < 2:
        return
    total_weight = sum(c["weight"] for c in contributions.values())
    for contribution in contributions.values():
        ckl = estimate_batch_ckl(trajectory, contribution["logp"], meta)
        stats_tracker.weighted_mean(
            "ppo_actor/student_teacher_ckl",
            ckl.value[0],
            ckl.available[0] * (contribution["weight"] / total_weight),
        )


def aggregate_mopd_targets(
    data: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Aggregate raw weighted teacher log-probabilities on an actor DP head."""
    if data is None:
        return None
    for trajectory in data:
        route = trajectory.get("mopd_route")
        contributions = trajectory.pop(MOPD_CONTRIBUTIONS_KEY, None)
        if not isinstance(contributions, dict) or not contributions:
            raise ValueError(
                f"MOPD route {route!r} has no positive teacher contributions"
            )

        logp_sum: torch.Tensor | None = None
        weight_sum: torch.Tensor | None = None
        for teacher_id, contribution in contributions.items():
            if not isinstance(contribution, dict):
                raise TypeError(
                    f"MOPD contribution from {teacher_id!r} must be a mapping"
                )
            teacher_logp = contribution.get("logp")
            weight = contribution.get("weight")
            if not isinstance(teacher_logp, torch.Tensor):
                raise TypeError(
                    f"MOPD contribution from {teacher_id!r} has no tensor logp"
                )
            if (
                not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isfinite(weight)
                or weight <= 0
            ):
                raise ValueError(
                    f"MOPD contribution weight from {teacher_id!r} must be "
                    "finite and positive"
                )
            if logp_sum is not None and teacher_logp.shape != logp_sum.shape:
                raise ValueError(
                    f"MOPD teacher logp shape mismatch for route {route!r}: "
                    f"expected {tuple(logp_sum.shape)}, got {tuple(teacher_logp.shape)}"
                )
            weighted_logp = teacher_logp * weight
            token_weight = torch.full_like(teacher_logp, weight)
            logp_sum = weighted_logp if logp_sum is None else logp_sum + weighted_logp
            weight_sum = (
                token_weight if weight_sum is None else weight_sum + token_weight
            )

        assert logp_sum is not None and weight_sum is not None
        _record_teacher_ckl(trajectory, contributions)
        trajectory["mopd_teacher_logp_sum"] = logp_sum
        trajectory["mopd_teacher_weight_sum"] = weight_sum
        trajectory.pop("mopd_route", None)
    return data
