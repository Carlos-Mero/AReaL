# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch


def infer_token_denominator(
    input_data: dict[str, Any],
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Infer the full token mask for stats logging.

    Context parallelism may slice intermediate tensors such as ``loss_mask`` or
    model outputs, while the original micro-batch metadata still describes the
    full logical sequence. Prefer that metadata for ``n_tokens`` so statistics
    stay consistent with and without context parallelism.
    """
    common_kwargs = {"dtype": torch.bool, "device": fallback.device}

    attention_mask = input_data.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return torch.ones_like(attention_mask, **common_kwargs)

    cu_seqlens = input_data.get("cu_seqlens")
    if isinstance(cu_seqlens, torch.Tensor) and cu_seqlens.numel() > 0:
        return torch.ones(int(cu_seqlens[-1].item()), **common_kwargs)

    input_ids = input_data.get("input_ids")
    # Tree-packed batches keep input_ids padded to tree size while token-level
    # stats stay at packed-token length. Only reuse input_ids when it already
    # matches the stat tensor shape.
    if isinstance(input_ids, torch.Tensor) and input_ids.shape == fallback.shape:
        return torch.ones_like(input_ids, **common_kwargs)

    return torch.ones_like(fallback, **common_kwargs)


def estimate_sequence_entropy(
    entropy: torch.Tensor,
    loss_mask: torch.Tensor,
    correct_response_mask: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Estimate per-sequence entropy from per-token entropy values.

    Returns sequence entropy sums, a valid-sequence mask, and optional correct /
    incorrect sequence masks when outcome labels are available.
    """
    entropy = entropy.float()
    loss_mask = loss_mask.bool()

    if entropy.ndim >= 2:
        token_entropy = torch.where(loss_mask, entropy, 0.0)
        sequence_entropy = token_entropy.sum(dim=-1)
        sequence_mask = loss_mask.any(dim=-1)
        correct_seq_mask = _sequence_outcome_mask(
            correct_response_mask,
            loss_mask,
            sequence_mask,
            sequence_entropy.shape,
        )
        incorrect_seq_mask = (
            sequence_mask & ~correct_seq_mask if correct_seq_mask is not None else None
        )
        return sequence_entropy, sequence_mask, correct_seq_mask, incorrect_seq_mask

    if (
        isinstance(cu_seqlens, torch.Tensor)
        and cu_seqlens.numel() > 1
        and int(cu_seqlens[-1].item()) == entropy.numel()
    ):
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(entropy.device)
        seq_ids = torch.repeat_interleave(
            torch.arange(seq_lens.numel(), device=entropy.device), seq_lens
        )
        sequence_entropy = torch.zeros(
            seq_lens.numel(), dtype=torch.float32, device=entropy.device
        )
        sequence_entropy.scatter_add_(
            0, seq_ids, torch.where(loss_mask, entropy, 0.0)
        )
        valid_counts = torch.zeros(
            seq_lens.numel(), dtype=torch.long, device=entropy.device
        )
        valid_counts.scatter_add_(0, seq_ids, loss_mask.long())
        sequence_mask = valid_counts > 0
        correct_seq_mask = _packed_sequence_outcome_mask(
            correct_response_mask, loss_mask, sequence_mask, seq_ids
        )
        incorrect_seq_mask = (
            sequence_mask & ~correct_seq_mask if correct_seq_mask is not None else None
        )
        return sequence_entropy, sequence_mask, correct_seq_mask, incorrect_seq_mask

    sequence_entropy = torch.where(loss_mask, entropy, 0.0).sum().view(1)
    sequence_mask = loss_mask.any().view(1)
    correct_seq_mask = None
    if (
        isinstance(correct_response_mask, torch.Tensor)
        and correct_response_mask.numel() == 1
    ):
        correct_seq_mask = sequence_mask & correct_response_mask.bool().view(1)
    incorrect_seq_mask = (
        sequence_mask & ~correct_seq_mask if correct_seq_mask is not None else None
    )
    return sequence_entropy, sequence_mask, correct_seq_mask, incorrect_seq_mask


def _sequence_outcome_mask(
    correct_response_mask: torch.Tensor | None,
    loss_mask: torch.Tensor,
    sequence_mask: torch.Tensor,
    sequence_shape: torch.Size,
) -> torch.Tensor | None:
    if not isinstance(correct_response_mask, torch.Tensor):
        return None

    correct_response_mask = correct_response_mask.bool()
    if correct_response_mask.shape == loss_mask.shape:
        return sequence_mask & correct_response_mask.any(dim=-1)
    if correct_response_mask.shape == sequence_shape:
        return sequence_mask & correct_response_mask
    return None


def _packed_sequence_outcome_mask(
    correct_response_mask: torch.Tensor | None,
    loss_mask: torch.Tensor,
    sequence_mask: torch.Tensor,
    seq_ids: torch.Tensor,
) -> torch.Tensor | None:
    if not isinstance(correct_response_mask, torch.Tensor):
        return None

    correct_response_mask = correct_response_mask.bool()
    if correct_response_mask.shape == sequence_mask.shape:
        return sequence_mask & correct_response_mask
    if correct_response_mask.shape != loss_mask.shape:
        return None

    correct_counts = torch.zeros(
        sequence_mask.numel(), dtype=torch.long, device=loss_mask.device
    )
    correct_counts.scatter_add_(0, seq_ids, (correct_response_mask & loss_mask).long())
    return sequence_mask & (correct_counts > 0)
