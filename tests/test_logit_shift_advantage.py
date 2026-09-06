# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from areal.api.cli_args import NormConfig, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import Normalization
from areal.utils.functional import logit_shift_advantage_shaping


def test_logit_shift_advantage_centers_non_final_tokens_only():
    """Test that the default mode preserves the final scaled token advantage."""
    advantages = torch.tensor([[1.0, 1.0, 100.0]])
    policy_logprobs = torch.tensor([[math.log(0.5), math.log(0.25), math.log(0.001)]])
    loss_mask = torch.tensor([[True, True, False]])

    shaped, weights, clipped = logit_shift_advantage_shaping(
        advantages=advantages,
        policy_logprobs=policy_logprobs,
        loss_mask=loss_mask,
        ls_clip=4.0,
    )

    torch.testing.assert_close(
        weights, torch.tensor([[2.0, 4.0, 0.0]]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        shaped, torch.tensor([[0.0, 1.0, 0.0]]), rtol=1e-6, atol=1e-6
    )
    assert torch.equal(clipped, torch.zeros_like(loss_mask))


def test_logit_shift_legacy_centers_scaled_valid_token_advantages():
    """Test that legacy mode retains global valid-token mean centering."""
    advantages = torch.tensor([[1.0, 1.0, 100.0]])
    policy_logprobs = torch.tensor([[math.log(0.5), math.log(0.25), math.log(0.001)]])
    loss_mask = torch.tensor([[True, True, False]])

    shaped, weights, clipped = logit_shift_advantage_shaping(
        advantages=advantages,
        policy_logprobs=policy_logprobs,
        loss_mask=loss_mask,
        ls_clip=4.0,
        centering="all",
    )

    torch.testing.assert_close(
        weights, torch.tensor([[2.0, 4.0, 0.0]]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        shaped, torch.tensor([[-0.25, 0.25, 0.0]]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        shaped[loss_mask].mean(), torch.tensor(0.0), rtol=0.0, atol=1e-6
    )
    assert torch.equal(clipped, torch.zeros_like(loss_mask))


@pytest.mark.parametrize(
    ("center", "expected"),
    [
        (True, torch.tensor([[-0.25, 0.25, 0.0]])),
        (False, torch.tensor([[0.5, 1.0, 0.0]])),
    ],
)
def test_logit_shift_supports_legacy_center_argument(center, expected):
    """Test backward compatibility for explicit callers of the old boolean API."""
    shaped, _, _ = logit_shift_advantage_shaping(
        advantages=torch.tensor([[1.0, 1.0, 100.0]]),
        policy_logprobs=torch.tensor(
            [[math.log(0.5), math.log(0.25), math.log(0.001)]]
        ),
        loss_mask=torch.tensor([[True, True, False]]),
        ls_clip=4.0,
        center=center,
    )

    torch.testing.assert_close(shaped, expected, rtol=1e-6, atol=1e-6)


def test_logit_shift_advantage_clips_extreme_probability_and_masks_padding():
    """Test that inverse weights remain finite and ignore masked positions."""
    shaped, weights, clipped = logit_shift_advantage_shaping(
        advantages=torch.tensor([[2.0, -1.0, 50.0]]),
        policy_logprobs=torch.tensor([[-1000.0, 0.0, -1000.0]]),
        loss_mask=torch.tensor([[True, True, False]]),
        ls_clip=10.0,
    )

    torch.testing.assert_close(
        weights, torch.tensor([[10.0, 1.0, 0.0]]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        shaped, torch.tensor([[0.0, -0.1, 0.0]]), rtol=1e-6, atol=1e-6
    )
    assert torch.equal(clipped, torch.tensor([[True, False, False]]))
    assert torch.isfinite(shaped).all()


def test_actor_logit_shift_advantage_runs_after_gae_and_preserves_returns():
    """Test GAE output is token-wise shaped without changing return targets."""
    config = PPOActorConfig(
        adv_norm=NormConfig(mean_level="logit-shift", std_level=None),
        recompute_logprob=True,
        kl_ctl=0.0,
        ls_clip=4.0,
    )
    actor = PPOActor(config, object())
    batch = {
        "input_ids": torch.zeros((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        "logprobs": torch.zeros((1, 3)),
        "prox_logp": torch.tensor([[math.log(0.5), math.log(0.25), 0.0]]),
        "rewards": torch.tensor([1.0]),
    }

    result = actor._compute_advantages(batch)

    torch.testing.assert_close(
        result["returns"], torch.tensor([[1.0, 1.0, 0.0]]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        result["advantages"],
        torch.tensor([[0.0, 1.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["logit_shift_advantage_weight"],
        torch.tensor([[2.0, 4.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["advantages"][:, 1], torch.tensor([1.0]), rtol=1e-6, atol=1e-6
    )


def test_actor_logit_shift_applies_scaling_after_non_final_centering():
    """Test that optional batch scaling follows non-final token centering."""
    config = PPOActorConfig(
        adv_norm=NormConfig(
            mean_level="logit-shift", std_level="batch", std_unbiased=True
        ),
        recompute_logprob=True,
        kl_ctl=0.0,
        ls_clip=4.0,
    )
    actor = PPOActor(config, object())
    batch = {
        "input_ids": torch.zeros((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        "logprobs": torch.zeros((1, 3)),
        "prox_logp": torch.tensor([[math.log(0.5), math.log(0.25), 0.0]]),
        "rewards": torch.tensor([1.0]),
    }

    result = actor._compute_advantages(batch)

    expected = torch.tensor([[0.0, 1.0 / (1.0 + config.adv_norm.eps), 0.0]])
    torch.testing.assert_close(result["advantages"], expected, rtol=1e-5, atol=1e-5)


def test_logit_shift_excludes_each_sequence_final_token_from_global_mean():
    """Test variable-length sequences each preserve their final valid token."""
    advantages = torch.ones((2, 3))
    policy_logprobs = torch.tensor(
        [
            [math.log(0.5), math.log(0.25), 0.0],
            [0.0, math.log(0.5), math.log(0.25)],
        ]
    )
    loss_mask = torch.tensor([[True, True, False], [True, True, True]])

    shaped, _, _ = logit_shift_advantage_shaping(
        advantages=advantages,
        policy_logprobs=policy_logprobs,
        loss_mask=loss_mask,
        ls_clip=4.0,
    )

    expected = torch.tensor(
        [
            [1.0 / 12.0, 1.0, 0.0],
            [-1.0 / 6.0, 1.0 / 12.0, 1.0],
        ]
    )
    torch.testing.assert_close(shaped, expected, rtol=1e-6, atol=1e-6)


def test_logit_shift_can_center_non_final_tokens_within_each_sequence():
    """Test per-sequence centering with variable, singleton, and empty masks."""
    advantages = torch.ones((4, 3))
    policy_logprobs = torch.tensor(
        [
            [math.log(0.5), math.log(0.25), 0.0],
            [0.0, math.log(0.5), math.log(0.25)],
            [0.0, math.log(0.25), 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    loss_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [False, True, False],
            [False, False, False],
        ]
    )

    shaped, _, _ = logit_shift_advantage_shaping(
        advantages=advantages,
        policy_logprobs=policy_logprobs,
        loss_mask=loss_mask,
        ls_clip=4.0,
        centering="exclude-last-per-sequence",
    )

    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [-0.125, 0.125, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(shaped, expected, rtol=1e-6, atol=1e-6)


def test_logit_shift_handles_noncontiguous_single_token_and_empty_masks():
    """Test last-valid-token detection for sparse and degenerate sequence masks."""
    advantages = torch.ones((3, 3))
    policy_logprobs = torch.full((3, 3), math.log(0.25))
    loss_mask = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
            [False, False, False],
        ]
    )

    shaped, weights, clipped = logit_shift_advantage_shaping(
        advantages=advantages,
        policy_logprobs=policy_logprobs,
        loss_mask=loss_mask,
        ls_clip=4.0,
    )

    torch.testing.assert_close(
        shaped,
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        weights,
        4.0 * loss_mask,
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.equal(clipped, torch.zeros_like(loss_mask))


def test_actor_logit_shift_legacy_preserves_centered_behavior():
    """Test that the legacy config reproduces the previous centered advantages."""
    config = PPOActorConfig(
        adv_norm=NormConfig(mean_level="logit-shift-legacy", std_level=None),
        recompute_logprob=True,
        kl_ctl=0.0,
        ls_clip=4.0,
    )
    actor = PPOActor(config, object())
    batch = {
        "input_ids": torch.zeros((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        "logprobs": torch.zeros((1, 3)),
        "prox_logp": torch.tensor([[math.log(0.5), math.log(0.25), 0.0]]),
        "rewards": torch.tensor([1.0]),
    }

    result = actor._compute_advantages(batch)

    torch.testing.assert_close(
        result["advantages"],
        torch.tensor([[-0.25, 0.25, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("mode", ["logit-shift", "logit-shift-legacy"])
def test_logit_shift_advantage_config_requires_preupdate_policy_forward(mode):
    """Test that the advantage mode cannot use stale rollout probabilities."""
    adv_norm = NormConfig(mean_level=mode, std_level=None)

    with pytest.raises(ValueError, match="requires a real pre-update policy forward"):
        PPOActorConfig(adv_norm=adv_norm)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        *[
            (
                {"reward_norm": NormConfig(mean_level=mode, std_level=None)},
                "only supported by actor.adv_norm",
            )
            for mode in ("logit-shift", "logit-shift-legacy")
        ],
        (
            {
                "adv_norm": NormConfig(
                    mean_level="logit-shift",
                    mean_leave1out=True,
                    std_level=None,
                ),
                "recompute_logprob": True,
            },
            "mean_leave1out is not supported",
        ),
        (
            {
                "adv_norm": NormConfig(mean_level="logit-shift", std_level=None),
                "recompute_logprob": True,
                "importance_sampling_level": "sequence",
            },
            "requires importance_sampling_level='token'",
        ),
        (
            {
                "adv_norm": NormConfig(mean_level="logit-shift", std_level=None),
                "reward_norm": NormConfig(mean_level=None, std_level="logit-shift"),
                "recompute_logprob": True,
            },
            "cannot be enabled together",
        ),
        (
            {
                "adv_norm": NormConfig(
                    mean_level="logit-shift-legacy",
                    mean_leave1out=True,
                    std_level=None,
                ),
                "recompute_logprob": True,
            },
            "mean_leave1out is not supported",
        ),
        (
            {
                "adv_norm": NormConfig(mean_level="logit-shift-legacy", std_level=None),
                "recompute_logprob": True,
                "importance_sampling_level": "sequence",
            },
            "requires importance_sampling_level='token'",
        ),
        (
            {
                "adv_norm": NormConfig(mean_level="logit-shift-legacy", std_level=None),
                "reward_norm": NormConfig(mean_level=None, std_level="logit-shift"),
                "recompute_logprob": True,
            },
            "cannot be enabled together",
        ),
    ],
)
def test_logit_shift_advantage_rejects_unsupported_configurations(kwargs, message):
    """Test that unsupported combinations fail instead of changing semantics."""
    with pytest.raises(ValueError, match=message):
        PPOActorConfig(**kwargs)


def test_logit_shift_advantage_rejects_clip_below_one():
    """Test that a clip which would make inverse weighting a no-op is rejected."""
    with pytest.raises(ValueError, match="finite and at least 1"):
        PPOActorConfig(ls_clip=0.5)


@pytest.mark.parametrize("mode", ["logit-shift", "logit-shift-legacy"])
def test_logit_shift_mean_level_is_actor_specific(mode):
    """Test that generic normalization rejects the actor-only sentinel."""
    config = NormConfig(mean_level=mode, std_level=None)

    with pytest.raises(ValueError, match="must be handled by PPOActor"):
        Normalization(config)
