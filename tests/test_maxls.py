# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from areal.api.cli_args import NormConfig, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import Normalization


def test_maxls_applies_logit_shift_after_maxrl_group_advantages():
    """Test MaxRL credit followed by token scaling and sequence centering."""
    config = PPOActorConfig(
        adv_norm=NormConfig(mean_level="maxls", std_level=None, group_size=2),
        recompute_logprob=True,
        kl_ctl=0.0,
        ls_clip=4.0,
    )
    actor = PPOActor(config, object())
    batch = {
        "input_ids": torch.zeros((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
        "logprobs": torch.zeros((2, 3)),
        "prox_logp": torch.tensor(
            [
                [math.log(0.5), math.log(0.25), 0.0],
                [0.0, math.log(0.5), 0.0],
            ]
        ),
        "rewards": torch.tensor([2.0, 0.0]),
    }

    result = actor._compute_advantages(batch, adv_group_sizes=[2])

    torch.testing.assert_close(
        result["advantages"],
        torch.tensor([[0.0, 1.0, 0.0], [0.0, -0.5, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["logit_shift_advantage_weight"],
        torch.tensor([[2.0, 4.0, 0.0], [1.0, 2.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["returns"],
        torch.tensor([[2.0, 2.0, 0.0], [0.0, 0.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_maxls_actor_uses_maxrl_normalizer_internally():
    """Test that the public MaxLS sentinel maps to the MaxRL group stage."""
    config = PPOActorConfig(
        adv_norm=NormConfig(mean_level="maxls", std_level=None, group_size=4),
        recompute_logprob=True,
    )

    actor = PPOActor(config, object())

    assert actor.use_maxls
    assert actor.use_logit_shift_advantage
    assert actor.adv_norm.mean_level == "maxrl"
    assert actor.adv_norm.group_size == 4


def test_maxls_with_unit_clip_keeps_only_terminal_maxrl_credit():
    """Test that constant unit weights center non-final MaxRL credit to zero."""
    config = PPOActorConfig(
        adv_norm=NormConfig(mean_level="maxls", std_level=None, group_size=2),
        recompute_logprob=True,
        kl_ctl=0.0,
        ls_clip=1.0,
    )
    actor = PPOActor(config, object())
    batch = {
        "input_ids": torch.zeros((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
        "logprobs": torch.zeros((2, 3)),
        "prox_logp": torch.tensor(
            [[math.log(0.5), math.log(0.25), 0.0], [0.0, math.log(0.5), 0.0]]
        ),
        "rewards": torch.tensor([2.0, 0.0]),
    }

    result = actor._compute_advantages(batch, adv_group_sizes=[2])

    torch.testing.assert_close(
        result["advantages"],
        torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_maxls_centers_each_sequence_non_final_tokens_independently():
    """Test that MaxLS does not share its token mean across sequences."""
    config = PPOActorConfig(
        adv_norm=NormConfig(mean_level="maxls", std_level=None, group_size=2),
        recompute_logprob=True,
        kl_ctl=0.0,
        ls_clip=4.0,
    )
    actor = PPOActor(config, object())
    batch = {
        "input_ids": torch.zeros((2, 4), dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "loss_mask": torch.tensor(
            [[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]
        ),
        "logprobs": torch.zeros((2, 4)),
        "prox_logp": torch.tensor(
            [
                [math.log(0.5), math.log(0.25), math.log(0.25), 0.0],
                [0.0, math.log(0.5), math.log(0.25), 0.0],
            ]
        ),
        "rewards": torch.tensor([2.0, 0.0]),
    }

    result = actor._compute_advantages(batch, adv_group_sizes=[2])

    torch.testing.assert_close(
        result["advantages"],
        torch.tensor(
            [[-0.25, 0.25, 1.0, 0.0], [0.125, -0.125, -1.0, 0.0]]
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["advantages"][:, :2].sum(dim=-1),
        torch.zeros(2),
        rtol=0.0,
        atol=1e-6,
    )


def test_maxls_config_requires_preupdate_policy_forward():
    """Test that MaxLS cannot use stale rollout token probabilities."""
    with pytest.raises(ValueError, match="requires a real pre-update policy forward"):
        PPOActorConfig(
            adv_norm=NormConfig(mean_level="maxls", std_level=None, group_size=2)
        )


def test_maxls_config_requires_token_importance_sampling():
    """Test that MaxLS retains the logit-shift token-level IS constraint."""
    with pytest.raises(ValueError, match="requires importance_sampling_level='token'"):
        PPOActorConfig(
            adv_norm=NormConfig(mean_level="maxls", std_level=None, group_size=2),
            recompute_logprob=True,
            importance_sampling_level="sequence",
        )


def test_maxls_config_rejects_additional_std_shaping():
    """Test that MaxLS has one unambiguous MaxRL-then-logit-shift order."""
    with pytest.raises(ValueError, match="std_level must be None"):
        PPOActorConfig(
            adv_norm=NormConfig(
                mean_level="maxls", std_level="batch", group_size=2
            ),
            recompute_logprob=True,
        )


def test_maxls_config_rejects_logit_shift_reward_shaping():
    """Test that MaxLS does not apply inverse-probability shaping twice."""
    with pytest.raises(ValueError, match="cannot be enabled together"):
        PPOActorConfig(
            adv_norm=NormConfig(mean_level="maxls", std_level=None, group_size=2),
            reward_norm=NormConfig(mean_level=None, std_level="logit-shift"),
            recompute_logprob=True,
        )


def test_maxls_config_rejects_nonpositive_group_size():
    """Test that MaxLS requires a valid prompt group size."""
    with pytest.raises(ValueError, match="group_size must be a positive integer"):
        NormConfig(mean_level="maxls", std_level=None, group_size=0)


def test_maxls_mean_level_is_not_supported_for_rewards():
    """Test that MaxLS cannot be routed through reward normalization."""
    with pytest.raises(ValueError, match="only supported by actor.adv_norm"):
        PPOActorConfig(
            reward_norm=NormConfig(
                mean_level="maxls", std_level=None, group_size=2
            )
        )


def test_maxls_mean_level_is_actor_specific():
    """Test that generic normalization cannot bypass MaxLS composition order."""
    config = NormConfig(mean_level="maxls", std_level=None, group_size=2)

    with pytest.raises(ValueError, match="must be handled by PPOActor"):
        Normalization(config)
