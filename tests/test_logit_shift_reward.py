# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import NormConfig, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import KLEstimator, Normalization
from areal.utils.functional import logit_shift_reward_shaping


def _actor(*, adv_norm: Normalization | None = None, recompute_logprob: bool = True):
    actor = object.__new__(PPOActor)
    actor.config = SimpleNamespace(
        overlong_reward_penalty=False,
        use_decoupled_loss=False,
        recompute_logprob=recompute_logprob,
        mask_no_eos_with_zero=False,
        ls_clip=10.0,
    )
    actor.reward_bias = 0.0
    actor.reward_scaling = 1.0
    actor.reward_clip = 20.0
    actor.reward_norm = None
    actor.use_logit_shift_reward = True
    actor.use_logit_shift_advantage = False
    actor.kl_ctl = 0.0
    actor.kl_estimator = KLEstimator("k1")
    actor.discount = 1.0
    actor.gae_lambda = 1.0
    actor.mask_no_eos_with_zero = False
    actor.adv_norm = adv_norm
    return actor


def test_logit_shift_reward_shaping_scales_masks_and_clips():
    """Test only positive rewards are reweighted before zero-sum balancing."""
    rewards = torch.tensor([[2.0, -3.0, 1.0, -1.0, 100.0]])
    policy_logprobs = torch.tensor(
        [[math.log(0.5), math.log(0.01), math.log(0.25), math.log(0.001), -1000.0]]
    )
    loss_mask = torch.tensor([[True, True, True, True, False]])

    shaped, weights, clipped = logit_shift_reward_shaping(
        rewards=rewards,
        policy_logprobs=policy_logprobs,
        loss_mask=loss_mask,
        ls_clip=10.0,
    )

    torch.testing.assert_close(
        weights,
        torch.tensor([[2.0, 1.0, 4.0, 1.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        shaped,
        torch.tensor([[2.0, -3.0, 2.0, -1.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(shaped.sum(), torch.tensor(0.0), rtol=0.0, atol=1e-6)
    assert torch.equal(clipped, torch.tensor([[False, False, False, False, False]]))


def test_logit_shift_reward_shaping_extreme_logprob_stays_finite():
    """Test only a tiny-probability positive token reports clip saturation."""
    shaped, weights, clipped = logit_shift_reward_shaping(
        rewards=torch.tensor([[3.0, -3.0]]),
        policy_logprobs=torch.tensor([[-1000.0, -1000.0]]),
        loss_mask=torch.tensor([[True, True]]),
        ls_clip=10.0,
    )

    torch.testing.assert_close(
        shaped, torch.tensor([[3.0, -3.0]]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        weights, torch.tensor([[10.0, 1.0]]), rtol=1e-6, atol=1e-6
    )
    assert torch.equal(clipped, torch.tensor([[True, False]]))


def test_logit_shift_reward_shaping_balances_each_group_independently():
    """Test positive redistribution preserves zero reward sum in every group."""
    rewards = torch.tensor([[1.0], [1.0], [-2.0], [1.0], [1.0], [-4.0]])
    policy_logprobs = torch.tensor(
        [
            [math.log(0.5)],
            [math.log(0.25)],
            [0.0],
            [math.log(0.1)],
            [math.log(0.5)],
            [0.0],
        ]
    )

    shaped, _, _ = logit_shift_reward_shaping(
        rewards,
        policy_logprobs,
        torch.ones_like(rewards, dtype=torch.bool),
        ls_clip=10.0,
        group_sizes=[3, 3],
    )

    expected = torch.tensor(
        [[2.0 / 3], [4.0 / 3], [-2.0], [10.0 / 3], [2.0 / 3], [-4.0]]
    )
    torch.testing.assert_close(shaped, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(shaped[:3].sum(), torch.tensor(0.0), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(shaped[3:].sum(), torch.tensor(0.0), rtol=0.0, atol=1e-6)


def test_logit_shift_reward_shaping_handles_one_sided_groups():
    """Test all-positive groups become zero while all-negative groups stay fixed."""
    rewards = torch.tensor([[1.0], [2.0], [-1.0], [-2.0]])

    shaped, weights, _ = logit_shift_reward_shaping(
        rewards,
        torch.zeros_like(rewards),
        torch.ones_like(rewards, dtype=torch.bool),
        group_sizes=[2, 2],
    )

    torch.testing.assert_close(shaped, torch.tensor([[0.0], [0.0], [-1.0], [-2.0]]))
    torch.testing.assert_close(weights, torch.ones_like(rewards))


def test_logit_shift_reward_uses_aligned_policy_probability_before_gae():
    """Test group-centered rewards use aligned policy probabilities before GAE."""
    config = PPOActorConfig(
        reward_norm=NormConfig(
            mean_level="group",
            std_level="logit-shift",
            group_size=3,
        ),
        recompute_logprob=True,
        kl_ctl=0.0,
    )
    actor = PPOActor(config, object())
    batch = {
        "input_ids": torch.zeros((3, 3), dtype=torch.long),
        "attention_mask": torch.ones((3, 3), dtype=torch.long),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]).expand(3, -1),
        "logprobs": torch.zeros((3, 3)),
        "prox_logp": torch.tensor(
            [
                [0.0, math.log(0.5), 0.0],
                [0.0, math.log(0.25), 0.0],
                [0.0, math.log(0.01), 0.0],
            ]
        ),
        "rewards": torch.tensor([1.0, 1.0, 0.0]),
    }

    result = actor.compute_advantages([batch])[0]

    expected_rewards = torch.tensor(
        [[0.0, 2.0 / 9, 0.0], [0.0, 4.0 / 9, 0.0], [0.0, -2.0 / 3, 0.0]]
    )
    expected_advantages = torch.tensor(
        [
            [2.0 / 9, 2.0 / 9, 0.0],
            [4.0 / 9, 4.0 / 9, 0.0],
            [-2.0 / 3, -2.0 / 3, 0.0],
        ]
    )
    torch.testing.assert_close(
        result["tot_rewards"], expected_rewards, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        result["advantages"], expected_advantages, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        result["tot_rewards"].sum(), torch.tensor(0.0), rtol=0.0, atol=1e-6
    )


def test_logit_shift_reward_recompute_uses_policy_probability():
    """Test reward shaping uses the existing pre-update actor forward."""
    actor = _actor(recompute_logprob=True)
    batch = {
        "input_ids": torch.zeros((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        "logprobs": torch.tensor([[0.0, math.log(0.5), math.log(0.25)]]),
        "prox_logp": torch.zeros((1, 3)),
        "rewards": torch.tensor([2.0]),
    }

    result = actor._compute_advantages(batch)

    # With no negative reward mass, zero-sum balancing removes the positive reward.
    torch.testing.assert_close(
        result["tot_rewards"], torch.zeros((1, 3)), rtol=1e-6, atol=1e-6
    )


def test_logit_shift_reward_requires_recomputed_policy_logprob():
    """Test shaping never silently falls back to rollout probabilities."""
    actor = _actor()
    batch = {
        "input_ids": torch.zeros((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        "logprobs": torch.zeros((1, 3)),
        "rewards": torch.tensor([1.0]),
    }

    with pytest.raises(ValueError, match="prox_logp is required"):
        actor._compute_advantages(batch)


def test_logit_shift_reward_flows_into_maxrl_group_normalization():
    """Test MaxRL consumes sequence sums after reward shaping."""
    actor = _actor(
        adv_norm=Normalization(
            NormConfig(mean_level="maxrl", std_level=None, group_size=2)
        )
    )
    actor.reward_norm = Normalization(
        NormConfig(mean_level="group", std_level=None, group_size=2)
    )
    batch = {
        "input_ids": torch.zeros((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.long),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
        "logprobs": torch.tensor(
            [[0.0, math.log(0.5), math.log(0.5)], [0.0, 0.0, 0.0]]
        ),
        "prox_logp": torch.tensor(
            [[math.log(0.5), math.log(0.5), 0.0], [0.0, 0.0, 0.0]]
        ),
        "rewards": torch.tensor([2.0, 0.0]),
    }

    result = actor._compute_advantages(batch, adv_group_sizes=[2])

    expected = torch.tensor([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])
    torch.testing.assert_close(result["advantages"], expected, rtol=1e-6, atol=1e-6)


def test_logit_shift_reward_config_and_loss_type_migration():
    """Test reward_norm enables shaping while the old loss type is rejected."""
    reward_norm = NormConfig(mean_level=None, std_level="logit-shift")
    config = PPOActorConfig(
        reward_norm=reward_norm, recompute_logprob=True, ls_clip=5.0
    )
    actor = PPOActor(config, object())

    assert config.reward_norm is reward_norm
    assert config.ls_clip == 5.0
    assert actor.use_logit_shift_reward
    assert actor.reward_norm.std_level is None
    assert config.should_compute_prox_logp()
    decoupled_config = PPOActorConfig(
        reward_norm=reward_norm,
        use_decoupled_loss=True,
        prox_logp_method="recompute",
    )
    assert decoupled_config.should_compute_prox_logp()
    with pytest.raises(ValueError, match="reward_norm.std_level='logit-shift'"):
        PPOActorConfig(loss_type="logit_shift")
    with pytest.raises(ValueError, match="finite and at least 1"):
        PPOActorConfig(ls_clip=0.5)


def test_logit_shift_config_requires_existing_policy_forward():
    """Test logit-shift does not introduce an implicit extra actor forward."""
    reward_norm = NormConfig(mean_level=None, std_level="logit-shift")

    with pytest.raises(ValueError, match="requires a real pre-update policy forward"):
        PPOActorConfig(reward_norm=reward_norm)
    with pytest.raises(ValueError, match="requires a real pre-update policy forward"):
        PPOActorConfig(
            reward_norm=reward_norm,
            use_decoupled_loss=True,
            prox_logp_method="loglinear",
        )


def test_logit_shift_std_level_is_not_a_generic_normalization_mode():
    """Test the sentinel requires PPOActor reward or advantage context."""
    config = NormConfig(mean_level=None, std_level="logit-shift")

    with pytest.raises(ValueError, match="must be handled by PPOActor"):
        Normalization(config)
    actor = PPOActor(PPOActorConfig(adv_norm=config), object())
    assert actor.use_logit_shift_advantage
    assert actor.adv_norm.std_level is None
