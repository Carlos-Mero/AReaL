# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import NormConfig, PPOActorConfig, RejectionSamplingConfig
from areal.trainer.ppo.actor import grpo_loss_fn
from areal.utils.functional import prob_sq_loss_fn


def test_prob_sq_matches_detached_importance_ratio_surrogate():
    """Test the loss is -advantage times detached ratio times probability."""
    logprobs = torch.tensor([-0.2, -1.4, -0.7], dtype=torch.float64)
    proximal_logprobs = torch.tensor([-0.5, -1.0, -1.2], dtype=torch.float64)
    advantages = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float64)

    loss, stat = prob_sq_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal_logprobs,
        advantages=advantages,
        loss_mask=torch.ones(3, dtype=torch.bool),
    )
    probabilities = logprobs.exp()
    importance_ratio = (logprobs.float() - proximal_logprobs.float()).exp()
    expected_per_token = -advantages * importance_ratio * probabilities

    torch.testing.assert_close(loss, expected_per_token.mean(), rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(stat["loss"], expected_per_token, rtol=1e-7, atol=1e-7)


def test_prob_sq_stop_gradient_blocks_ratio_gradient():
    """Test current logprob receives no gradient through the importance ratio."""
    logprobs = torch.tensor([-0.3, -1.1], dtype=torch.float64, requires_grad=True)
    proximal_logprobs = torch.tensor(
        [-0.6, -0.9], dtype=torch.float64, requires_grad=True
    )
    advantages = torch.tensor([2.0, -0.75], dtype=torch.float64, requires_grad=True)

    loss, _ = prob_sq_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal_logprobs,
        advantages=advantages,
        loss_mask=torch.ones(2, dtype=torch.bool),
    )
    loss.backward()
    ratio = (logprobs.detach().float() - proximal_logprobs.detach().float()).exp()
    probability = logprobs.detach().exp()
    expected_gradient = -advantages.detach() * ratio * probability / 2

    torch.testing.assert_close(logprobs.grad, expected_gradient, rtol=1e-7, atol=1e-7)
    assert advantages.grad is None
    assert proximal_logprobs.grad is None


def test_prob_sq_logit_gradient_follows_advantage_sign():
    """Test positive advantages raise targets and negative advantages lower them."""
    logits = torch.tensor(
        [[0.2, -0.4, 0.8], [0.1, 0.7, -0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([2, 1])
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
    logprobs = (
        torch.log_softmax(logits, dim=-1)
        .gather(dim=-1, index=targets.unsqueeze(-1))
        .squeeze(-1)
    )

    loss, _ = prob_sq_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=torch.zeros_like(logprobs),
        advantages=advantages,
        loss_mask=torch.ones(2, dtype=torch.bool),
    )
    loss.backward()

    assert logits.grad[0, targets[0]] < 0
    assert logits.grad[1, targets[1]] > 0


def test_prob_sq_respects_mask_and_original_denominator():
    """Test masked tokens are zero while normalization can use the original count."""
    logprobs = torch.tensor([-0.2, -0.8], requires_grad=True)
    advantages = torch.tensor([1.0, 3.0])

    loss, stat = prob_sq_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=torch.zeros_like(logprobs),
        advantages=advantages,
        loss_mask=torch.tensor([True, False]),
        loss_mask_count=torch.tensor(2),
    )
    expected = -logprobs[0].exp().square() / 2

    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(stat["loss"][1], torch.tensor(0.0), rtol=0.0, atol=0.0)


def test_prob_sq_all_masked_returns_differentiable_zero():
    """Test an empty mask returns zero without disconnecting autograd."""
    logprobs = torch.tensor([-0.5, -1.5], requires_grad=True)

    loss, _ = prob_sq_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=torch.zeros_like(logprobs),
        advantages=torch.tensor([2.0, -3.0]),
        loss_mask=torch.zeros(2, dtype=torch.bool),
    )
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(0.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        logprobs.grad, torch.zeros_like(logprobs), rtol=0.0, atol=0.0
    )


def test_prob_sq_sequence_level_uses_geometric_mean_ratio():
    """Test sequence mode broadcasts the GSPO geometric-mean ratio."""
    logprobs = torch.tensor([[-0.2, -0.8], [-0.4, -1.0]], requires_grad=True)
    log_ratio = torch.tensor([[0.2, 0.6], [-0.4, 0.2]])
    proximal_logprobs = logprobs.detach() - log_ratio
    advantages = torch.tensor([[1.0, 3.0], [-2.0, 0.0]])
    loss_mask = torch.ones_like(logprobs, dtype=torch.bool)

    loss, stat = prob_sq_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        importance_sampling_level="sequence",
    )
    sequence_ratio = log_ratio.mean(dim=-1, keepdim=True).exp().expand_as(logprobs)
    sequence_advantage = advantages.mean(dim=-1, keepdim=True).expand_as(advantages)
    expected = (-sequence_advantage * sequence_ratio * logprobs.exp()).mean()

    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        stat["importance_weight"], sequence_ratio, rtol=1e-6, atol=1e-6
    )


def test_prob_sq_decoupled_combines_proximal_and_behavior_ratios():
    """Test rejection correction composes current/proximal and proximal/behavior."""
    logprobs = torch.tensor([-0.3, -0.9], requires_grad=True)
    proximal_logprobs = torch.tensor([-0.5, -0.7])
    old_logprobs = torch.tensor([-0.8, -1.1])
    advantages = torch.tensor([1.0, -2.0])
    loss_mask = torch.ones(2, dtype=torch.bool)
    rejection_sampling = RejectionSamplingConfig(
        level="token", action="clamp", metric="ratio", upper=100.0
    )

    loss, stat = prob_sq_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        old_logprobs=old_logprobs,
        rejection_sampling=rejection_sampling,
    )
    current_to_proximal = (logprobs.detach() - proximal_logprobs).exp()
    proximal_to_behavior = (proximal_logprobs - old_logprobs).exp()
    expected = (
        -advantages * current_to_proximal * proximal_to_behavior * logprobs.exp()
    ).mean()

    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        stat["prob_sq_weight"],
        current_to_proximal * proximal_to_behavior,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "adv_norm",
    [
        NormConfig(mean_level="group", std_level="group", group_size=2),
        NormConfig(mean_level="maxrl", std_level=None, group_size=2),
    ],
)
def test_prob_sq_config_accepts_grpo_and_maxrl_normalization(adv_norm: NormConfig):
    """Test prob_sq composes with standard GRPO and MaxRL normalization configs."""
    config = PPOActorConfig(loss_type="prob_sq", adv_norm=adv_norm)

    assert config.loss_type == "prob_sq"
    assert config.adv_norm is adv_norm


@pytest.mark.parametrize("use_sapo_loss,use_cispo_loss", [(True, False), (False, True)])
def test_prob_sq_config_rejects_conflicting_actor_losses(
    use_sapo_loss: bool, use_cispo_loss: bool
):
    """Test prob_sq cannot be combined with SAPO or CISPO."""
    kwargs = {"eps_clip_higher": 4.0} if use_cispo_loss else {}
    with pytest.raises(ValueError, match="mutually exclusive"):
        PPOActorConfig(
            loss_type="prob_sq",
            use_sapo_loss=use_sapo_loss,
            use_cispo_loss=use_cispo_loss,
            **kwargs,
        )


def test_grpo_loss_dispatches_to_prob_sq():
    """Test the actor wrapper dispatches final advantages to prob_sq."""
    logprobs = torch.tensor([[-0.4, -1.2]], requires_grad=True)
    proximal_logprobs = torch.tensor([[-0.1, -0.7]])
    advantages = torch.tensor([[2.0, -0.5]])
    input_data = {
        "input_ids": torch.tensor([[1, 2]]),
        "logprobs": torch.zeros_like(logprobs),
        "prox_logp": proximal_logprobs,
        "advantages": advantages,
        "loss_mask": torch.ones_like(logprobs, dtype=torch.bool),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker") as tracker:
        tracker.denominator = MagicMock()
        tracker.stat = MagicMock()
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            loss_type="prob_sq",
        )

    probabilities = logprobs.exp()
    importance_ratio = (logprobs.detach() - proximal_logprobs).exp()
    expected = (-advantages * importance_ratio * probabilities).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)
