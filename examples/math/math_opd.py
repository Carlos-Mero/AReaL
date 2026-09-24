# SPDX-License-Identifier: Apache-2.0

"""Pure OPD against a fresh, pre-update actor snapshot on every rollout batch."""

import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.trainer.rl_trainer import validate_self_distillation
from areal.utils.hf_utils import load_hf_tokenizer


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    validate_self_distillation(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = (
        get_custom_dataset(
            split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
        )
        if config.valid_dataset is not None
        else None
    )
    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_tokens=config.gconfig.max_tokens,
        max_completion_tokens=config.gconfig.max_new_tokens,
    )
    with PPOTrainer(config, train_dataset, valid_dataset) as trainer:
        trainer.train(
            workflow="areal.workflow.openai.math_agent.MathAgent",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.openai.math_agent.MathAgent",
            eval_workflow_kwargs={**workflow_kwargs, "temperature": 0.6},
            pre_update_actor_teacher=True,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
