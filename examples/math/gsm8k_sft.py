import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset, is_dapo_math_17k_path
from areal.dataset.gsm8k_teacher import TeacherGenerationConfig, prepare_teacher_traces
from areal.dataset.sft_budget import SFTDataBudget
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer


@dataclass
class GSM8KSFTConfig(SFTConfig):
    """Optionally prepare a fixed set of teacher demonstrations before student SFT."""

    teacher_generation: TeacherGenerationConfig | None = None
    prepare_only: bool = False


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GSM8KSFTConfig)
    budget = SFTDataBudget.from_config(config)
    if budget is not None and budget.steps == 0:
        logging.getLogger("SFTTrainer").info(
            "total_train_steps=0: no training data or teacher traces are required"
        )
        return
    if config.teacher_generation is not None:
        if config.train_dataset.type != "teacher_sft" or not config.train_dataset.path:
            raise ValueError(
                "Teacher generation requires train_dataset.type=teacher_sft and a dataset directory"
            )
        if (
            config.teacher_generation.validation_size == 0
            and config.valid_dataset is not None
            and config.valid_dataset.path == config.train_dataset.path
            and config.valid_dataset.split == "validation"
        ):
            raise ValueError(
                "Set valid_dataset=null or supply a validation dataset when validation_size=0"
            )
        in_selection_order = prepare_teacher_traces(
            config.teacher_generation,
            Path(config.train_dataset.path),
            budget,
            config.tokenizer_path,
            config.train_dataset.max_length,
        )
        if in_selection_order and budget is not None:
            # Use the same eligible prefix when reusing a larger teacher cache.
            # The student dataloader still uses train_dataset.shuffle.
            budget = replace(budget, shuffle=False)
    if config.prepare_only:
        if config.teacher_generation is None:
            raise ValueError("prepare_only requires teacher_generation")
        return

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    loader_kwargs = {}
    train_path = config.train_dataset.path or ""
    if budget is not None and (
        config.train_dataset.type == "teacher_sft"
        or (
            config.train_dataset.type == "sft"
            and ("gsm8k" in train_path or is_dapo_math_17k_path(train_path))
        )
    ):
        loader_kwargs["training_budget"] = asdict(budget)
    train_dataset = get_custom_dataset(
        split=config.train_dataset.split,
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
        **loader_kwargs,
    )
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            split=config.valid_dataset.split,
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )

    with SFTTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
