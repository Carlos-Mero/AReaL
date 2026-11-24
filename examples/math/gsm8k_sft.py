import sys
from dataclasses import dataclass
from pathlib import Path

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.dataset.gsm8k_teacher import TeacherGenerationConfig, prepare_teacher_traces
from areal.utils.hf_utils import load_hf_tokenizer


@dataclass
class GSM8KSFTConfig(SFTConfig):
    """Optionally prepare a fixed set of teacher demonstrations before student SFT."""

    teacher_generation: TeacherGenerationConfig | None = None
    prepare_only: bool = False


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GSM8KSFTConfig)
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
        prepare_teacher_traces(
            config.teacher_generation, Path(config.train_dataset.path)
        )
    if config.prepare_only:
        if config.teacher_generation is None:
            raise ValueError("prepare_only requires teacher_generation")
        return

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split=config.train_dataset.split,
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
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
