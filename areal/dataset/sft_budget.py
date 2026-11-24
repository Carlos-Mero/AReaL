# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from datasets import Dataset, concatenate_datasets

from areal.api.alloc_mode import ModelAllocation
from areal.utils import logging
from areal.utils.environ import is_single_controller

if TYPE_CHECKING:
    from areal.api.cli_args import SFTConfig

logger = logging.getLogger("Dataset")


@dataclass(frozen=True)
class SFTDataBudget:
    """Global SFT batch and sampler settings available before tokenization."""

    steps: int
    epochs: int
    batch_size: int
    drop_last: bool
    sampler_multiple: int
    shuffle: bool
    seed: int

    @classmethod
    def from_config(cls, config: "SFTConfig") -> "SFTDataBudget | None":
        if config.total_train_steps is None:
            return None
        dataset = config.train_dataset
        if config.total_train_steps < 0 or dataset.batch_size <= 0:
            raise ValueError(
                "SFT requires non-negative total_train_steps and positive batch_size"
            )
        if is_single_controller():
            # The controller batches globally. Remote workers can discard the
            # final incomplete sampler group before reporting their total size.
            multiple = (
                max(1, dataset.num_dataset_workers)
                if dataset.scheduling_spec is not None and dataset.drop_last
                else 1
            )
        else:
            multiple = ModelAllocation.from_str(config.actor.backend).parallel.dp_size
            if dataset.batch_size % multiple:
                raise ValueError(
                    f"Global train batch size {dataset.batch_size} must be divisible by actor DP size {multiple}"
                )
        return cls(
            config.total_train_steps,
            config.total_train_epochs,
            dataset.batch_size,
            dataset.drop_last,
            multiple,
            dataset.shuffle,
            config.seed,
        )

    @property
    def target_rows(self) -> int:
        target = self.steps * self.batch_size
        return (
            (target + self.sampler_multiple - 1) // self.sampler_multiple
        ) * self.sampler_multiple

    def limit_for(self, raw_train_rows: int) -> int | None:
        """Plan against the upper bound before filtering. Rejected rows are backfilled."""
        sampled = raw_train_rows - raw_train_rows % self.sampler_multiple
        epoch_steps = (
            sampled + (0 if self.drop_last else self.batch_size - 1)
        ) // self.batch_size
        limit = self.target_rows if self.steps < epoch_steps else None
        logger.info(
            "SFT plan before filtering: %d training candidates, at most %d steps/epoch, "
            "%d planned steps, target %d training rows",
            raw_train_rows,
            epoch_steps,
            min(self.steps, epoch_steps * self.epochs),
            raw_train_rows if limit is None else limit,
        )
        return limit


def prepare_sft_dataset(
    dataset: Dataset,
    tokenize: Callable,
    max_length: int | None,
    training_budget: dict | None,
    remove_columns: list[str] | None = None,
) -> Dataset:
    """Tokenize only enough candidate rows to fill a sub-epoch training run."""

    def process(rows: Dataset) -> Dataset:
        rows = rows.map(
            tokenize,
            remove_columns=rows.column_names
            if remove_columns is None
            else remove_columns,
        )
        if max_length is not None:
            rows = rows.filter(lambda row: len(row["input_ids"]) <= max_length)
        return rows

    budget = SFTDataBudget(**training_budget) if training_budget is not None else None
    limit = budget.limit_for(len(dataset)) if budget else None
    if limit is None:
        return process(dataset)
    if budget.shuffle:
        dataset = dataset.shuffle(seed=budget.seed)
    parts = []
    start = accepted = 0
    while start < len(dataset) and accepted < limit:
        end = min(start + limit - accepted, len(dataset))
        rows = process(dataset.select(range(start, end)))
        if len(rows):
            parts.append(rows)
            accepted += len(rows)
        start = end
    return (
        concatenate_datasets(parts)
        if parts
        else Dataset.from_dict({"input_ids": [], "loss_mask": []})
    )
