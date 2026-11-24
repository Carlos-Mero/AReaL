# SPDX-License-Identifier: Apache-2.0

"""Prepare reusable math teacher completions before allocating the SFT actor."""

import asyncio
import fcntl
import json
import math
import multiprocessing
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import TYPE_CHECKING, Any

from areal.api.alloc_mode import ModelAllocation
from areal.dataset.sft_budget import SFTDataBudget
from areal.utils import logging

if TYPE_CHECKING:
    from sglang import Engine
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger("Dataset")


@dataclass
class TeacherGenerationConfig:
    """Offline generation settings. Validation questions are held out of train."""

    model: str = ""
    dataset_path: str = "openai/gsm8k"
    seed: int = 1
    validation_size: int = 256
    batch_size: int = 8
    max_prompt_length: int = 1024
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    enable_thinking: bool = False
    dataset_format: str | None = None
    backend: str = "sglang:d1p1t1"
    mem_fraction_static: float | None = None
    max_running_requests: int | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("teacher_generation.model must be a teacher checkpoint")
        alloc = ModelAllocation.from_str(self.backend)
        if (
            alloc.backend != "sglang"
            or min(
                alloc.parallel.dp_size, alloc.parallel.pp_size, alloc.parallel.tp_size
            )
            < 1
        ):
            raise ValueError(
                "teacher_generation.backend requires an SGLang DP/PP/TP allocation"
            )
        if (
            self.mem_fraction_static is not None
            and not 0 < self.mem_fraction_static <= 1
        ):
            raise ValueError("teacher_generation.mem_fraction_static must be in (0, 1]")
        if self.max_running_requests is not None and self.max_running_requests < 1:
            raise ValueError("teacher_generation.max_running_requests must be positive")
        if min(self.batch_size, self.max_prompt_length, self.max_new_tokens) <= 0:
            raise ValueError("Teacher batch size and length limits must be positive")
        if self.validation_size < 0:
            raise ValueError("Teacher validation size must be non-negative")
        if self.dataset_format not in (None, "gsm8k", "dapo"):
            raise ValueError(
                "teacher_generation.dataset_format must be gsm8k, dapo, or null"
            )
        if (
            not math.isfinite(self.temperature)
            or self.temperature < 0
            or not 0 < self.top_p <= 1
        ):
            raise ValueError(
                "Teacher sampling requires temperature >= 0 and 0 < top_p <= 1"
            )

    @property
    def uses_dapo_prompts(self) -> bool:
        if self.dataset_format is not None:
            return self.dataset_format == "dapo"
        return "dapo" in self.dataset_path.lower()


def _create_teacher_engine(config: TeacherGenerationConfig) -> "Engine":
    from sglang import Engine

    parallel = ModelAllocation.from_str(config.backend).parallel
    options = {
        key: value
        for key, value in {
            "mem_fraction_static": config.mem_fraction_static,
            "max_running_requests": config.max_running_requests,
        }.items()
        if value is not None
    }
    return Engine(
        model_path=config.model,
        dp_size=parallel.dp_size,
        pp_size=parallel.pp_size,
        tp_size=parallel.tp_size,
        random_seed=config.seed,
        **options,
    )


async def _generate_split(
    engine: "Engine",
    config: TeacherGenerationConfig,
    candidates: Iterator[tuple[dict, list[int]]],
    path: Path,
    limit: int | None,
    student_tokenizer: "PreTrainedTokenizerBase | None" = None,
    max_length: int | None = None,
) -> int:
    from areal.dataset.teacher_sft import tokenize_teacher_row

    sampling = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": -1,
        "skip_special_tokens": False,
        "no_stop_trim": False,
        "spaces_between_special_tokens": False,
    }

    async def generate(prompt: dict, ids: list[int]) -> dict:
        result = await engine.async_generate(
            input_ids=ids, sampling_params=sampling, stream=False
        )
        reason = result["meta_info"]["finish_reason"]
        if reason["type"] not in ("length", "stop"):
            raise RuntimeError(f"SGLang teacher request failed: {reason}")
        return {
            **prompt,
            "response": result["text"],
            "finish_reason": reason["type"],
            "enable_thinking": config.enable_thinking,
        }

    # Spool results as they finish. Keeping only offsets in RAM lets short
    # requests refill every replica even when an early answer is very long.
    pending: dict[asyncio.Task, int] = {}
    offsets: dict[int, int] = {}
    submitted = 0
    exhausted = False
    with TemporaryFile(dir=path.parent) as spool:
        try:
            while True:
                while (
                    not exhausted
                    and len(pending) < config.batch_size
                    and (limit is None or len(offsets) + len(pending) < limit)
                ):
                    candidate = next(candidates, None)
                    if candidate is None:
                        exhausted = True
                        break
                    task = asyncio.create_task(generate(*candidate))
                    pending[task] = submitted
                    submitted += 1
                if not pending:
                    break
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    index = pending.pop(task)
                    row = task.result()
                    if student_tokenizer is not None and max_length is not None:
                        ids = tokenize_teacher_row(row, student_tokenizer)["input_ids"]
                        if len(ids) > max_length:
                            continue
                    offsets[index] = spool.tell()
                    spool.write(
                        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                    )
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        with path.open("wb") as stream:
            for index in sorted(offsets):
                spool.seek(offsets[index])
                stream.write(spool.readline())
    logger.info(
        "Teacher %s: wrote %d rows from %d requests", path.stem, len(offsets), submitted
    )
    return len(offsets)


def _subset_settings(
    budget: SFTDataBudget, tokenizer_path: str | None, max_length: int | None
) -> dict:
    return {
        "target_rows": budget.target_rows,
        "shuffle": budget.shuffle,
        "seed": budget.seed if budget.shuffle else None,
        "student_tokenizer": tokenizer_path if max_length is not None else None,
        "max_length": max_length,
    }


def _write_teacher_traces(
    config: TeacherGenerationConfig,
    output: Path,
    budget: SFTDataBudget | None = None,
    tokenizer_path: str | None = None,
    max_length: int | None = None,
) -> dict | None:
    from areal.dataset.dapo_math_17k import (
        format_dapo_math_prompt,
        load_dapo_math_17k_source,
    )
    from areal.dataset.gsm8k import gsm8k_messages, load_gsm8k_source
    from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer

    source = (
        load_dapo_math_17k_source(config.dataset_path, "train")
        if config.uses_dapo_prompts
        else load_gsm8k_source(config.dataset_path, "train")
    )
    if len(source) <= config.validation_size:
        raise ValueError(
            f"{len(source)} source rows cannot supply a training split with validation_size={config.validation_size}"
        )
    limit = budget.limit_for(len(source) - config.validation_size) if budget else None
    if config.validation_size:
        # Split before applying the train budget, keeping the same eligible holdout
        # for full and short runs. Prompt rendering stays lazy across both splits.
        source = source.shuffle(seed=config.seed)
    elif limit is not None and budget.shuffle:
        source = source.shuffle(seed=budget.seed)
    tokenizer = load_hf_tokenizer(config.model)
    student = None
    if limit is not None and max_length is not None:
        if not tokenizer_path:
            raise ValueError(
                "Budgeted teacher generation requires the student tokenizer for length filtering"
            )
        student = load_hf_tokenizer(tokenizer_path)

    def eligible_prompts() -> Iterator[tuple[dict, list[int]]]:
        for example in source:
            prompt = (
                format_dapo_math_prompt(example)
                if config.uses_dapo_prompts
                else {"messages": gsm8k_messages(example), "answer": example["answer"]}
            )
            ids = apply_chat_template(
                tokenizer,
                prompt["messages"],
                add_generation_prompt=True,
                enable_thinking=config.enable_thinking,
            )
            if len(ids) <= config.max_prompt_length:
                yield prompt, ids

    async def generate() -> None:
        candidates = eligible_prompts()
        logger.info(
            "Starting offline teacher %s using %s with up to %d requests in flight",
            config.model,
            config.backend,
            config.batch_size,
        )
        engine = _create_teacher_engine(config)
        try:
            if config.validation_size:
                valid = await _generate_split(
                    engine,
                    config,
                    islice(candidates, config.validation_size),
                    output / "validation.jsonl",
                    config.validation_size,
                )
                if valid < config.validation_size:
                    raise ValueError(
                        f"Only {valid} prompts passed the teacher prompt limit for validation_size={config.validation_size}"
                    )
            count = await _generate_split(
                engine,
                config,
                candidates,
                output / "train.jsonl",
                limit,
                student,
                max_length,
            )
            if count == 0:
                raise ValueError(
                    "No eligible teacher training rows remain after holdout and length filtering"
                )
            if limit is not None and count < limit:
                logger.warning(
                    "Training source exhausted after %d eligible rows (requested %d)",
                    count,
                    limit,
                )
        finally:
            engine.shutdown()

    asyncio.run(generate())
    return (
        _subset_settings(budget, tokenizer_path, max_length)
        if limit is not None
        else None
    )


def _covers_budget(saved: dict | None, requested: dict | None) -> bool:
    if saved is None:
        return True  # Full traces are filtered and cropped by the student loader.
    if requested is None:
        return False
    return saved["target_rows"] >= requested["target_rows"] and all(
        saved.get(key) == value
        for key, value in requested.items()
        if key != "target_rows"
    )


def prepare_teacher_traces(
    config: TeacherGenerationConfig,
    output: Path,
    budget: SFTDataBudget | None = None,
    tokenizer_path: str | None = None,
    max_length: int | None = None,
) -> bool:
    """Publish or reuse traces. Return whether their train rows are in selection order."""
    settings = asdict(config)
    settings["dataset_format"] = "dapo" if config.uses_dapo_prompts else "gsm8k"
    manifest: dict[str, Any] = {"format_version": 2, "teacher_generation": settings}
    requested = _subset_settings(budget, tokenizer_path, max_length) if budget else None
    splits = ("train", "validation") if config.validation_size else ("train",)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            manifest_path = output / "manifest.json"
            saved = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else {}
            )
            if (
                not all(saved.get(k) == value for k, value in manifest.items())
                or "training_subset" not in saved
                or not _covers_budget(saved["training_subset"], requested)
                or not all((output / f"{split}.jsonl").is_file() for split in splits)
            ):
                raise ValueError(
                    f"Incomplete, insufficient, or different teacher dataset at {output}. "
                    "Choose a new train_dataset.path or use teacher_generation=null to train on existing traces."
                )
            logger.info("Reusing teacher traces at %s", output)
            return bool(config.validation_size) or saved["training_subset"] is not None

        with TemporaryDirectory(
            dir=output.parent, prefix=f".{output.name}."
        ) as staging:
            with ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                manifest["training_subset"] = pool.submit(
                    _write_teacher_traces,
                    config,
                    Path(staging),
                    budget,
                    tokenizer_path,
                    max_length,
                ).result()
            (Path(staging) / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            Path(staging).rename(output)
        logger.info("Published teacher traces at %s", output)
        return bool(config.validation_size) or manifest["training_subset"] is not None
