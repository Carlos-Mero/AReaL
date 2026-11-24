# SPDX-License-Identifier: Apache-2.0

"""Prepare reusable math teacher completions before allocating the SFT actor."""

import fcntl
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from areal.utils import logging

logger = logging.getLogger("Dataset")

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


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

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("teacher_generation.model must be a teacher checkpoint")
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


def _decode_completion(
    tokens: list[int], stop_ids: set[int], tokenizer: "PreTrainedTokenizerBase"
) -> tuple[str, str]:
    # All tokens following the first generated EOS are batch padding.
    end = next((i for i, token in enumerate(tokens) if token in stop_ids), len(tokens))
    text = tokenizer.decode(
        tokens[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return text, "stop" if end < len(tokens) else "length"


def _write_teacher_traces(config: TeacherGenerationConfig, output: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig, set_seed

    from areal.dataset.dapo_math_17k import get_dapo_math_17k_rl_dataset
    from areal.dataset.gsm8k import get_gsm8k_rl_dataset
    from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer

    tokenizer = load_hf_tokenizer(config.model, padding_side="left")
    load_prompts = (
        get_dapo_math_17k_rl_dataset
        if config.uses_dapo_prompts
        else get_gsm8k_rl_dataset
    )
    prompts = load_prompts(
        config.dataset_path, "train", tokenizer, max_length=config.max_prompt_length
    )
    splits = {"train": prompts}
    if config.validation_size:
        holdout = prompts.train_test_split(
            test_size=config.validation_size, seed=config.seed
        )
        splits = {"train": holdout["train"], "validation": holdout["test"]}
    set_seed(config.seed)
    model = AutoModelForCausalLM.from_pretrained(
        config.model, dtype="auto", device_map="auto"
    )
    model.eval()
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if eos_ids is None or tokenizer.pad_token_id is None:
        raise ValueError("Teacher generation requires EOS and padding token IDs")
    stop_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    sampling = GenerationConfig(
        num_return_sequences=1,
        num_beams=1,
        return_dict_in_generate=False,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.temperature > 0,
        temperature=config.temperature if config.temperature > 0 else 1.0,
        top_k=0 if config.temperature > 0 else None,
        top_p=config.top_p if config.temperature > 0 else 1.0,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.inference_mode():
        for name, split in splits.items():
            logger.info("Generating %d teacher responses for %s", len(split), name)
            with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for batch in split.iter(batch_size=config.batch_size):
                    messages = batch["messages"]
                    input_ids = [
                        apply_chat_template(
                            tokenizer,
                            prompt,
                            add_generation_prompt=True,
                            enable_thinking=config.enable_thinking,
                        )
                        for prompt in messages
                    ]
                    inputs = tokenizer.pad(
                        {"input_ids": input_ids},
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    ).to(model.device)
                    sequences = model.generate(**inputs, generation_config=sampling)
                    completions = sequences[:, inputs["input_ids"].shape[1] :].tolist()
                    for prompt, answer, tokens in zip(
                        messages, batch["answer"], completions, strict=True
                    ):
                        response, finish_reason = _decode_completion(
                            tokens, stop_ids, tokenizer
                        )
                        row = {
                            "messages": prompt,
                            "answer": answer,
                            "response": response,
                            "finish_reason": finish_reason,
                            "enable_thinking": config.enable_thinking,
                        }
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_teacher_traces(config: TeacherGenerationConfig, output: Path) -> None:
    """Publish both splits together. Concurrent runs on shared storage reuse one copy."""
    settings = asdict(config)
    if config.uses_dapo_prompts:
        settings["dataset_format"] = "dapo"
    else:
        # Older manifests implicitly used the GSM8K loader.
        settings.pop("dataset_format")
    manifest = {"format_version": 1, "teacher_generation": settings}
    splits = ("train", "validation") if config.validation_size else ("train",)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            manifest_path = output / "manifest.json"
            if (
                not manifest_path.is_file()
                or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest
                or not all((output / f"{split}.jsonl").is_file() for split in splits)
            ):
                raise ValueError(
                    f"Incomplete or different teacher dataset at {output}. "
                    "Choose a new train_dataset.path or use teacher_generation=null for your own traces."
                )
            logger.info("Reusing teacher traces at %s", output)
            return

        with TemporaryDirectory(
            dir=output.parent, prefix=f".{output.name}."
        ) as staging:
            # A child process releases the teacher weights and accelerator context
            # before SFTTrainer starts, including when generation raises.
            with ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                pool.submit(_write_teacher_traces, config, Path(staging)).result()
            (Path(staging) / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            Path(staging).rename(output)
        logger.info("Published teacher traces at %s", output)
