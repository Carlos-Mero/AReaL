# SPDX-License-Identifier: Apache-2.0

"""Single-turn SFT on offline teacher completions."""

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from areal.utils.hf_utils import apply_chat_template


def get_teacher_sft_dataset(
    path: str,
    split: str | None,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
) -> Dataset:
    """Read a split JSONL with prompt messages, raw completion, and finish reason."""
    split = split or "train"
    if tokenizer is None:
        raise ValueError("Teacher SFT requires the student tokenizer")
    source = Path(path)
    if source.is_dir():
        source = source / f"{split}.jsonl"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing teacher SFT data for split {split!r}: {source}"
        )
    dataset = load_dataset("json", data_files={split: str(source)}, split=split)

    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        messages = row["messages"]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("Teacher SFT prompt messages must end with the user turn")
        template_kwargs = {}
        if row.get("enable_thinking") is not None:
            template_kwargs["enable_thinking"] = row["enable_thinking"]
        prompt = apply_chat_template(
            tokenizer, messages, add_generation_prompt=True, **template_kwargs
        )
        response = tokenizer.encode(row["response"], add_special_tokens=False)
        if row["finish_reason"] == "stop":
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Stopped teacher completions require a student EOS token"
                )
            response.append(tokenizer.eos_token_id)
        elif row["finish_reason"] != "length":
            raise ValueError(f"Unknown teacher finish reason: {row['finish_reason']!r}")
        if not prompt or not response:
            raise ValueError(
                "Teacher SFT requires a prompt and at least one target token"
            )
        return {
            "input_ids": prompt + response,
            "loss_mask": [0] * len(prompt) + [1] * len(response),
        }

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    if max_length is not None:
        dataset = dataset.filter(lambda row: len(row["input_ids"]) <= max_length)
    if not len(dataset):
        raise ValueError(
            f"No teacher SFT samples remain in {source} (max_length={max_length})"
        )
    return dataset
