# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from datasets import (
    Dataset,
    DatasetDict,
    ReadInstruction,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from areal.dataset.sft_budget import prepare_sft_dataset
from areal.utils.hf_utils import apply_chat_template


def load_dapo_math_17k_source(path: str, split: str | None) -> Dataset:
    """Load the original or unique DAPO-Math split from the Hub or local files."""
    dataset_root = Path(path)
    split = split or "train"
    if (dataset_root / "dataset_dict.json").is_file() or (
        (dataset_root / "state.json").is_file()
        and (dataset_root / "dataset_info.json").is_file()
    ):
        stored = load_from_disk(path)
        splits = (
            stored
            if isinstance(stored, DatasetDict)
            else {str(stored.split or "train"): stored}
        )
        if split in splits:
            return splits[split]
        # Apply the same train[:N] / train[N:] syntax as the Hub loader.
        lengths = {name: len(data) for name, data in splits.items()}
        parts = []
        for part in ReadInstruction.from_spec(split).to_absolute(lengths):
            data = splits[part.splitname]
            indices = range(len(data))[part.from_ : part.to]
            parts.append(data.select(indices))
        return parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if dataset_root.is_file():
        loaders = {".parquet": "parquet", ".jsonl": "json", ".json": "json"}
        loader = loaders.get(dataset_root.suffix.lower())
        if loader is None:
            raise ValueError(f"Unsupported DAPO data file: {dataset_root}")
        return load_dataset(loader, data_files=str(dataset_root), split=split)
    if dataset_root.is_absolute() and not dataset_root.exists():
        raise FileNotFoundError(
            f"DAPO dataset directory does not exist: {dataset_root}"
        )
    if any((dataset_root / "data").glob("*.parquet")):
        return load_dataset("parquet", data_dir=str(dataset_root / "data"), split=split)
    return load_dataset(path, split=split)


def format_dapo_math_prompt(sample: dict) -> dict:
    if "messages" in sample and "answer" in sample:
        return {"messages": sample["messages"], "answer": str(sample["answer"])}
    answer = sample.get("label")
    if answer is None:
        answer = sample["reward_model"]["ground_truth"]
    return {"messages": sample["prompt"], "answer": str(answer)}


def get_dapo_math_17k_sft_dataset(
    path: str,
    split: str | None,
    tokenizer,
    max_length: int | None = None,
    training_budget: dict | None = None,
) -> Dataset:
    """Supervise the reference answer and EOS for raw DAPO math questions."""
    dataset = load_dapo_math_17k_source(path, split)
    if {"input_ids", "loss_mask"}.issubset(dataset.column_names):
        return prepare_sft_dataset(dataset, None, max_length, training_budget)
    if tokenizer is None or tokenizer.eos_token_id is None:
        raise ValueError(
            "DAPO reference-answer SFT requires a tokenizer with an EOS token"
        )

    def tokenize(sample: dict) -> dict[str, list[int]]:
        row = format_dapo_math_prompt(sample)
        prompt = apply_chat_template(
            tokenizer, row["messages"], add_generation_prompt=True
        )
        answer = tokenizer.encode(row["answer"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        return {
            "input_ids": prompt + answer,
            "loss_mask": [0] * len(prompt) + [1] * len(answer),
        }

    return prepare_sft_dataset(dataset, tokenize, max_length, training_budget)


def get_dapo_math_17k_rl_dataset(
    path: str,
    split: str | None,
    tokenizer,
    max_length: int | None = None,
) -> Dataset:
    dataset = load_dapo_math_17k_source(path, split)
    if not {"messages", "answer"}.issubset(dataset.column_names):
        dataset = dataset.map(
            format_dapo_math_prompt, remove_columns=dataset.column_names
        )

    if max_length is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when max_length is set")

        def filter_length(sample):
            prompt_ids = apply_chat_template(
                tokenizer, sample["messages"], add_generation_prompt=True, tokenize=True
            )
            return len(prompt_ids) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
