# SPDX-License-Identifier: Apache-2.0

from datasets import Dataset, load_dataset

from areal.dataset.sft_budget import prepare_sft_dataset


def load_gsm8k_source(path: str, split: str) -> Dataset:
    return load_dataset(path=path, name="main", split=split)


def gsm8k_messages(sample: dict) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": sample["question"]
            + "\nPlease put your final answer within \\boxed{}.",
        }
    ]


def get_gsm8k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    training_budget: dict | None = None,
):
    dataset = load_gsm8k_source(path, split)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    return prepare_sft_dataset(
        dataset,
        process,
        max_length,
        training_budget,
        remove_columns=["question", "answer"],
    )


def get_gsm8k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_gsm8k_source(path, split)

    def process(sample):
        return {"messages": gsm8k_messages(sample)}

    dataset = dataset.map(process).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
