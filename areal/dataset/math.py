# SPDX-License-Identifier: Apache-2.0

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

BOXED_INSTRUCTION = "\nPlease put your final answer within \\boxed{}."

HENDRYCKS_MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def get_hendrycks_math_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    name: str | None = None,
):
    dataset = _load_hendrycks_math_dataset(path=path, split=split, name=name)
    original_columns = dataset.column_names

    def process(sample):
        question = sample["problem"]
        answer = sample["solution"]
        seq_token = tokenizer.encode(question + answer + tokenizer.eos_token)
        prompt_token = tokenizer.encode(question)
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process, remove_columns=original_columns)

    if max_length is not None:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_hendrycks_math_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    name: str | None = None,
):
    dataset = _load_hendrycks_math_dataset(path=path, split=split, name=name)

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["problem"] + BOXED_INSTRUCTION,
            }
        ]
        return {"messages": messages, "answer": sample["solution"]}

    dataset = dataset.map(process).remove_columns(["problem", "solution"])
    return _filter_text_dataset_by_length(dataset, tokenizer, max_length)


def get_dapo_math_17k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path=path, split=split)
    original_columns = dataset.column_names

    def process(sample):
        question = _prompt_to_text(sample["prompt"])
        answer = _boxed_answer(_extract_ground_truth(sample))
        seq_token = tokenizer.encode(question + answer + tokenizer.eos_token)
        prompt_token = tokenizer.encode(question)
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process, remove_columns=original_columns)

    if max_length is not None:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_dapo_math_17k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        messages = _normalize_prompt_messages(sample["prompt"])
        return {
            "messages": messages,
            "answer": _boxed_answer(_extract_ground_truth(sample)),
        }

    dataset = dataset.map(process).remove_columns(["prompt", "reward_model"])
    return _filter_text_dataset_by_length(dataset, tokenizer, max_length)


def _load_hendrycks_math_dataset(path: str, split: str, name: str | None = None):
    if name is not None:
        return load_dataset(path=path, name=name, split=split)

    try:
        return load_dataset(path=path, split=split)
    except ValueError:
        pass

    config_names = get_dataset_config_names(path) or HENDRYCKS_MATH_CONFIGS
    datasets = [
        load_dataset(path=path, name=config_name, split=split)
        for config_name in config_names
    ]
    return concatenate_datasets(datasets)


def _filter_text_dataset_by_length(dataset, tokenizer, max_length: int | None):
    if max_length is None:
        return dataset

    def filter_length(sample):
        content = _prompt_to_text(sample["messages"])
        tokens = tokenizer.encode(content)
        return len(tokens) <= max_length

    return dataset.filter(filter_length)


def _normalize_prompt_messages(prompt) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        return [
            {"role": message["role"], "content": message["content"]}
            for message in prompt
        ]
    return [{"role": "user", "content": str(prompt)}]


def _prompt_to_text(prompt) -> str:
    if isinstance(prompt, list):
        return "\n".join(str(message.get("content", "")) for message in prompt)
    return str(prompt)


def _extract_ground_truth(sample) -> str:
    reward_model = sample["reward_model"]
    if isinstance(reward_model, dict):
        return str(reward_model["ground_truth"])
    return str(reward_model)


def _boxed_answer(answer: str) -> str:
    if "\\boxed" in answer:
        return answer
    return f"\\boxed{{{answer}}}"
