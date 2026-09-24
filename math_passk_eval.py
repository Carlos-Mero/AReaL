#!/usr/bin/env python3
"""Evaluate a local or Hugging Face language model on math benchmarks with SGLang.

Only a local checkpoint path or Hugging Face model ID is required::

    python math_passk_eval.py /path/to/checkpoint
    python math_passk_eval.py Qwen/Qwen3-8B

All visible CUDA GPUs are used automatically as data-parallel replicas. For a
model that needs two GPUs per replica, for example, use ``--tp-size 2``; on an
eight-GPU server this creates four data-parallel replicas. Set ``--dp-size``
explicitly to override the automatic replica count.

By default the script uses the AReaL math training rollout protocol: a user-only
prompt, temperature=1, top_p=1, 4 samples, and full-response math-verify grading.
It evaluates MATH-500, AIME 2025, AIME 2024, and APEX 2025 and writes all
artifacts under ``./math_passk_results/<run-id>/``.
Use --datasets math_train to evaluate the training split (with the training
prompt-length filter), or math_test for the complete held-out MATH split.
Use --protocol benchmark-v2 to reproduce the previous corrected evaluator.

Use ``--datasets arxivmath`` to evaluate ArXivMath 02/26, 01/26, and 12/25 as
one combined benchmark with one set of metrics.
Use ``--datasets apexsl-2025`` to evaluate Apex Shortlist 2025.

Required packages:
    sglang, datasets, transformers, matplotlib, math-verify
For comparisons with existing runs use math-verify==0.8.0 on both sides.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, DecimalException
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. Work through the problem "
    "step by step. Put only the final answer inside \\boxed{} at the end."
)
SCORING_VERSION = "2-preserve-reference-latex"
AREAL_SCORING_VERSION = "3-areal-math-agent-prediction-first"
BOXED_INSTRUCTION = "\nPlease put your final answer within \\boxed{}."
PROTOCOLS = {
    "areal-rollout": dict(temperature=1.0, top_p=1.0, top_k=100_000_000,
                          num_samples=4, seed=1, prompt_style="areal",
                          scorer="areal", dtype="bfloat16"),
    "benchmark-v2": dict(temperature=0.8, top_p=0.95, top_k=-1,
                         num_samples=128, seed=42, prompt_style="default",
                         scorer="benchmark-v2", dtype="auto"),
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_path: str
    hf_name: str | None
    split: str
    question_field: str
    answer_field: str | None
    solution_field: str | None


DATASETS: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(
        key="gsm8k",
        hf_path="openai/gsm8k",
        hf_name="main",
        split="test",
        question_field="question",
        answer_field="answer",
        solution_field="answer",
    ),
    "math500": DatasetSpec(
        key="math500",
        hf_path="HuggingFaceH4/MATH-500",
        hf_name=None,
        split="test",
        question_field="problem",
        answer_field="answer",
        solution_field="solution",
    ),
    "aime2025": DatasetSpec(
        key="aime2025",
        hf_path="MathArena/aime_2025",
        hf_name=None,
        split="train",
        question_field="problem",
        answer_field="answer",
        solution_field=None,
    ),
    "aime2024": DatasetSpec(
        key="aime2024",
        hf_path="HuggingFaceH4/aime_2024",
        hf_name=None,
        split="train",
        question_field="problem",
        answer_field="answer",
        solution_field="solution",
    ),
    "apex2025": DatasetSpec(
        key="apex2025",
        hf_path="MathArena/apex_2025",
        hf_name=None,
        split="train",
        question_field="problem",
        answer_field="answer",
        solution_field=None,
    ),
    "apexsl-2025": DatasetSpec(
        key="apexsl-2025",
        hf_path="MathArena/apex-shortlist",
        hf_name=None,
        split="train",
        question_field="problem",
        answer_field="answer",
        solution_field=None,
    ),
    "arxivmath0226": DatasetSpec(
        key="arxivmath0226",
        hf_path="MathArena/arxivmath-0226",
        hf_name=None,
        split="train",
        question_field="problem",
        answer_field="answer",
        solution_field=None,
    ),
    "arxivmath0126": DatasetSpec(
        key="arxivmath0126",
        hf_path="MathArena/arxivmath-0126",
        hf_name=None,
        split="train",
        question_field="problem",
        answer_field="answer",
        solution_field=None,
    ),
    "arxivmath1225": DatasetSpec(
        key="arxivmath1225",
        hf_path="MathArena/arxivmath-1225",
        hf_name=None,
        split="train",
        question_field="problem",
        answer_field="answer",
        solution_field=None,
    ),
    "aime2026": DatasetSpec(
        key="aime2026",
        hf_path="MathArena/aime_2026",
        hf_name=None,
        split="train",
        question_field="problem",
        answer_field="answer",
        solution_field=None,
    ),
}

# Group aliases are evaluated and reported as one logical dataset. The source
# dataset remains encoded in every problem ID and snapshot row.
DATASET_GROUPS: dict[str, tuple[str, ...]] = {
    "arxivmath": ("arxivmath0226", "arxivmath0126", "arxivmath1225"),
}

DEFAULT_DATASETS = ["math500", "aime2025", "aime2024", "apex2025"]
for _split in ("train", "test"):
    DATASETS[f"math_{_split}"] = DatasetSpec(
        key=f"math_{_split}", hf_path="EleutherAI/hendrycks_math", hf_name=None,
        split=_split, question_field="problem", answer_field=None, solution_field="solution",
    )


def resolve_protocol(args: argparse.Namespace) -> None:
    defaults = PROTOCOLS[args.protocol]
    args.protocol_overrides = {
        key: getattr(args, key) for key, value in defaults.items()
        if getattr(args, key) is not None and getattr(args, key) != value
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def scoring_version(scorer: str) -> str:
    return AREAL_SCORING_VERSION if scorer == "areal" else SCORING_VERSION


def parse_k_values(value: str, n_samples: int) -> list[int]:
    """Parse comma-separated integers/ranges, e.g. '1,2,4,8-16'."""
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            parts = item.split("-", maxsplit=1)
            try:
                lo, hi = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid k range: {item!r}") from exc
            if lo > hi:
                raise argparse.ArgumentTypeError(f"Invalid descending k range: {item!r}")
            result.update(range(lo, hi + 1))
        else:
            try:
                result.add(int(item))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid k value: {item!r}") from exc
    if not result or min(result) < 1:
        raise argparse.ArgumentTypeError("k values must be positive integers")
    if max(result) > n_samples:
        raise argparse.ArgumentTypeError(
            f"Largest k ({max(result)}) exceeds --num-samples ({n_samples})"
        )
    return sorted(result)


def positive_int_or_auto(value: str) -> int | str:
    """Parse a positive integer or the literal ``auto``."""
    if value.lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer or 'auto'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer or 'auto'")
    return parsed


def detect_cuda_gpu_count() -> int:
    """Return the number of CUDA devices visible to this process."""
    try:
        import torch

        return int(torch.cuda.device_count())
    except (ImportError, RuntimeError):
        return 0


def resolve_parallelism(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Resolve automatic DP against visible GPUs and validate GPU allocation."""
    visible_gpus = detect_cuda_gpu_count()
    requested_dp_size = args.dp_size

    if requested_dp_size == "auto":
        # Each DP replica consumes tp_size GPUs. The default tp_size=1 therefore
        # uses every visible GPU and maximizes independent-request throughput.
        args.dp_size = max(1, visible_gpus // args.tp_size)

    required_gpus = args.tp_size * args.dp_size
    if visible_gpus and required_gpus > visible_gpus:
        parser.error(
            f"--tp-size {args.tp_size} × --dp-size {args.dp_size} requires "
            f"{required_gpus} GPUs, but only {visible_gpus} CUDA GPUs are visible"
        )

    args.requested_dp_size = requested_dp_size
    args.visible_gpu_count = visible_gpus
    args.engine_gpu_count = required_gpus

    if visible_gpus == 0:
        print(
            "[warning] No CUDA GPUs were detected before SGLang startup; "
            f"using tp_size={args.tp_size}, dp_size={args.dp_size}.",
            file=sys.stderr,
        )
    elif required_gpus < visible_gpus:
        print(
            f"[warning] Using {required_gpus} of {visible_gpus} visible GPUs. "
            "Adjust --tp-size/--dp-size if this was not intended.",
            file=sys.stderr,
        )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def extract_boxed(text: str) -> str | None:
    """Return the contents of the last balanced LaTeX \boxed{...}."""
    marker = "\\boxed{"
    starts = [match.start() for match in re.finditer(re.escape(marker), text)]
    for start in reversed(starts):
        depth = 1
        content_start = start + len(marker)
        for index in range(content_start, len(text)):
            char = text[index]
            if char == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif char == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    return text[content_start:index].strip()
    return None


FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer\s+is|therefore)[^:=\n]{0,20}\s*[:=]?\s*([^\n]+)",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"[-+]?\s*(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
)


def extract_answer(text: str, dataset_key: str | None = None) -> str:
    text = str(text or "").strip()
    if dataset_key == "gsm8k" and "####" in text:
        return text.rsplit("####", maxsplit=1)[-1].strip()
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    matches = list(FINAL_ANSWER_RE.finditer(text))
    if matches:
        candidate = matches[-1].group(1).strip()
        return candidate.rstrip(".。 ")
    numbers = NUMBER_RE.findall(text)
    return numbers[-1].replace(" ", "") if numbers else text


def strip_outer_braces(text: str) -> str:
    while text.startswith("{") and text.endswith("}"):
        depth = 0
        balanced = True
        for index, char in enumerate(text):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0 and index != len(text) - 1:
                balanced = False
                break
        if not balanced:
            break
        text = text[1:-1].strip()
    return text


def canonicalize_answer(value: str) -> str:
    value = str(value).strip()
    value = value.replace("−", "-").replace("–", "-")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\,", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("$", "").replace("\\%", "%")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = value.rstrip(".。 ")
    value = strip_outer_braces(value)
    return re.sub(r"\s+", "", value).lower()


def numeric_value(value: str) -> Decimal | None:
    """Parse finite numbers, returning None for invalid or overflowing input."""
    value = canonicalize_answer(value).replace(",", "")
    is_percent = value.endswith("%")
    if is_percent:
        value = value[:-1]
    frac_match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", value)
    plain_frac = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)", value)
    try:
        fraction = frac_match or plain_frac
        if fraction:
            numerator = Decimal(fraction.group(1))
            denominator = Decimal(fraction.group(2))
            if not numerator.is_finite() or not denominator.is_finite() or denominator == 0:
                return None
            number = numerator / denominator
        else:
            number = Decimal(value)
        if not number.is_finite():
            return None
        if is_percent:
            number /= 100
        return number if number.is_finite() else None
    except DecimalException:
        return None


_MATH_VERIFY_WARNING_EMITTED = False


def answers_equivalent(prediction: str, reference: str) -> bool:
    """Check equivalence with math-verify, then conservative fallbacks."""
    global _MATH_VERIFY_WARNING_EMITTED
    try:
        from math_verify import LatexExtractionConfig, parse, verify

        # These are extracted answers, not prose. Bare LaTeX can otherwise be
        # parsed as just one number: parse(r"3\sqrt{13}") returns 3.
        def parse_answer(value: str):
            value = value.strip()
            for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
                if value.startswith(left) and value.endswith(right):
                    value = value[len(left):-len(right)].strip()
                    break
            return parse(
                f"\\boxed{{{value}}}",
                extraction_config=[LatexExtractionConfig()],
                fallback_mode="no_fallback",
                extraction_mode="first_match",
            )

        if bool(verify(parse_answer(reference), parse_answer(prediction))):
            return True
    except ImportError:
        if not _MATH_VERIFY_WARNING_EMITTED:
            print(
                "[warning] math-verify is not installed; using numeric/string checking. "
                "Install math-verify for stronger symbolic grading.",
                file=sys.stderr,
            )
            _MATH_VERIFY_WARNING_EMITTED = True
    except Exception:
        # Malformed model output should not abort an entire evaluation.
        pass

    pred_norm = canonicalize_answer(prediction)
    ref_norm = canonicalize_answer(reference)
    if pred_norm == ref_norm:
        return True
    pred_number, ref_number = numeric_value(prediction), numeric_value(reference)
    if pred_number is None or ref_number is None:
        return False
    try:
        scale = max(abs(ref_number), Decimal(1))
        difference = abs(pred_number - ref_number)
        tolerance = Decimal("1e-6") * scale
        return difference.is_finite() and tolerance.is_finite() and difference <= tolerance
    except DecimalException:
        # Extreme model output can overflow even when both inputs are finite.
        return False


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
    if not 0 <= c <= n:
        raise ValueError(f"Expected 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"Expected 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    product = 1.0
    for item in range(n - c + 1, n + 1):
        product *= 1.0 - k / item
    return 1.0 - product


def summarize_passk(correct_counts: Sequence[int], n: int, ks: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in ks:
        values = [pass_at_k(n, count, k) for count in correct_counts]
        mean = fmean(values) if values else float("nan")
        if len(values) > 1:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            standard_error = math.sqrt(variance / len(values))
        else:
            standard_error = 0.0
        rows.append(
            {
                "k": k,
                "pass_at_k": mean,
                "standard_error": standard_error,
                "num_problems": len(values),
                "num_samples_per_problem": n,
            }
        )
    return rows


def build_prompt(
    tokenizer: Any, question: str, system_prompt: str, prompt_style: str = "default",
    *, tokenize: bool = False,
) -> str | list[int]:
    user_prompt = (
        f"Solve the following math problem. Show your reasoning, then end with "
        f"the final answer in \\boxed{{}}.\n\nProblem:\n{question}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if prompt_style == "areal":
        messages = [{"role": "user", "content": question + BOXED_INSTRUCTION}]
    if getattr(tokenizer, "chat_template", None):
        result = tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=True,
            **({"tools": None} if prompt_style == "areal" else {}),
        )
        # transformers 5 returns a BatchEncoding for tokenize=True.
        if tokenize:
            return list(result["input_ids"] if hasattr(result, "keys") else result)
        return result
    if prompt_style == "areal":
        raise ValueError("The AReaL prompt style requires the training tokenizer's chat template")
    text = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    return tokenizer.encode(text, add_special_tokens=False) if tokenize else text


def areal_output_text(output: dict[str, Any], tokenizer: Any) -> str:
    """Decode like AReaL, retaining immediate-stop responses as empty samples."""
    meta = output["meta_info"]
    reason = meta["finish_reason"]
    reason = reason["type"] if isinstance(reason, dict) else reason
    if reason == "abort":
        raise RuntimeError(f"SGLang aborted a request: {meta['finish_reason']}")
    if "output_token_logprobs" not in meta:
        raise ValueError("AReaL decoding requires return_logprob=True and output_token_logprobs")
    tokens = [item[1] for item in meta["output_token_logprobs"]]
    stops = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
    # Immediate EOS/PAD is a possible sampled answer. Training's strict
    # assertion would abort evaluation; keep this draw in the denominator and
    # grade it as an empty, incorrect answer instead of dropping/resampling it.
    if not tokens or all(token in stops for token in tokens):
        return ""
    if reason != "length" and tokens:
        if tokens[-1] not in stops:
            raise ValueError("Stopped output does not end in the training tokenizer's EOS/PAD")
        while tokens and tokens[-1] in stops:
            tokens.pop()
    return tokenizer.decode(tokens)


def reward_reference(example: dict[str, Any]) -> tuple[str, str]:
    """Use the full solution like MathAgent; adapt answer-only benchmarks."""
    spec = DATASETS[example["source_dataset"]]
    row = example.get("source_row", {})
    # Older snapshots may stringify a null solution as "None". Prefer the
    # original field whenever it is present, including its actual null value.
    solution = (row[spec.solution_field] if spec.solution_field in row
                else example.get("reference_solution"))
    if solution is not None and str(solution).strip():
        return str(solution), "full_solution"
    answer = reference_answer_from_row(spec, row) if row else example["reference_answer"]
    if not answer:
        raise ValueError(f"No reward reference for {example.get('problem_id')}")
    return f"\\boxed{{{answer}}}", "boxed_answer_no_solution"


class _ScoringWarnings(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def grade_completion(
    completion: str, example: dict[str, Any], scorer: str,
) -> tuple[bool, list[str]]:
    if not completion.strip():
        return False, []
    if scorer == "benchmark-v2":
        return answers_equivalent(extract_answer(completion), example["reference_answer"]), []
    from math_verify import parse, verify

    reference, _ = reward_reference(example)
    # Observe warnings without changing math-verify's default timeout/failure
    # behavior. Keep the exact argument order in AReaL's current MathAgent.
    handler = _ScoringWarnings()
    logger = logging.getLogger("math_verify")
    logger.addHandler(handler)
    try:
        prediction = parse(completion)
        gold = parse(reference)
        return bool(verify(prediction, gold)), handler.messages
    finally:
        logger.removeHandler(handler)


def normalize_sglang_outputs(outputs: Any, expected: int) -> list[dict[str, Any]]:
    """Normalize SGLang output shapes seen across Engine API versions."""
    if isinstance(outputs, list):
        normalized = outputs
    elif isinstance(outputs, dict) and isinstance(outputs.get("text"), list):
        texts = outputs["text"]
        metadata = outputs.get("meta_info", [{} for _ in texts])
        if isinstance(metadata, dict):
            metadata = [metadata for _ in texts]
        normalized = [
            {"text": text, "meta_info": metadata[index] if index < len(metadata) else {}}
            for index, text in enumerate(texts)
        ]
    elif isinstance(outputs, dict):
        normalized = [outputs]
    else:
        raise TypeError(f"Unexpected SGLang output type: {type(outputs)!r}")
    if len(normalized) != expected:
        raise RuntimeError(f"SGLang returned {len(normalized)} outputs for {expected} prompts")
    return normalized


def output_text(output: dict[str, Any]) -> str:
    text = output.get("text", "")
    if isinstance(text, list):
        text = text[0] if text else ""
    return str(text)


def reference_answer_from_row(spec: DatasetSpec, row: dict[str, Any]) -> str:
    """Preserve structured gold answers; extract only from full solutions."""
    raw = row.get(spec.answer_field) if spec.answer_field else None
    if raw is not None and str(raw).strip():
        answer = str(raw).strip()
        if spec.key == "gsm8k":
            return extract_answer(answer, spec.key)
        boxed = extract_boxed(answer)
        return boxed if boxed is not None else answer
    solution = row.get(spec.solution_field) if spec.solution_field else None
    return extract_answer(str(solution or ""), spec.key)


def load_benchmark(
    spec: DatasetSpec, limit: int | None, seed: int, tokenizer: Any = None,
    train_max_length: int = 1024,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: install `datasets`.") from exc

    kwargs: dict[str, Any] = {"path": spec.hf_path, "split": spec.split}
    if spec.hf_name is not None:
        kwargs["name"] = spec.hf_name
    if spec.key in {"math_train", "math_test"}:
        try:
            dataset = load_dataset(**kwargs)
        except ValueError:
            from datasets import concatenate_datasets, get_dataset_config_names

            names = get_dataset_config_names(spec.hf_path) or [
                "algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
            ]
            dataset = concatenate_datasets([
                load_dataset(path=spec.hf_path, name=name, split=spec.split) for name in names
            ])
    else:
        dataset = load_dataset(**kwargs)
    indices = list(range(len(dataset)))
    if spec.key == "math_train" and train_max_length:
        if tokenizer is None:
            raise ValueError("math_train requires a tokenizer for AReaL's length filter")
        # Match AReaL's filter: tokenize the raw user content, not the chat template.
        indices = [i for i in indices if len(tokenizer.encode(
            str(dataset[i][spec.question_field]) + BOXED_INSTRUCTION
        )) <= train_max_length]
    if limit is not None and limit < len(indices):
        indices = sorted(random.Random(seed).sample(indices, limit))

    examples: list[dict[str, Any]] = []
    for source_index in indices:
        row = dict(dataset[source_index])
        question = str(row[spec.question_field])
        solution = str(row.get(spec.solution_field) or "") if spec.solution_field else ""
        reference = reference_answer_from_row(spec, row)
        if not reference:
            raise ValueError(f"Could not extract reference answer from {spec.key}[{source_index}]")
        examples.append(
            {
                "problem_id": f"{spec.key}:{source_index}",
                "source_dataset": spec.key,
                "source_index": source_index,
                "question": question,
                "reference_answer": reference,
                "reference_solution": solution,
                "source_row": row,
            }
        )
    return examples


def load_dataset_selection(
    dataset_key: str, limit: int | None, seed: int, tokenizer: Any = None,
    train_max_length: int = 1024,
) -> list[dict[str, Any]]:
    """Load either one benchmark or a group alias as one logical dataset."""
    if dataset_key in DATASETS:
        return load_benchmark(DATASETS[dataset_key], limit, seed, tokenizer, train_max_length)

    source_keys = DATASET_GROUPS[dataset_key]
    examples = [
        example
        for source_key in source_keys
        for example in load_benchmark(DATASETS[source_key], None, seed)
    ]
    if limit is not None and limit < len(examples):
        indices = sorted(random.Random(seed).sample(range(len(examples)), limit))
        examples = [examples[index] for index in indices]
    return examples


def write_dataset_snapshot(path: Path, examples: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, default=str) + "\n")


def make_sampling_params(
    args: argparse.Namespace, seeds: Sequence[int], tokenizer: Any = None,
) -> list[dict[str, Any]]:
    params = [
        {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_new_tokens": args.max_new_tokens,
            "sampling_seed": seed,
        }
        for seed in seeds
    ]
    if args.protocol == "areal-rollout":
        stops = sorted({tokenizer.eos_token_id, tokenizer.pad_token_id} - {None})
        for item in params:
            item.update(stop_token_ids=stops, ignore_eos=False,
                        skip_special_tokens=True, frequency_penalty=0.0)
    return params


def evaluate_dataset(
    engine: Any,
    tokenizer: Any,
    dataset_key: str,
    examples: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    dataset_dir: Path,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    samples_path = dataset_dir / "samples.jsonl"
    correct_counts: list[int] = []
    finishes: Counter[str] = Counter()
    warning_samples = timeout_samples = empty_samples = 0
    started = time.monotonic()

    total_batches = math.ceil(len(examples) / args.question_batch_size)
    for batch_index, offset in enumerate(
        range(0, len(examples), args.question_batch_size), start=1
    ):
        batch = examples[offset : offset + args.question_batch_size]
        prompts: list[Any] = []
        request_info: list[tuple[dict[str, Any], int, int]] = []
        seeds: list[int] = []
        for local_index, example in enumerate(batch):
            prompt = build_prompt(
                tokenizer, example["question"], args.system_prompt, args.prompt_style,
                tokenize=args.protocol == "areal-rollout",
            )
            if args.protocol == "areal-rollout" and len(prompt) + args.max_new_tokens > args.context_length:
                raise ValueError(f"Prompt + completion exceeds --context-length: {example['problem_id']}")
            global_problem_index = offset + local_index
            for sample_index in range(args.num_samples):
                request_seed = args.seed + global_problem_index * args.num_samples + sample_index
                prompts.append(prompt)
                seeds.append(request_seed)
                request_info.append((example, sample_index, request_seed))

        params = make_sampling_params(args, seeds, tokenizer)
        if args.protocol == "areal-rollout":
            generated = engine.generate(input_ids=prompts, sampling_params=params, return_logprob=True)
        else:
            generated = engine.generate(prompts, params)
        outputs = normalize_sglang_outputs(generated, len(prompts))
        grouped: dict[str, list[dict[str, Any]]] = {
            example["problem_id"]: [] for example in batch
        }
        for output, (example, sample_index, request_seed) in zip(outputs, request_info):
            completion = (areal_output_text(output, tokenizer)
                          if args.protocol == "areal-rollout" else output_text(output))
            empty_completion = not completion.strip()
            empty_samples += empty_completion
            predicted_answer = extract_answer(completion)
            correct, warnings = grade_completion(completion, example, args.scorer)
            warning_samples += bool(warnings)
            timeout_samples += any("timeout" in warning.lower() for warning in warnings)
            finish = output.get("meta_info", {}).get("finish_reason", "unknown")
            finishes[str(finish.get("type", "unknown") if isinstance(finish, dict) else finish)] += 1
            # Token logprobs are used for training-compatible decoding, not saved
            # per token: this evaluator does not train on the resulting samples.
            meta = {k: v for k, v in output.get("meta_info", {}).items() if "logprobs" not in k}
            grouped[example["problem_id"]].append(
                {
                    "sample_index": sample_index,
                    "seed": request_seed,
                    "completion": completion,
                    "empty_completion": empty_completion,
                    "predicted_answer": predicted_answer,
                    "correct": correct,
                    "scoring_warnings": warnings,
                    "meta_info": meta,
                }
            )

        for example in batch:
            samples = sorted(grouped[example["problem_id"]], key=lambda item: item["sample_index"])
            count = sum(bool(sample["correct"]) for sample in samples)
            correct_counts.append(count)
            append_jsonl(
                samples_path,
                {
                    "problem_id": example["problem_id"],
                    "source_dataset": example["source_dataset"],
                    "source_index": example["source_index"],
                    "question": example["question"],
                    "reference_answer": example["reference_answer"],
                    "reward_reference": reward_reference(example)[0] if args.scorer == "areal" else example["reference_answer"],
                    "scorer": args.scorer,
                    "num_correct": count,
                    "num_samples": args.num_samples,
                    "samples": samples,
                },
            )

        elapsed = time.monotonic() - started
        completed = min(offset + len(batch), len(examples))
        print(
            f"[{dataset_key}] batch {batch_index}/{total_batches}; "
            f"problems {completed}/{len(examples)}; elapsed {elapsed / 60:.1f} min",
            flush=True,
        )

    metrics = summarize_passk(correct_counts, args.num_samples, args.k_values)
    return metrics, correct_counts, {
        "empty_completion_samples": empty_samples,
        "scoring_warning_samples": warning_samples,
        "scoring_timeout_samples": timeout_samples,
        "finish_reasons": dict(finishes),
    }


def write_metrics_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(path: Path, dataset_name: str, rows: Sequence[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Missing dependency: install `matplotlib` to create PDF curves.") from exc

    ks = [row["k"] for row in rows]
    values = [row["pass_at_k"] for row in rows]
    errors = [row["standard_error"] for row in rows]
    lower = [max(0.0, value - 1.96 * error) for value, error in zip(values, errors)]
    upper = [min(1.0, value + 1.96 * error) for value, error in zip(values, errors)]

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(ks, values, color="#2166ac", linewidth=2.0, label="pass@k")
    axis.fill_between(ks, lower, upper, color="#67a9cf", alpha=0.25, label="95% CI")
    axis.set(xlabel="k", ylabel="pass@k", title=f"{dataset_name}: pass@k")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlim(min(ks), max(ks) if len(ks) > 1 else min(ks) + 1)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a local checkpoint or Hugging Face model's pass@k on math "
            "benchmarks using SGLang."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        help="Local checkpoint directory or Hugging Face model ID (for example Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--protocol", choices=sorted(PROTOCOLS), default="areal-rollout",
        help="Default: match examples/math/gsm8k_rl.py + math_grpo.yaml rollout; benchmark-v2 retains the old protocol",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys() | DATASET_GROUPS.keys()),
        default=DEFAULT_DATASETS,
        help=(
            "Built-in benchmarks to evaluate; arxivmath combines "
            "arxivmath0226, arxivmath0126, and arxivmath1225"
        ),
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Samples per problem; protocol default: areal=4, benchmark-v2=128")
    parser.add_argument(
        "--k",
        default=None,
        help="Comma-separated k values/ranges; default is every k from 1 to num-samples",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--question-batch-size",
        type=int,
        default=4,
        help="Problems submitted per Engine.generate call (requests = this value * num-samples)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Randomly evaluate at most N problems per dataset")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                        help="Only used with --prompt-style default")
    parser.add_argument(
        "--prompt-style", choices=["default", "areal"], default=None,
        help="areal matches AReaL's hendrycks_math user message (no system message)",
    )
    parser.add_argument("--scorer", choices=["areal", "benchmark-v2"], default=None,
                        help="areal uses full completion/solution and the training verify(prediction, gold) order")
    parser.add_argument("--tokenizer-path", default=None, help="Training tokenizer; defaults to model path")
    parser.add_argument("--train-max-length", type=int, default=1024,
                        help="Raw user-content token filter for math_train only; 0 disables")
    parser.add_argument("--output-dir", type=Path, default=Path("math_passk_results"))

    engine = parser.add_argument_group("SGLang engine")
    engine.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help=(
            "GPUs used by each tensor-parallel model replica; keep at 1 when "
            "the model fits on one GPU"
        ),
    )
    engine.add_argument(
        "--dp-size",
        type=positive_int_or_auto,
        default="auto",
        help=(
            "Number of data-parallel replicas; 'auto' uses all visible GPUs "
            "after accounting for --tp-size"
        ),
    )
    engine.add_argument("--mem-fraction-static", type=float, default=0.88)
    engine.add_argument("--dtype", default=None)
    engine.add_argument("--context-length", type=int, default=32768)
    engine.add_argument("--quantization", default=None)
    engine.add_argument("--trust-remote-code", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    resolve_protocol(args)
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("--datasets must not contain duplicate names")
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.question_batch_size < 1:
        parser.error("--question-batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.tp_size < 1:
        parser.error("--tp-size must be >= 1")
    if args.max_new_tokens < 1 or args.context_length < 1:
        parser.error("--max-new-tokens and --context-length must be >= 1")
    if args.train_max_length < 0:
        parser.error("--train-max-length must be >= 0")
    if not 0.0 <= args.temperature:
        parser.error("--temperature must be >= 0")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1]")
    if not 0.0 <= args.min_p <= 1.0:
        parser.error("--min-p must be in [0, 1]")
    requested = args.k if args.k is not None else f"1-{args.num_samples}"
    try:
        args.k_values = parse_k_values(requested, args.num_samples)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))


def resolve_model_reference(value: str, parser: argparse.ArgumentParser) -> tuple[str, str]:
    """Resolve an existing local path, otherwise retain a Hugging Face model ID."""
    candidate = Path(value).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            parser.error(f"Local checkpoint must be a directory: {candidate}")
        return str(candidate.resolve()), "local"

    # An explicitly local-looking path should fail early instead of being sent to
    # Hugging Face as an invalid repository ID. Values such as ``org/model`` are
    # intentionally not considered explicit paths because that is the standard
    # Hugging Face repository-ID form.
    explicit_local = (
        value.startswith(("./", "../", "~/", "/"))
        or value in {".", "..", "~"}
        or Path(value).is_absolute()
    )
    if explicit_local:
        parser.error(f"Local checkpoint path does not exist: {candidate}")
    return value, "huggingface"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    args.path, args.model_source = resolve_model_reference(args.path, parser)
    resolve_parallelism(args, parser)
    if args.scorer == "areal":
        try:
            from math_verify import parse, verify  # noqa: F401
        except ImportError as exc:
            raise SystemExit("AReaL scoring requires math-verify; use the same version as training (existing runs: 0.8.0).") from exc

    try:
        import sglang as sgl
    except ImportError as exc:
        raise SystemExit("Missing dependency: install SGLang (`pip install sglang`).") from exc
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency: install `transformers`.") from exc

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "protocol": args.protocol,
        "protocol_overrides": args.protocol_overrides,
        "model_reference": args.path,
        "model_source": args.model_source,
        "checkpoint_path": args.path if args.model_source == "local" else None,
        "datasets": args.datasets,
        "num_samples": args.num_samples,
        "k_values": args.k_values,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_new_tokens": args.max_new_tokens,
            "base_seed": args.seed,
        },
        "prompt": {
            "style": args.prompt_style,
            "system": args.system_prompt if args.prompt_style == "default" else None,
            "uses_chat_template": None,
        },
        "scorer": args.scorer,
        "scoring_version": scoring_version(args.scorer),
        "scoring": {
            "input": "full_completion" if args.scorer == "areal" else "extracted_answer",
            "verify_order": "prediction,gold" if args.scorer == "areal" else "gold,prediction",
            "reference": "full_solution_else_boxed_answer" if args.scorer == "areal" else "structured_answer",
            "extra_string_numeric_fallback": args.scorer != "areal",
            "empty_completion_policy": "count_as_incorrect_without_resampling",
            "timeouts": "math-verify defaults; warnings recorded; no additional retry",
        },
        "train_max_length": args.train_max_length,
        "tokenizer_path": args.tokenizer_path or args.path,
        "engine": {
            "tp_size": args.tp_size,
            "dp_size": args.dp_size,
            "requested_dp_size": args.requested_dp_size,
            "visible_gpu_count": args.visible_gpu_count,
            "engine_gpu_count": args.engine_gpu_count,
            "mem_fraction_static": args.mem_fraction_static,
            "dtype": args.dtype,
            "context_length": args.context_length,
            "skip_tokenizer_init": args.protocol == "areal-rollout",
            "quantization": args.quantization,
            "trust_remote_code": args.trust_remote_code,
            "question_batch_size": args.question_batch_size,
        },
        "limit_per_dataset": args.limit,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in ["sglang", "datasets", "transformers", "matplotlib", "math-verify", "sympy", "latex2sympy2_extended"]
            },
        },
        "dataset_specs": {
            key: (
                asdict(DATASETS[key])
                if key in DATASETS
                else {
                    "key": key,
                    "members": [asdict(DATASETS[member]) for member in DATASET_GROUPS[key]],
                }
            )
            for key in args.datasets
        },
        "status": "initializing",
    }
    config_path = run_dir / "config.json"
    json_dump(config_path, config)
    print(f"Results directory: {run_dir}")
    print(
        f"SGLang parallelism: tp_size={args.tp_size}, dp_size={args.dp_size}, "
        f"GPUs={args.engine_gpu_count}/{args.visible_gpu_count or 'undetected'}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config["prompt"]["uses_chat_template"] = bool(getattr(tokenizer, "chat_template", None))
    if args.protocol == "areal-rollout":
        config["sampling"].update(
            stop_token_ids=sorted({tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}),
            ignore_eos=False, skip_special_tokens=True, frequency_penalty=0.0,
        )

    engine_kwargs: dict[str, Any] = {
        "model_path": args.path,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "mem_fraction_static": args.mem_fraction_static,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "context_length": args.context_length,
        "random_seed": args.seed,
    }
    if args.protocol == "areal-rollout":
        engine_kwargs["skip_tokenizer_init"] = True
    elif args.tokenizer_path:
        engine_kwargs["tokenizer_path"] = args.tokenizer_path
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization

    llm = None
    all_metrics: dict[str, Any] = {}
    try:
        config["status"] = "running"
        json_dump(config_path, config)
        llm = sgl.Engine(**engine_kwargs)
        for dataset_offset, dataset_key in enumerate(args.datasets):
            dataset_dir = run_dir / dataset_key
            dataset_dir.mkdir()
            examples = load_dataset_selection(
                dataset_key, args.limit, args.seed + dataset_offset,
                tokenizer=tokenizer, train_max_length=args.train_max_length,
            )
            if not examples:
                raise ValueError(f"No problems remain after selection/filtering: {dataset_key}")
            for example in examples:
                if args.scorer == "areal":
                    reference, source = reward_reference(example)
                    example.update(reward_reference=reference, reward_reference_source=source)
            json_dump(dataset_dir / "prompt_example.json", {
                "problem_id": examples[0]["problem_id"],
                "rendered_prompt": build_prompt(tokenizer, examples[0]["question"], args.system_prompt, args.prompt_style),
                "input_ids": build_prompt(tokenizer, examples[0]["question"], args.system_prompt, args.prompt_style, tokenize=True),
            })
            write_dataset_snapshot(dataset_dir / "dataset.jsonl", examples)
            print(
                f"[{dataset_key}] loaded {len(examples)} problems; "
                f"generating {len(examples) * args.num_samples} samples"
            )
            metrics, correct_counts, diagnostics = evaluate_dataset(
                llm, tokenizer, dataset_key, examples, args, dataset_dir
            )
            result = {
                "dataset": dataset_key,
                "protocol": args.protocol,
                "scorer": args.scorer,
                "scoring_version": scoring_version(args.scorer),
                "mean_sample_reward": fmean(correct_counts) / args.num_samples,
                "num_problems": len(examples),
                "num_samples_per_problem": args.num_samples,
                "mean_correct_samples_per_problem": fmean(correct_counts),
                "problems_with_at_least_one_correct": sum(count > 0 for count in correct_counts),
                "pass_at_k": metrics,
                **diagnostics,
            }
            all_metrics[dataset_key] = result
            json_dump(dataset_dir / "metrics.json", result)
            write_metrics_csv(dataset_dir / "pass_at_k.csv", metrics)
            plot_curve(dataset_dir / "pass_at_k.pdf", dataset_key, metrics)
            json_dump(run_dir / "summary.json", all_metrics)

        config["status"] = "completed"
        config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        json_dump(config_path, config)
        print(f"Evaluation completed. Results: {run_dir}")
        return 0
    except BaseException as exc:
        config["status"] = "failed"
        config["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        config["error"] = f"{type(exc).__name__}: {exc}"
        json_dump(config_path, config)
        raise
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except Exception as shutdown_error:
                print(f"[warning] SGLang shutdown failed: {shutdown_error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
