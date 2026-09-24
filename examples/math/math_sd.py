# SPDX-License-Identifier: Apache-2.0

"""Fixed-model generation followed by hard-label SFT, in separate processes."""

import asyncio
import json
import sys
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from omegaconf import MISSING

from areal.api.cli_args import (
    GRPOConfig,
    SFTConfig,
    SGLangConfig,
    TrainEngineConfig,
    vLLMConfig,
)
from areal.utils import logging

logger = logging.getLogger("SelfDistillation")


@dataclass
class NonIterativeSDConfig(GRPOConfig):
    """Example-only settings; generation never loads or updates a train actor."""

    stage: str = "generate"
    generated_dataset_path: str = MISSING

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage not in ("generate", "train"):
            raise ValueError("stage must be 'generate' or 'train'")
        if self.actor.init_from_scratch:
            raise ValueError(
                "Self-distillation must initialize the student from actor.path"
            )
        if self.gconfig.n_samples < 1 or self.train_dataset.batch_size < 1:
            raise ValueError("n_samples and train_dataset.batch_size must be positive")


def trajectory_rows(trajectory: dict[str, torch.Tensor]) -> list[dict[str, list[int]]]:
    """Export exact tokens and unshifted SFT masks, dropping only padding."""
    ids = trajectory["input_ids"]
    attention = trajectory["attention_mask"].bool()
    mask = trajectory["loss_mask"]
    if ids.ndim != 2 or ids.shape != attention.shape or ids.shape != mask.shape:
        raise ValueError(
            "input_ids, attention_mask and loss_mask must have shape [B,S]"
        )
    rows = []
    for tokens, valid, loss_mask in zip(ids, attention, mask):
        tokens, loss_mask = tokens[valid], loss_mask[valid]
        if not loss_mask.bool().any():
            raise ValueError("Cannot train SFT on a generated sample with no response")
        if not ((loss_mask == 0) | (loss_mask == 1)).all():
            raise ValueError("SFT loss_mask must contain only 0 and 1")
        rows.append(
            {"input_ids": tokens.tolist(), "loss_mask": loss_mask.int().tolist()}
        )
    return rows


async def _clear_remote_batch(batch: list[dict[str, Any]]) -> None:
    from areal.infra.rpc.rtensor import RTensor

    await asyncio.gather(
        *(
            RTensor.clear_node(addr, ids)
            for addr, ids in RTensor.collect_shards(batch).items()
        )
    )


def generate_dataset(config: NonIterativeSDConfig) -> None:
    from datasets import DatasetDict, load_dataset

    from areal.api.alloc_mode import ModelAllocation
    from areal.dataset import get_custom_dataset
    from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
    from areal.infra import LocalScheduler, SlurmScheduler
    from areal.infra.rpc.rtensor import RTensor
    from areal.infra.utils.concurrent import run_async_task
    from areal.utils.hf_utils import load_hf_tokenizer

    output = Path(config.generated_dataset_path)
    if output.exists():
        raise FileExistsError(f"Generated dataset already exists: {output}")
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    sources = {}
    for split, dataset_config in (
        ("train", config.train_dataset),
        ("test", config.valid_dataset),
    ):
        if dataset_config is None:
            continue
        # This process consumes prompts locally; no unconnected RDataset proxy.
        sources[split] = get_custom_dataset(
            split=split,
            dataset_config=replace(
                dataset_config, scheduling_spec=None, drop_last=False
            ),
            tokenizer=tokenizer,
            **(dataset_config.dataset_kwargs or {}),
        )
        if len(sources[split]) == 0:
            raise ValueError(f"No prompts in {split} split")

    rollout_config = deepcopy(config.rollout)
    # StalenessManager counts completed prompt groups even with frozen weights.
    # Budget the whole finite corpus without advancing the model version.
    n_prompts = sum(len(dataset) for dataset in sources.values())
    rollout_config.max_head_offpolicyness = n_prompts
    rollout_config.consumer_batch_size = config.train_dataset.batch_size
    allocation = ModelAllocation.from_str(rollout_config.backend, name="rollout")
    if allocation.backend == "sglang":
        engine_cls = RemoteSGLangEngine
        server_args = SGLangConfig.build_args(
            sglang_config=replace(config.sglang, model_path=config.actor.path),
            tp_size=allocation.parallel.tp_size,
            base_gpu_id=0,
        )
    elif allocation.backend == "vllm":
        engine_cls = RemotevLLMEngine
        server_args = vLLMConfig.build_args(
            vllm_config=replace(config.vllm, model=config.actor.path),
            tp_size=allocation.parallel.tp_size,
            pp_size=allocation.parallel.pp_size,
        )
    else:
        raise ValueError("Fixed generation requires a sglang or vllm rollout backend")
    schedulers = {"local": LocalScheduler, "slurm": SlurmScheduler}
    if config.scheduler.type not in schedulers:
        raise ValueError(
            "MathAgent proxy generation requires scheduler.type=local or slurm"
        )
    # With direct script execution, NonIterativeSDConfig lives in __main__ and
    # cannot be imported by RPC workers. Keep stage/output settings local and
    # send only the framework's importable experiment config to every worker.
    worker_config = GRPOConfig(
        **{f.name: deepcopy(getattr(config, f.name)) for f in fields(GRPOConfig)}
    )
    scheduler = schedulers[config.scheduler.type](exp_config=worker_config)
    rollout = engine_cls.as_controller(rollout_config, scheduler)
    try:
        rollout.initialize(role="rollout", server_args=server_args)
        # Use the same math agent and chat rendering as math_isd / math_opd.
        rollout.start_proxy()
        with TemporaryDirectory() as scratch:
            generated = {}
            for split, prompts in sources.items():
                json_path = Path(scratch) / f"{split}.jsonl"
                n_rows = 0
                with json_path.open("w") as stream:
                    batch_size = config.train_dataset.batch_size
                    for start in range(0, len(prompts), batch_size):
                        batch = [
                            prompts[i]
                            for i in range(start, min(start + batch_size, len(prompts)))
                        ]
                        remote_batch = rollout.rollout_batch(
                            batch,
                            workflow="areal.workflow.openai.math_agent.MathAgent",
                            workflow_kwargs=dict(
                                temperature=config.gconfig.temperature,
                                top_p=config.gconfig.top_p,
                                max_tokens=config.gconfig.max_tokens,
                                max_completion_tokens=config.gconfig.max_new_tokens,
                            ),
                            group_size=config.gconfig.n_samples,
                        )
                        try:
                            rows = [
                                row
                                for traj in RTensor.localize(remote_batch)
                                for row in trajectory_rows(traj)
                            ]
                            if len(rows) != len(batch) * config.gconfig.n_samples:
                                raise RuntimeError(
                                    "Generation returned an incomplete batch"
                                )
                            for row in rows:
                                stream.write(json.dumps(row) + "\n")
                            n_rows += len(rows)
                        finally:
                            run_async_task(_clear_remote_batch, remote_batch)
                logger.info(
                    "Generated %d %s responses from the fixed initial model",
                    n_rows,
                    split,
                )
                generated[split] = load_dataset(
                    "json", data_files=str(json_path), split="train", cache_dir=scratch
                )
            DatasetDict(generated).save_to_disk(str(output))
            manifest = {
                "model_path": config.actor.path,
                "tokenizer_path": config.tokenizer_path,
                "n_samples": config.gconfig.n_samples,
                "temperature": config.gconfig.temperature,
                "top_p": config.gconfig.top_p,
                "seed": config.seed,
            }
            (output / "self_distillation.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
    finally:
        try:
            rollout.destroy()
        finally:
            scheduler.delete_workers()


def make_sft_config(config: NonIterativeSDConfig) -> SFTConfig:
    """Keep base training settings; an SFT batch contains responses, not prompts."""
    sft = SFTConfig(
        **{f.name: deepcopy(getattr(config, f.name)) for f in fields(SFTConfig)}
    )
    sft.actor = TrainEngineConfig(
        **{
            f.name: deepcopy(getattr(config.actor, f.name))
            for f in fields(TrainEngineConfig)
        }
    )
    for dataset_config in (sft.train_dataset, sft.valid_dataset):
        if dataset_config is not None:
            dataset_config.path = config.generated_dataset_path
            dataset_config.type = "sft"
            dataset_config.batch_size *= config.gconfig.n_samples
            dataset_config.max_length = None  # Already enforced at generation time.
            dataset_config.scheduling_spec = None
    return sft


def train_dataset(config: NonIterativeSDConfig) -> None:
    from datasets import DatasetDict, load_from_disk

    from areal import SFTTrainer

    path = Path(config.generated_dataset_path)
    manifest = json.loads((path / "self_distillation.json").read_text())
    for key in ("tokenizer_path", "n_samples"):
        expected = getattr(config.gconfig if key == "n_samples" else config, key)
        if manifest[key] != expected:
            raise ValueError(f"Generated dataset {key} does not match training config")
    if manifest["model_path"] != config.actor.path:
        raise ValueError(
            "Self-distillation student must start from the generating model"
        )
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, DatasetDict) or "train" not in dataset:
        raise ValueError("Expected a generated DatasetDict with a train split")
    sft = make_sft_config(config)
    if sft.valid_dataset is not None and "test" not in dataset:
        raise ValueError(
            "Validation is configured but generated dataset has no test split"
        )
    with SFTTrainer(sft, dataset["train"], dataset.get("test")) as trainer:
        trainer.train()


def main(args: list[str]) -> None:
    from areal.api.cli_args import load_expr_config
    from areal.utils import seeding
    from areal.utils.environ import is_single_controller

    if not is_single_controller():
        raise ValueError(
            "math_sd.py requires single-controller mode; run it with python"
        )
    config, _ = load_expr_config(args, NonIterativeSDConfig)
    seeding.set_random_seed(config.seed, key="self-distillation")
    if config.stage == "generate":
        generate_dataset(config)
    else:
        train_dataset(config)


if __name__ == "__main__":
    main(sys.argv[1:])
