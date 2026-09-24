# A Distribution Matching Perspective on LLM Training Algorithms

This is the source code repository for **A Distribution Matching Perspective on LLM
Training Algorithms**. The training framework is built on
[AReaL](https://github.com/areal-project/AReaL).

## Environment setup

Follow the environment setup for **AReaL v1.0.4**. The
[installation guide](https://areal-ai.io/docs/en/tutorial/installation.html) includes the corresponding
runtime images and installation instructions. The training configurations in this
repository use SGLang for inference and FSDP for training.

Run the commands below from the root of this repository with the training environment
activated. Replace `<exp_name>` and `<trial_name>` with names for your experiment and
run.

## GRPO training

To run standard GRPO on **Qwen/Qwen3-1.7B-Base** with the MATH training set:

```sh
python3 examples/math/gsm8k_rl.py \
  --config examples/math/math_grpo.yaml \
  scheduler.type=local \
  experiment_name="<exp_name>" trial_name="<trial_name>"
```

To monitor training metrics, configure either **SwanLab** or **Weights & Biases** in the
YAML configuration:

- **SwanLab:** set `stats_logger.swanlab.mode` to `cloud` and provide credentials
  through `SWANLAB_API_KEY`. Optionally set `stats_logger.swanlab.project`.
- **Weights & Biases:** set `stats_logger.wandb.mode` to `online` and provide
  credentials through `WANDB_API_KEY`. Optionally set `stats_logger.wandb.project`.

## Reproducing the main experiments

### Iterative self-distillation and reversed iterative self-distillation

Use the same training entry point with the corresponding configuration:

```sh
# Iterative self-distillation (ISD)
python3 examples/math/gsm8k_rl.py \
  --config examples/math/math_isd.yaml \
  scheduler.type=local \
  experiment_name="<exp_name>" trial_name="<trial_name>"

# Reversed iterative self-distillation (RISD)
python3 examples/math/gsm8k_rl.py \
  --config examples/math/math_risd.yaml \
  scheduler.type=local \
  experiment_name="<exp_name>" trial_name="<trial_name>"
```

### Conditional KL and KL regularization

Starting from `math_grpo.yaml`, set `actor.ckl_ctl` for Conditional KL or `actor.kl_ctl`
for standard KL regularization. The two main settings in the paper are:

```sh
# Conditional KL, coefficient 1.0
python3 examples/math/gsm8k_rl.py \
  --config examples/math/math_grpo.yaml \
  scheduler.type=local actor.ckl_ctl=1.0 \
  experiment_name="<exp_name>" trial_name="<trial_name>"
```

```sh
# Standard KL regularization, coefficient 0.3
python3 examples/math/gsm8k_rl.py \
  --config examples/math/math_grpo.yaml \
  scheduler.type=local actor.kl_ctl=0.3 \
  experiment_name="<exp_name>" trial_name="<trial_name>"
```

Both coefficients default to zero in `math_grpo.yaml`, so these commands enable only the
specified regularizer.

## Checkpoints and evaluation

AReaL automatically saves Hugging Face checkpoints according to the `saver`
configuration. Checkpoints are stored under:

```text
<cluster.fileroot>/checkpoints/<username>/<exp_name>/<trial_name>/default/
  epoch<epoch>epochstep<step>globalstep<global_step>/
```

Use the final saved checkpoint for evaluation. **For a step-limited run, ensure that
checkpoint saving includes the last training step:** the current trainer does not force
an extra save when `total_train_steps` stops training before the configured epoch
boundary. For example, append `saver.freq_steps=200` to a run that completes 200
updates, or `saver.freq_steps=400` to a run that completes 400 updates.

Evaluate the checkpoint with [math_passk_eval.py](math_passk_eval.py):

```sh
python3 math_passk_eval.py "<checkpoint_path>"
```

The positional argument accepts either a local Hugging Face checkpoint directory or a
Hugging Face model ID, such as `Qwen/Qwen3-1.7B-Base` for evaluating the initial model.

By default, the script evaluates **MATH-500, AIME 2025, AIME 2024, and APEX 2025** using
SGLang. Its `areal-rollout` protocol uses a user-only math prompt, temperature `1.0`,
`top_p=1.0`, up to `2048` new tokens, and **4 samples per problem**. Answers are graded
with `math-verify`, and pass@k is reported for every k from 1 through the sample count.
Use the same `math-verify` version as training when comparing results; the evaluator
notes version `0.8.0` for existing runs.

For example, to collect 128 samples per problem and report selected pass@k values:

```sh
python3 math_passk_eval.py "<checkpoint_path>" \
  --datasets math500 aime2025 aime2024 apex2025 \
  --num-samples 128 \
  --k 1,2,4,8,16,32,64,128 \
  --output-dir math_passk_results
```

Useful evaluation options:

| Option                        | Description                                                                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--datasets`                  | Select one or more benchmarks. Additional choices include `gsm8k`, `math_test`, `aime2026`, `apexsl-2025`, and `arxivmath` (a combined benchmark). |
| `--num-samples N`             | Generate N answers per problem. Every requested k must be at most N.                                                                               |
| `--k 1,2,4`                   | Select pass@k values; comma-separated ranges such as `1-8` are also supported.                                                                     |
| `--max-new-tokens N`          | Set the maximum response length; default: 2048.                                                                                                    |
| `--tp-size N` / `--dp-size N` | Set GPUs per model replica and the number of replicas. Defaults: TP=1 and automatic DP using all visible GPUs.                                     |
| `--limit N`                   | Evaluate a random subset of at most N problems per selected benchmark for a quick check.                                                           |
| `--tokenizer-path PATH`       | Use the training tokenizer if it is stored separately from the checkpoint.                                                                         |

Results are written to `math_passk_results/<run-id>/` by default. The run directory
contains `config.json` and `summary.json`; each benchmark subdirectory contains the
generated `samples.jsonl`, `metrics.json`, `pass_at_k.csv`, and `pass_at_k.pdf`. Run
`python3 math_passk_eval.py --help` for all options.

## Hardware requirements

The training experiments generally require **at least eight NVIDIA A100 GPUs**. The
provided configurations allocate four GPUs to rollout generation and four to the
training actor.

## License and acknowledgement

This repository builds on AReaL and retains the [Apache License 2.0](LICENSE). We thank
the AReaL contributors for the underlying training infrastructure.
