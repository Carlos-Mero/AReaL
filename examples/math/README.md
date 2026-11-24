## GSM8K supervised distillation

Run the existing SFT entry point with the teacher-trace recipe from the repository root.
The teacher writes samples to `./datasets/<experiment_name>/<trial_name>/` under the
launch directory. The student reads `train.jsonl` and, when validation is enabled,
`validation.jsonl` from that directory. Training logs and checkpoints use
`AREAL_FILEROOT`, which defaults to `$PWD/experiments`.

```bash
python3 examples/math/gsm8k_sft.py --config examples/math/gsm8k_sft_distill.yaml
```

The first run launches an offline SGLang teacher on the driver node. The default
`teacher_generation.backend=sglang:d8p1t1` gives each of eight visible GPUs its own
Qwen3-14B replica. The same node can run four two-GPU replicas with `sglang:d4p1t2`. The
driver submits tokenized chat prompts directly to SGLang and keeps up to
`teacher_generation.batch_size` requests in flight across all replicas (256 in this
recipe). It saves the generated answers, releases the teacher GPUs, then starts
Qwen2.5-1.5B-Instruct student SFT using `actor.backend`.

The distillation recipes set `train_dataset.num_workers=0` and
`valid_dataset.num_workers=0` so the student builds batches in the process that
consumes them. With worker subprocesses, passing a batch of per-sample tensors through the
queue can exhaust the Linux file descriptor limit (`Too many open files`). You can
apply these two overrides to an existing recipe and reuse its completed teacher traces.

The teacher displays a progress bar for each split. The counter tracks retained samples,
and `requests done` includes completions rejected by the student length filter.
`output tok/s` is the average output token rate across all replicas, measured from
completed requests since that split started. During full training-split generation, the
progress bar adjusts its total as it discovers overlong prompts.

The teacher draws validation prompts from the source training split using
`teacher_generation.seed`. The holdout is fixed across training-step budgets. The
official GSM8K test split remains available for final evaluation.

With an explicit `total_train_steps`, the math SFT entry point logs the estimated steps
per epoch immediately after loading the source. If the requested steps fit strictly
inside one epoch, preparation stops at the requested steps times the student's global
`train_dataset.batch_size` (rounded up for sampler sharding when needed). Prompt-length
rejections cost no teacher requests. Generated answers that exceed the student length
limit are replaced until the budget is met or the source runs out. Validation has its
own budget. For example, ten steps at batch size 256 require 2,560 usable training
responses plus the configured validation holdout. The trainer logs the final step count
after filtering and uses it for the learning rate schedule. `total_train_steps=null`
prepares the full training split.

Repeating a trial reuses compatible teacher responses in its directory. To compare
learning rates across trials using the same samples, point the new trial at the
first trial's directory.

```bash
python3 examples/math/gsm8k_sft.py --config examples/math/gsm8k_sft_distill.yaml \
  trial_name=lr1e5 train_dataset.path=./datasets/gsm8k-sft-distill/baseline \
  actor.optimizer.lr=1e-5 total_train_epochs=2
```

To prepare the data on a separate accelerator host, pass `prepare_only=true` and set
`train_dataset.path` to the same absolute shared directory for preparation and training.
Set a new `train_dataset.path` when changing
the teacher, its sampling settings, or its checkpoint contents. A full compatible cache
can serve a short run, and a capped cache can serve a smaller compatible budget. Choose
a fresh directory to increase the budget beyond a capped cache. The recipes use separate
default directories for the SGLang traces. To explicitly train on traces from the
previous Transformers generator or other teacher data, set `teacher_generation=null` and
point `train_dataset.path` at a directory with `train.jsonl` and `validation.jsonl`.
Each line contains a prompt and the raw teacher continuation, without the terminal stop
marker:

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"}],"response":"The answer is 4.","finish_reason":"stop","enable_thinking":false}
```

The loader masks the student chat prompt, including the assistant generation prefix. It
supervises the response and includes the student EOS for `stop` completions. `length`
marks a truncated completion and trains on the available tokens. The generated JSONL
also retains the reference `answer` for scoring. Use the same prompts, thinking mode and
generation evaluator for SFT and OPD task accuracy. The independent GSM8K evaluator in
`examples/math/gsm8k_eval.py` disables thinking by default. To train on the full GSM8K
training split used by the RL recipe, set
`teacher_generation.validation_size=0 valid_dataset=null` and choose a fresh data
directory.

The DAPO recipe defaults to `tongyx361/DAPO-Math-Unique-17k`. It reads the `train` split
and uses its chat prompts for teacher generation. The teacher supplies the SFT targets,
and `label` or `reward_model.ground_truth` supplies the scoring reference. The recipe
holds out 256 prompts for validation and caps student training at 400 steps. With the
inherited global batch size of 256 and one epoch, the full deduplicated split fits
within that budget. With `train_dataset.batch_size=32`, 400 steps target 12,800 usable
training examples.
[Deduplicated dataset](https://huggingface.co/datasets/tongyx361/DAPO-Math-Unique-17k)

The shared dataset loader also accepts this Hub ID for `train_dataset.path`. With
`type: rl`, it returns the original chat messages and a reference answer for scoring.
With `type: sft`, the student learns the reference answer and EOS. The distillation
recipe below uses `teacher_sft` to train on teacher-generated solutions. The Hub dataset
provides a single `train` split. For RL, set `valid_dataset=null` or configure a
separate evaluation set.

```bash
python3 examples/math/gsm8k_sft.py --config examples/math/dapo_math_sft_distill.yaml
```

Set `DAPO_MATH_17K_PATH` to use a local mirror or a single Parquet or JSONL file. To use
every eligible training example, add
`total_train_steps=null teacher_generation.validation_size=0 valid_dataset=null`. When
comparing against a previous run with a different split, set a new `train_dataset.path`
for its traces.

## Hyper-parameters for GSM8K Finetuning on Qwen2.5-1.5b-Instruct

`gsm8k_grpo.yaml` uses `6e-6`, retuned for the current FSDP FP32-master optimizer path.

### Current FP32-Master Recipe

A seed-1 run on 8 NVIDIA A800 GPUs completed the official 10-epoch schedule with the
following held-out evaluation results:

| lr     | weight decay | group size | best eval reward | final eval reward |
| ------ | ------------ | ---------- | ---------------- | ----------------- |
| 6.0E-6 | 0.017        | 4          | **0.78412**      | 0.77767           |

### Historical Pre-FP32-Master Sweep

The results below were collected with BF16 parameter and optimizer-state storage. They
are retained for reference and are not directly comparable with the current recipe.

| lr       | weight decay | group size | max task_reward |
| -------- | ------------ | ---------- | --------------- |
| 1.70E-05 | 0.017        | 4          | **0.79570**     |
| 1.30E-05 | 0.015        | 8          | 0.79355         |
| 1.50E-05 | 0.01         | 4          | 0.79043         |
| 1.50E-05 | 0.02         | 4          | 0.78984         |
| 1.00E-05 | 0.02         | 4          | 0.78311         |
| 1.00E-05 | 0.01         | 8          | 0.78066         |

#### Training Details

- Devices: 8 Nvidia H800 GPUs
- Optimizer: Adam
- LR Scheduler: Constant
- Gradient Clipping: 1.0
- Max_new_tokens: 1024
- Max_head_offpolicyness: 2
- Training Time: ~35 minutes (batchsize 4), ~65 minutes (batchsize 8)

## Score centering with SGLang

`gsm8k_score_centering.py` uses the native single-turn `RLVRWorkflow` to carry SGLang's
sampler head IDs and probabilities into FSDP or Megatron training. Unlike `gsm8k_rl.py`,
it does not use the OpenAI agent transport, which does not carry these heads.

Start from the existing recipe with these overrides, and set model paths, resource
allocation and output directories for your environment:

```bash
python3 examples/math/gsm8k_score_centering.py \
  --config examples/math/gsm8k_grpo.yaml scheduler.type=local \
  actor.use_score_centering=true actor.recompute_logprob=false \
  actor.use_decoupled_loss=false actor.rejection_sampling=null \
  gconfig.score_centering_top_k=128 gconfig.top_k=-1 gconfig.top_p=1.0 \
  gconfig.temperature=1.0 actor.temperature=1.0
```

Enabling score centering selects REINFORCE with a detached head/tail correction,
replacing the clipped PPO surrogate. With `actor.rejection_sampling=null`, scores remain
unweighted. To enable token-level truncated importance sampling (TIS), use AReaL's
existing configuration:

```yaml
actor:
  use_score_centering: true
  recompute_logprob: false
  use_decoupled_loss: false
  rejection_sampling:
    level: token
    metric: ratio
    action: clamp
    upper: 2.0
    lower: null
```

TIS uses detached current-trainer/sampler probability ratios, capped at `upper`. The
correction centers the weighted scores; it is not multiplied by the sampled weight
again. Sequence-level rejection, masking, KL metrics and positive lower bounds are
unsupported. `upper` must be finite, greater than one and representable in FP32. The
existing default cap remains unchanged.

Ordinary FSDP training supports TP and Ulysses CP, including TP=2/CP=2. Configure the
actor allocation with the existing parallelism syntax, for example `fsdp:d1t2c2`, and
allocate four trainer GPUs plus the rollout resources. Vocabulary size must divide
evenly across TP ranks; model attention-head divisibility constraints also apply. FSDP
does not provide pipeline parallelism.

Megatron also supports TP, packed CP and PP with the same SC/TIS settings. Use the
existing actor allocation syntax, for example `megatron:d1t2c2p2` for eight trainer
GPUs, plus separate rollout resources. The correction is computed on the last pipeline
stage. See the [Megatron validation notes](score_centering_megatron.md) for tested
layouts and the corrected-verifier Countdown experiment.

The calculation uses PyTorch operations and checkpointed chunks. TP normalizes across
vocabulary shards without gathering full-vocabulary logits. CP partitions sampler heads
with the input tokens and differentiably gathers token outputs. Full local logits remain
live. No new dependency or SGLang runtime patch is required.

Support requires native SGLang rollout, temperature 1, and no sampling truncation or
penalties. SAPO/CISPO, decoupled PPO, tree training, OpenAI agent workflows and
distillation remain unsupported. Megatron additionally requires ordinary packed logits:
chunked LM-head loss, MTP training and differentiable entropy are unsupported.
Evaluation disables head collection. Other training paths retain their existing behavior
when score centering is disabled.

Tests cover dense-reference gradients and optimizer updates, vocabulary shards, CP
padding and inactive partitions, actor alignment and sampler metadata. Run:

```bash
python -m pytest tests/test_score_centering.py tests/test_score_centering_tis.py \
  tests/test_score_centering_vocab_parallel.py tests/test_score_centering_parallel.py \
  tests/test_score_centering_megatron.py -q
```

The opt-in real-model tests require a local Qwen3 checkpoint and four CUDA devices:

```bash
AREAL_TEST_MODEL=/path/to/local/qwen3-checkpoint python -m pytest \
  tests/test_score_centering_parallel.py -k local_checkpoint -q
```

The Megatron checkpoint tests additionally cover PP and CP=4; the largest layout needs
eight CUDA devices. Set the same `AREAL_TEST_MODEL` and select
`tests/test_score_centering_megatron.py -k local_checkpoint` to run them.

The earlier [Countdown reproduction notes](score_centering_countdown.md) describe plain
SC. Multi-node and NPU execution have not been validated. This feature makes no
performance claim.
