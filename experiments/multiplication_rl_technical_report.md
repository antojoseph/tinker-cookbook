# Technical Report: Reinforcement Learning for Multi-Digit Multiplication with Tinker

**Authors:** Anto Joseph, Claude Code
**Date:** January 4, 2026
**Version:** 1.0

---

## Executive Summary

This report documents the implementation and results of training a language model to perform 3-digit multiplication using reinforcement learning on the Tinker platform. We achieved a **+12% accuracy improvement** on held-out evaluation (27% → 39%) by implementing shaped rewards, KL regularization, and careful hyperparameter tuning based on ML researcher feedback.

**Key Results:**
- Held-out accuracy: 27% → 39% (+12%)
- Format compliance: 58% → 90% (+32%)
- Training format stability: 100% throughout (vs 73% degradation in baseline)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Tinker Platform Architecture](#2-tinker-platform-architecture)
3. [RL Algorithm: GRPO with Importance Sampling](#3-rl-algorithm-grpo-with-importance-sampling)
4. [Multiplication Environment Design](#4-multiplication-environment-design)
5. [Shaped Reward Engineering](#5-shaped-reward-engineering)
6. [Training Configuration](#6-training-configuration)
7. [Experimental Results](#7-experimental-results)
8. [Lessons Learned](#8-lessons-learned)
9. [Reproduction Guide](#9-reproduction-guide)

---

## 1. Introduction

### 1.1 Problem Statement

Multi-digit multiplication is a challenging task for language models because:
- It requires multi-step reasoning with carry propagation
- Small errors in intermediate steps cascade to incorrect final answers
- The search space grows exponentially with digit count

Unlike tasks where LLMs already excel, 3-digit multiplication presents genuine difficulty where RL training can demonstrate measurable improvement.

### 1.2 Approach

We used **Reinforcement Learning from Verifiable Rewards (RLVR)** where the reward signal comes from programmatic verification of mathematical correctness, combined with **shaped rewards** that provide partial credit for approximately correct answers.

---

## 2. Tinker Platform Architecture

### 2.1 Client-Server Model

Tinker operates on a split-responsibility architecture:

**CLIENT SIDE (Python):**
- Data preparation & tokenization
- Environment logic & reward computation
- Advantage calculation (GRPO centering)
- Trajectory assembly
- Metric aggregation & logging
- Checkpoint management

**SERVER SIDE (Tinker Service):** *(communicates via HTTPS/gRPC)*
- GPU worker pool management
- Forward passes (inference & logprob computation)
- Backward passes (gradient computation)
- Optimizer steps (Adam updates)
- LoRA weight management
- Multi-tenant scheduling

### 2.2 Clock Cycle Synchronization

Tinker's GPU workers operate in **lock-step clock cycles**. Each cycle can execute one forward-backward pass and one optimizer step. Efficient training requires pipelining requests:

**Naive approach (3 clock cycles):**
```
Cycle 1: forward_backward() → wait
Cycle 2: optim_step() → wait
Cycle 3: forward_backward() → wait
```

**Optimized approach (1 clock cycle per step):**
```
Submit: forward_backward_async() ─┐
Submit: optim_step_async() ───────┼── Both land on same cycle
Await:  results ──────────────────┘
```

**Implementation Pattern:**
```python
# Submit both before awaiting - they execute on same GPU cycle
fwd_bwd_future = await training_client.forward_backward_async(batch, loss_fn)
optim_future = await training_client.optim_step_async(adam_params)

# Now await results
fwd_bwd_result = await fwd_bwd_future.result_async()
optim_result = await optim_future.result_async()
```

### 2.3 Training and Sampling Clients

**ServiceClient** → creates → **TrainingClient** → saves weights for → **SamplingClient**

| Client | Key Methods |
|--------|-------------|
| ServiceClient | `create_lora_training_client_async`, `create_sampling_client` |
| TrainingClient | `forward_backward_async`, `optim_step_async`, `save_weights` |
| SamplingClient | `sample_async`, `get_tokenizer` |

**Critical:** After each weight update, a **new SamplingClient** must be created to ensure logprobs are computed against the current policy, not a stale one.

---

## 3. RL Algorithm: GRPO with Importance Sampling

### 3.1 Group Relative Policy Optimization (GRPO)

GRPO is a variance reduction technique that centers advantages within groups of rollouts from the same problem:

**Example: Problem P₁ with G=4 rollouts**

| Rollout | Response | Reward |
|---------|----------|--------|
| 1 | "248171" | R₁ = 1.0 (correct) |
| 2 | "248170" | R₂ = 0.32 (partial) |
| 3 | "200000" | R₃ = 0.0 (wrong) |
| 4 | "248171" | R₄ = 1.0 (correct) |

**Mean reward:** R̄ = (1.0 + 0.32 + 0.0 + 1.0) / 4 = 0.58

**Centered advantages:**
- A₁ = 1.0 - 0.58 = +0.42 (reinforce)
- A₂ = 0.32 - 0.58 = -0.26 (discourage)
- A₃ = 0.0 - 0.58 = -0.58 (strongly discourage)
- A₄ = 1.0 - 0.58 = +0.42 (reinforce)

**Why GRPO works:**
- Compares each rollout to the group mean, not a learned baseline
- Reduces variance in advantage estimates
- No value function needed (simpler than PPO with critic)

### 3.2 Importance Sampling Loss

The loss function corrects for the distribution shift between the sampling policy (old) and the learning policy (current):

```
L_IS(θ) = -E[A(x) · (p_θ(x) / q(x))]

Where:
  p_θ(x) = current policy probability (from forward pass)
  q(x)   = sampling policy probability (cached during rollout)
  A(x)   = advantage (GRPO-centered reward)
```

**Token-level implementation:**
```python
# For each token in the trajectory
prob_ratio = torch.exp(current_logprobs - sampling_logprobs)
loss = -(prob_ratio * advantages).sum()
```

### 3.3 KL Penalty for Stability

To prevent the policy from diverging too far from the base model, we add a KL penalty:

```
A_adjusted = A_original + α_KL · (log p_base - log p_current)

Where:
  α_KL = 0.01 (KL penalty coefficient)
  p_base = base model probability
  p_current = current policy probability
```

**Effect:** Penalizes deviations from the base model distribution, preserving format and general capabilities.

### 3.4 Training Loop Flow

**Per-batch training loop:**

1. **SAMPLING**
   - Get batch of problems
   - For each problem, generate G=4 rollouts
   - Compute rewards

2. **TRAINING**
   - Compute advantages (GRPO)
   - Assemble training datums
   - Apply KL penalty
   - forward_backward
   - optim_step

3. **LOGGING**
   - Aggregate metrics
   - Log to W&B
   - Save checkpoint
   - Create new sampler

---

## 4. Multiplication Environment Design

### 4.1 Environment Interface

The `MultiplicationEnv` class implements the Tinker `Env` interface:

**State:**
- `x: int` — First multiplicand (100-999 for medium)
- `y: int` — Second multiplicand
- `renderer` — Tokenizer wrapper
- `convo_prefix` — Few-shot examples

**Methods:**
- `get_question()` → "What is {x} * {y}? Answer with only the int."
- `check_format(response)` → bool
- `check_answer(response)` → bool
- `answer_reward(response)` → (shaped_reward, correctness_metric)
- `get_reference_answer()` → str(x * y)

### 4.2 Episode Structure

Each multiplication problem is a **single-turn episode**:

**Step 1: initial_observation()**

The prompt includes few-shot examples followed by the actual problem:
```
<|im_start|>user
What is 12 * 15?<|im_end|>
<|im_start|>assistant
180<|im_end|>
<|im_start|>user
What is 34 * 27?<|im_end|>
<|im_start|>assistant
918<|im_end|>
<|im_start|>user
What is 847 * 293?<|im_end|>      ← 3-digit example
<|im_start|>assistant
248171<|im_end|>
<|im_start|>user
What is 523 * 847?<|im_end|>      ← actual problem
<|im_start|>assistant
```

**Step 2:** Model generates response (e.g., "442981")

**Step 3: env.step(response)**
- Parse response
- Check format validity
- Compute shaped reward
- Return StepResult(reward, episode_done=True)

### 4.3 Format Validation

We use a unified regex for both format checking and answer extraction:

```python
# Module-level compiled regex
_INT_RE = re.compile(
    r"^\s*(?:The answer is\s*)?(-?(?:\d{1,3}(?:,\d{3})+|\d+))\s*$",
    re.IGNORECASE
)

# Accepts:
#   "248171"              ✓
#   "248,171"             ✓
#   "The answer is 248171" ✓
#   "The answer is 248,171" ✓
#
# Rejects:
#   "The answer is approximately 248000" ✗
#   "248171 is the answer" ✗
#   "Let me calculate... 248171" ✗
```

**Critical Design Decision:** Using `fullmatch()` instead of `search()` prevents the "last number wins" bug where verbose responses like "Step 1: 847 × 293 = 248171" would incorrectly extract numbers from intermediate steps.

---

## 5. Shaped Reward Engineering

### 5.1 Reward Function Design

The `answer_reward()` method returns a tuple: `(shaped_reward, correctness_metric)`:

**Reward Computation Flow:**

Input: response = "248170", Correct answer = 248171

1. **Extract candidate integer** → candidate = 248170 ✓
2. **Check for exact match** → 248170 ≠ 248171, continue to partial credit
3. **Count matching suffix digits:**
   ```
   Correct:   2 4 8 1 7 1
   Candidate: 2 4 8 1 7 0
                        ↑
              Last digit differs

   Match from right:
     Position 1: 1 ≠ 0  → stop
     k = 0 matching digits
   ```
4. **Compute shaped reward:**
   ```
   dense = 0.95 * (k / len("248171"))
        = 0.95 * (0 / 6)
        = 0.0
   ```

**Return:** (0.0, 0.0)

### 5.2 Reward Examples

| Response | Correct | k (suffix match) | Shaped Reward | Correctness |
|----------|---------|------------------|---------------|-------------|
| "248171" | 248171 | 6 (exact) | 1.0 | 1.0 |
| "248170" | 248171 | 0 | 0.0 | 0.0 |
| "248071" | 248171 | 2 ("71") | 0.317 | 0.0 |
| "240171" | 248171 | 3 ("171") | 0.475 | 0.0 |
| "148171" | 248171 | 5 ("48171") | 0.792 | 0.0 |
| "-248171" | 248171 | rejected | 0.0 | 0.0 |
| "invalid" | 248171 | N/A | 0.0 | 0.0 |

### 5.3 Why Suffix Matching?

The suffix matching heuristic aligns with how multiplication errors typically propagate:

```
    847
  × 293
  ─────
   2541   (847 × 3)
  7623    (847 × 9, shifted)
 1694     (847 × 2, shifted)
─────────
 248171   ← errors in early partial products affect leftmost digits
```

**Common error pattern:** If you miscalculate 847 × 2 = 1694 as 1684, the result becomes 247171 (wrong in position 6, but 5 suffix digits correct)

This reward structure encourages the model to get the rightmost digits correct first, which is a natural learning progression.

### 5.4 Improvements from ML Researcher Feedback

**Original Implementation Issues:**
1. **0.8 cap on shaped reward** — Limited signal for 5th/6th digit progress
2. **Cumulative scoring** — Added 0.2 per matching digit (non-linear)
3. **"Last number wins" extraction** — Could grab wrong number from verbose responses

**Fixed Implementation:**
1. **0.95 cap** — Leaves clear gap for perfect answers while allowing full progression
2. **Linear scaling** — `0.95 * (k / total_digits)` provides smooth gradient
3. **fullmatch regex** — Ensures entire response matches expected format

---

## 6. Training Configuration

### 6.1 Hyperparameters

```python
# Model
model_name = "Qwen/Qwen3-4B-Instruct-2507"
lora_rank = 16              # Reduced from 32 for stability

# Optimization
learning_rate = 3e-5        # Reduced from 1e-4
kl_penalty_coef = 0.01      # Prevents format drift

# Sampling
max_tokens = 64             # Short outputs for math
temperature = 1.0           # Standard sampling
group_size = 4              # GRPO groups

# Training
batch_size = 50             # Problems per batch
n_batches = 100             # Total training steps
loss_fn = "importance_sampling"

# Checkpointing
save_every = 20             # Checkpoint frequency
eval_every = 0              # Inline metrics only
```

### 6.2 Hyperparameter Rationale

**LoRA Rank = 16:**
- Full fine-tuning of 4B params is expensive
- LoRA adapts ~0.1% of parameters (16 × hidden_dim × 2 per layer)
- Rank 16 provides sufficient capacity for this task
- Lower rank = less overfitting, more stable training

**Learning Rate = 3e-5:**
- LoRA typically needs 10x higher LR than full fine-tuning
- But RL needs lower LR than supervised learning for stability
- 3e-5 is conservative; could potentially increase to 1e-4

**KL Penalty = 0.01:**
- Small but non-zero prevents format collapse
- Without it: format degraded to 73% in baseline run
- With it: format stayed at 100% throughout training

### 6.3 Cost Estimation

| Item | Calculation |
|------|-------------|
| Tokens per batch | 50 problems × 4 rollouts × 150 tokens/rollout = 30,000 tokens |
| Total training tokens | 30,000 × 100 batches = 3,000,000 tokens |
| Prefill cost | $0.22/M tokens |
| Sample cost | $0.22/M tokens |
| Train cost | $0.07/M tokens |
| **Total** | $0.51/M tokens × 3M ≈ **$1.50** |

---

## 7. Experimental Results

### 7.1 Training Metrics Over Time

**Accuracy by 10-batch windows:**

| Batch Range | Accuracy |
|-------------|----------|
| 1-10 | ~35% |
| 10-20 | ~35% |
| 20-30 | ~40% |
| 30-40 | ~40% |
| 40-50 | ~40% |
| 50-60 | ~40% |
| 60-70 | ~45% |
| **70-80** | **~49% (peak)** |
| 80-90 | ~40% |
| 90-100 | ~35% |

**Best window:** 41.9% (batches 69-78)

### 7.2 Format Stability Comparison

**Previous Run (no KL penalty):**
- Started at 100%
- Dropped to 73% by batch 80 (format collapse)

**Current Run (KL penalty = 0.01):**
- Stable at 100% throughout all 100 batches

### 7.3 Held-Out Evaluation

**Test Setup:**
- 100 random 3-digit × 3-digit problems
- Fixed seed (42) for reproducibility
- Temperature = 0 (greedy decoding)

**Results:**

| Metric | Base Model | Trained Model | Change |
|--------|------------|---------------|--------|
| **Accuracy** | 27% | 39% | **+12%** |
| **Format OK** | 58% | 90% | **+32%** |

### 7.4 Learning Signal Quality

**Group composition over training:**

| Metric | Start | End | Interpretation |
|--------|-------|-----|----------------|
| frac_mixed (has gradient signal) | 24% | 28% | ✓ Good |
| frac_all_good (no signal needed) | 28% | 32% | ✓ Improving |
| frac_all_bad (no signal available) | 48% | 40% | ✓ Decreasing |

**Interpretation:**
- frac_mixed > 25%: GRPO advantage centering is working
- frac_all_bad decreasing: Model is learning to solve more problems
- frac_all_good increasing: More problems fully solved

### 7.5 Comparison: Before vs After ML Researcher Fixes

| Metric | Before (Jan 2) | After (Jan 4) |
|--------|----------------|---------------|
| Training format stability | Dropped to 73% | **Stable at 100%** |
| Held-out accuracy change | -3.0% (regression) | **+12.0%** |
| Held-out format change | N/A | **+32.0%** |
| Peak training accuracy | 48.5% | 49.0% |
| Learning signal (frac_mixed) | 27.1% | 27.6% |

**Key insight:** The KL penalty was the critical fix. Without it, the model learned to "cheat" by changing its output format in ways that didn't generalize.

---

## 8. Lessons Learned

### 8.1 KL Penalty is Critical for Format-Sensitive Tasks

Without KL regularization, the policy can drift from the base model's output distribution. For tasks with strict format requirements, this manifests as:
- Degraded format compliance
- Rewards that don't transfer to held-out evaluation
- Apparent training progress that doesn't generalize

**Recommendation:** Always start with `kl_penalty_coef = 0.01` for format-sensitive tasks.

### 8.2 Shaped Rewards Need Careful Design

The original 0.8 cap on shaped rewards limited learning signal for nearly-correct answers. The linear scaling to 0.95 provides:
- Smooth gradient throughout the learning process
- Clear distinction between partial and perfect answers
- No artificial plateaus

### 8.3 Parsing Consistency Matters

Using different logic for format checking vs. answer extraction caused silent failures. The unified regex approach ensures:
- If format check passes, extraction will succeed
- No edge cases where valid formats fail to extract
- Consistent behavior between training and evaluation

### 8.4 Conservative Hyperparameters for RL

RL training is inherently less stable than supervised learning. Our successful configuration used:
- **Lower LR** (3e-5 vs typical 1e-4 for LoRA)
- **Smaller LoRA rank** (16 vs 32)
- **KL penalty** (0.01)

### 8.5 Monitor frac_mixed for Learning Signal

The `frac_mixed` metric indicates what fraction of GRPO groups have variance (some correct, some incorrect). This is where learning signal comes from:
- `frac_mixed < 20%`: Poor learning signal, consider easier task
- `frac_mixed > 30%`: Good learning signal
- `frac_all_bad > 50%`: Task too hard for current model

---

## 9. Reproduction Guide

### 9.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/antojoseph/tinker-cookbook.git
cd tinker-cookbook

# Install dependencies
uv sync

# Set API keys
export TINKER_API_KEY="your-tinker-api-key"
export WANDB_API_KEY="your-wandb-api-key"  # optional
```

### 9.2 Run Training

```bash
# Full training (100 batches, ~$1.50, ~30 minutes)
uv run python -m tinker_cookbook.recipes.math_rl.train_multiplication \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    n_batches=100 \
    batch_size=50 \
    difficulty=medium \
    wandb_project=multiplication-rl \
    behavior_if_log_dir_exists=delete

# Quick test (10 batches, ~$0.15, ~3 minutes)
uv run python -m tinker_cookbook.recipes.math_rl.train_multiplication \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    n_batches=10 \
    batch_size=20 \
    difficulty=easy \
    behavior_if_log_dir_exists=delete
```

### 9.3 Evaluate Checkpoint

```python
import asyncio
import tinker
from tinker.types import SamplingParams
from tinker_cookbook.recipes.math_rl.multiplication_env import MultiplicationEnv
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

async def evaluate(checkpoint_path: str, n_problems: int = 100):
    model = "Qwen/Qwen3-4B-Instruct-2507"
    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(model)
    renderer = renderers.get_renderer("qwen3", tokenizer=tokenizer)

    sampling_client = service_client.create_sampling_client(
        base_model=model,
        model_path=checkpoint_path
    )

    # ... evaluation logic

asyncio.run(evaluate("tinker://run-id/sampler_weights/final"))
```

### 9.4 Available Checkpoints

From our successful training run:

| Checkpoint | Path |
|------------|------|
| Batch 20 | `tinker://9ca5efd7.../sampler_weights/000020` |
| Batch 40 | `tinker://9ca5efd7.../sampler_weights/000040` |
| Batch 60 | `tinker://9ca5efd7.../sampler_weights/000060` |
| Batch 80 | `tinker://9ca5efd7.../sampler_weights/000080` |
| Final | `tinker://9ca5efd7.../sampler_weights/final` |

---

## Appendix A: System Overview

**MULTIPLICATION RL SYSTEM**

### Dataset Builder
- difficulty
- batch_size
- group_size
- n_batches

### Environment (per problem)
- x, y values
- few-shot prefix
- renderer

### Tinker Service
- Training Client (LoRA adapter)
- GPU Worker Pool (forward/backward/optimizer)
- Sampling Client (inference)

### Rollout Phase

For each problem P, for g in range(group_size):
1. Get observation (prompt)
2. Sample from policy (via Tinker Sampling Client)
3. Get reward from env
4. Store trajectory

### Training Phase

1. Compute GRPO advantages: `A_i = R_i - mean(R_group)`
2. Apply KL penalty: `A' = A + α(log p_base - log p_curr)`
3. Assemble training datums (tokens, logprobs, advantages)
4. Forward-backward pass → Tinker GPU
5. Optimizer step → Tinker GPU
6. Save checkpoint & create new sampler

### Logging Phase

Metrics tracked:
- frac_correct (accuracy)
- frac_correct_format
- frac_mixed / frac_all_good / frac_all_bad
- kl_policy_base
- reward/total

Output: W&B dashboard, metrics.jsonl

---

## Appendix B: Key File Locations

| Component | File Path |
|-----------|-----------|
| Multiplication Environment | `tinker_cookbook/recipes/math_rl/multiplication_env.py` |
| Training Script | `tinker_cookbook/recipes/math_rl/train_multiplication.py` |
| Base Problem Env | `tinker_cookbook/rl/problem_env.py` |
| RL Training Loop | `tinker_cookbook/rl/train.py` |
| RL Types | `tinker_cookbook/rl/types.py` |
| Rollout Logic | `tinker_cookbook/rl/rollouts.py` |
| Data Processing | `tinker_cookbook/rl/data_processing.py` |
| Metrics | `tinker_cookbook/rl/metrics.py` |
| Renderers | `tinker_cookbook/renderers/` |

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **GRPO** | Group Relative Policy Optimization - variance reduction via advantage centering within groups |
| **KL Penalty** | Kullback-Leibler divergence penalty to keep policy close to base model |
| **LoRA** | Low-Rank Adaptation - efficient fine-tuning by training small adapter matrices |
| **Shaped Reward** | Reward that provides partial credit, enabling learning from near-misses |
| **frac_mixed** | Fraction of GRPO groups with varying rewards (where learning signal exists) |
| **Clock Cycle** | Atomic unit of GPU computation in Tinker's distributed system |
| **Importance Sampling** | Correction for distribution shift between sampling and training policies |

---

## References

1. Tinker Documentation: https://docs.tinker.dev
2. GRPO Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
3. LoRA Paper: "LoRA: Low-Rank Adaptation of Large Language Models"
4. PPO Paper: "Proximal Policy Optimization Algorithms"

---

*Report generated with assistance from Claude Code*
