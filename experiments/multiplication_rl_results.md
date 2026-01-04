# Multiplication RL Experiments

**Date:** January 2-4, 2026
**Goal:** Test shaped reward changes for multiplication RL task

## Experiment Overview

Tested whether shaped rewards improve RL training for 3-digit multiplication using Tinker.

## Model Baseline Comparisons

Evaluated base model accuracy on 100 random 3-digit × 3-digit multiplication problems (seed=42):

| Model | Parameters | Base Accuracy | RL Potential |
|-------|-----------|---------------|--------------|
| Qwen/Qwen3-235B-A22B-Instruct-2507 | 235B (22B active) | **96%** | Too good (diminishing returns) |
| deepseek-ai/DeepSeek-V3.1 | 671B (37B active) | **84%** | Excellent |
| Qwen/Qwen3-4B-Instruct-2507 | 4B | 36% | Marginal |
| Qwen/Qwen3-30B-A3B | 30B (3B active) | 34% | Marginal |
| Qwen/Qwen3-32B | 32B | 26% | Poor |
| Qwen/Qwen3-8B | 8B | 24% | Poor |
| meta-llama/Llama-3.1-8B-Instruct | 8B | 14% | Very Poor |
| meta-llama/Llama-3.3-70B-Instruct | 70B | 0% | Format issues |

### Key Insight
- Instruction tuning matters more than size for math tasks
- Qwen3-4B-Instruct beats Qwen3-32B and Qwen3-8B
- Llama models struggle with this task regardless of size

---

## Successful Training Run (January 4, 2026)

After implementing ML researcher feedback, achieved **+12% accuracy improvement** on held-out evaluation.

### Changes Implemented

Based on ML researcher recommendations (categories A-G):

1. **Fixed Parsing/Format (Priority A)**
   - Unified regex for format checking and extraction
   - Allow "The answer is" prefix and comma-formatted numbers
   - Use `fullmatch` to prevent "last number wins" bug

2. **Improved Shaped Reward (Priority A)**
   - Removed 0.8 cap that limited signal for 5th/6th digits
   - Smooth scaling: `0.95 * (k / len(correct_str))`
   - Reject negative numbers (multiplication of positives)

3. **Training Stability (Priority B)**
   - Reduced learning rate: 1e-4 → 3e-5
   - Reduced LoRA rank: 32 → 16
   - Added KL penalty coefficient: 0.01

4. **Few-shot Alignment (Priority C)**
   - Added 3-digit example to few-shot prefix
   - Examples: 12×15=180, 34×27=918, 847×293=248171

### Training Config

```
Model: Qwen/Qwen3-4B-Instruct-2507
Difficulty: medium (3-digit × 3-digit)
Batches: 100
Batch size: 50 problems × 4 samples = 200 samples/batch
Learning rate: 3e-5
LoRA rank: 16
KL penalty: 0.01
```

**Run path:** `/tmp/tinker-examples/multiplication/multiply-medium-Qwen3-4B-Instruct-2507-lr3e-05-bs50-20260104-0129`

**Checkpoint:** `tinker://9ca5efd7-8e63-5fe5-95f7-6fa6673bc424:train:0/sampler_weights/final`

**WandB:** https://wandb.ai/mail2djanto-home/multiplication-rl/runs/riaecjki

### Training Metrics

| Window | Accuracy | Format | Mixed Groups |
|--------|----------|--------|--------------|
| Batches 0-9 | 35.5% | 100% | 24.4% |
| Batches 10-19 | 37.4% | 100% | - |
| Batches 20-29 | 35.8% | 100% | - |
| Batches 30-39 | 38.0% | 100% | - |
| Batches 40-49 | 39.7% | 100% | - |
| Batches 50-59 | 39.1% | 100% | - |
| Batches 60-69 | 36.0% | 100% | - |
| Batches 70-79 | **41.2%** | 100% | - |
| Batches 80-89 | 36.6% | 100% | - |
| Batches 90-99 | 37.1% | 100% | 27.6% |

- **Peak accuracy:** 49% at batch 70
- **Best 10-batch window:** 41.9% (batches 69-78)
- **Format stability:** 100% throughout (KL penalty worked!)

### Held-Out Evaluation (100 fresh problems)

| Model | Accuracy | Format |
|-------|----------|--------|
| Base model | 27% | 58% |
| Trained model | 39% | 90% |
| **Improvement** | **+12%** | **+32%** |

### What Worked

1. **KL penalty (0.01)** - Prevented format drift that plagued previous runs
2. **Lower learning rate (3e-5)** - More stable optimization
3. **LoRA rank 16** - Sufficient capacity without overfitting
4. **Improved shaped rewards** - Smooth scaling, no saturation cap
5. **Fixed regex parsing** - Consistent format checking

---

## Previous Training Run (January 2, 2026)

Initial attempt before implementing ML researcher feedback.

### Config
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Learning rate: 1e-4
- LoRA rank: 32
- No KL penalty

**Run path:** `/tmp/tinker-examples/multiplication/multiply-medium-Qwen3-4B-Instruct-2507-lr0.0001-bs50-20260102-1654`

### Results

| Metric | Value |
|--------|-------|
| Start accuracy | 33.0% |
| End accuracy | 42.0% |
| Peak accuracy | 48.5% (batch 21) |
| Min format | **73.0%** (batch 60) - degradation! |
| Held-out improvement | **-3.0%** (regression) |

### Learning Signal Quality
- `frac_mixed` (has gradient signal): 27.1%
- `frac_all_good` (no signal - all correct): 28.6%
- `frac_all_bad` (no signal - all wrong): **44.3%**

High all-bad rate indicated insufficient learning signal.

---

## Comparison: Before vs After ML Researcher Fixes

| Metric | Before (Jan 2) | After (Jan 4) |
|--------|----------------|---------------|
| Training format stability | Dropped to 73% | **Stable at 100%** |
| Held-out accuracy change | -3.0% | **+12.0%** |
| Held-out format change | - | **+32.0%** |
| Peak training accuracy | 48.5% | 49.0% |
| Learning signal (frac_mixed) | 27.1% | 27.6% |

**Key difference:** KL penalty prevented format collapse, allowing learning to transfer to held-out evaluation.

---

## RL Viability Guidelines

```
Base Accuracy    RL Outcome
─────────────────────────────────────────
0-5%             No learning (guessing)
5-20%            Slow/unreliable
20-50%           Best zone (good variance)
50-80%           Good, faster saturation
80%+             Diminishing returns
```

Key metric to watch: `frac_mixed` > 30% indicates good learning signal.

---

## Conclusions

### What Made Training Work

1. **KL penalty is critical** - Without it, RL can "cheat" by changing output format
2. **Conservative hyperparameters** - Lower LR and LoRA rank improve stability
3. **Consistent parsing** - Unified regex prevents format/reward mismatch
4. **Shaped rewards need smooth scaling** - Caps limit learning signal

### Recommendations

1. **Always use KL penalty** (0.01 is a good starting point) for format-sensitive tasks
2. **Start with lower LR** (3e-5) for LoRA training
3. **Verify format at T=0** before enabling shaped rewards
4. **Monitor frac_mixed** - should be >25% for good learning signal
5. **Use LoRA rank 16** unless task requires more capacity

---

## Reproduction

```bash
# Run training with improved config
TINKER_API_KEY=<key> WANDB_API_KEY=<key> uv run python -m tinker_cookbook.recipes.math_rl.train_multiplication \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    n_batches=100 \
    batch_size=50 \
    difficulty=medium \
    wandb_project=multiplication-rl \
    behavior_if_log_dir_exists=delete
```

---

## Available Tinker Models

```
deepseek-ai/DeepSeek-V3.1
deepseek-ai/DeepSeek-V3.1-Base
moonshotai/Kimi-K2-Thinking
meta-llama/Llama-3.1-70B
meta-llama/Llama-3.1-8B
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.2-1B
meta-llama/Llama-3.2-3B
meta-llama/Llama-3.3-70B-Instruct
Qwen/Qwen3-235B-A22B-Instruct-2507
Qwen/Qwen3-30B-A3B
Qwen/Qwen3-30B-A3B-Base
Qwen/Qwen3-30B-A3B-Instruct-2507
Qwen/Qwen3-32B
Qwen/Qwen3-4B-Instruct-2507
Qwen/Qwen3-8B
Qwen/Qwen3-8B-Base
Qwen/Qwen3-VL-235B-A22B-Instruct
Qwen/Qwen3-VL-30B-A3B-Instruct
openai/gpt-oss-120b
openai/gpt-oss-20b
```
