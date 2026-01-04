"""
Train a model to multiply 3-digit numbers using RL.

This is a task where LLMs genuinely struggle, so you'll see real improvement!

SETUP:
    1. Set your Tinker API key:
       export TINKER_API_KEY=your-key

    2. (Optional) Set Weights & Biases key for logging:
       export WANDB_API_KEY=your-key

Example usage:
    # Quick test (~$0.50) - verify everything works
    python -m tinker_cookbook.recipes.math_rl.train_multiplication \
        model_name=Qwen/Qwen3-4B-Instruct-2507 \
        n_batches=10 \
        batch_size=20 \
        difficulty=easy

    # Medium run with W&B logging (~$5)
    python -m tinker_cookbook.recipes.math_rl.train_multiplication \
        model_name=Qwen/Qwen3-4B-Instruct-2507 \
        n_batches=100 \
        batch_size=50 \
        difficulty=medium \
        wandb_project=multiplication-rl

    # Full training (~$15)
    python -m tinker_cookbook.recipes.math_rl.train_multiplication \
        model_name=meta-llama/Llama-3.1-8B-Instruct \
        n_batches=200 \
        batch_size=50 \
        difficulty=medium \
        wandb_project=multiplication-rl

Difficulty levels:
    - easy:   2-digit × 2-digit (10-99)    - models ~70% accurate
    - medium: 3-digit × 3-digit (100-999)  - models ~20% accurate
    - hard:   4-digit × 4-digit (1000-9999) - models ~5% accurate

Model recommendations:
    - Qwen/Qwen3-4B-Instruct-2507  : Cheapest, good for testing ($0.22/M tokens)
    - meta-llama/Llama-3.1-8B-Instruct : Good balance ($0.40/M tokens)
    - Qwen/Qwen3-30B-A3B : Best value for capability ($0.30/M tokens, MoE)
"""

import asyncio
from datetime import datetime
from typing import Literal

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl.multiplication_env import MultiplicationDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    """Command-line configuration for multiplication RL training."""

    # Model configuration
    # Recommended: Qwen/Qwen3-4B-Instruct-2507 (cheap), meta-llama/Llama-3.1-8B-Instruct (balanced)
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 16         # Reduced from 32 for RL stability
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Training hyperparameters
    batch_size: int = 50        # Problems per batch
    group_size: int = 4         # Samples per problem (for GRPO variance)
    n_batches: int = 100        # Total training batches
    learning_rate: float = 3e-5 # Reduced from 1e-4 for RL stability
    kl_penalty_coef: float = 0.01  # Small KL penalty to prevent format drift
    max_tokens: int = 64        # Short responses for math (just the number)

    # Task configuration
    # easy: 2-digit (70% base acc), medium: 3-digit (20% acc), hard: 4-digit (5% acc)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    include_fewshot: bool = True

    # Weights & Biases configuration
    # Set WANDB_API_KEY env var, then pass wandb_project to enable logging
    wandb_project: str | None = None  # e.g., "multiplication-rl"
    wandb_name: str | None = None     # Auto-generated if not provided

    # Local logging
    log_path: str | None = None
    eval_every: int = 10        # Evaluate every N batches
    save_every: int = 20        # Save checkpoint every N batches

    # Service configuration
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def main(config: CLIConfig) -> None:
    """Run multiplication RL training."""

    # Get renderer name
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )

    # Build run name
    model_tag = config.model_name.split("/")[-1]
    run_name = (
        f"multiply-{config.difficulty}-{model_tag}-"
        f"lr{config.learning_rate}-bs{config.batch_size}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M')}"
    )

    # Set log path
    log_path = config.log_path or f"/tmp/tinker-examples/multiplication/{run_name}"

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)

    # Build dataset
    dataset_builder = MultiplicationDatasetBuilder(
        batch_size=config.batch_size,
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        n_batches=config.n_batches,
        group_size=config.group_size,
        include_fewshot=config.include_fewshot,
        difficulty=config.difficulty,
    )

    # Build training config
    train_config = train.Config(
        model_name=config.model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        max_tokens=config.max_tokens,
        lora_rank=config.lora_rank,
        kl_penalty_coef=config.kl_penalty_coef,
        eval_every=config.eval_every,
        save_every=config.save_every,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name or run_name,
        base_url=config.base_url,
        load_checkpoint_path=config.load_checkpoint_path,
    )

    # Estimate cost based on model
    tokens_per_batch = config.batch_size * config.group_size * 150  # ~150 tokens/sample
    total_tokens = tokens_per_batch * config.n_batches

    # Cost rates per model (combined prefill+sample+train per M tokens)
    cost_rates = {
        "Qwen/Qwen3-4B-Instruct-2507": 0.51,   # $0.07 + $0.22 + $0.22
        "Qwen/Qwen3-8B": 0.93,                  # $0.13 + $0.40 + $0.40
        "meta-llama/Llama-3.1-8B-Instruct": 0.93,
        "meta-llama/Llama-3.1-8B": 0.93,
        "Qwen/Qwen3-30B-A3B": 0.78,            # $0.12 + $0.30 + $0.36 (MoE!)
    }
    rate = cost_rates.get(config.model_name, 0.93)  # Default to 8B rate
    est_cost = total_tokens * rate / 1_000_000

    # Print training info
    print()
    print("=" * 60)
    print("🔢 MULTIPLICATION RL TRAINING")
    print("=" * 60)
    print(f"Model:        {config.model_name}")
    print(f"LoRA rank:    {config.lora_rank}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"KL penalty:   {config.kl_penalty_coef}")
    print(f"Difficulty:   {config.difficulty}")
    print(f"Batches:      {config.n_batches}")
    print(f"Batch size:   {config.batch_size} problems × {config.group_size} samples")
    print(f"Total samples: {config.n_batches * config.batch_size * config.group_size:,}")
    print("-" * 60)
    print(f"Log path:     {log_path}")
    if config.wandb_project:
        print(f"W&B project:  {config.wandb_project}")
        print(f"W&B run:      {config.wandb_name or run_name}")
    else:
        print("W&B:          disabled (pass wandb_project to enable)")
    print("-" * 60)
    print(f"Est. cost:    ~${est_cost:.2f}")
    print("=" * 60)
    print()

    # Run training
    await train.main(train_config)


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    asyncio.run(main(config))
