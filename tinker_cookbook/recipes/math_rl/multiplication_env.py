"""
Multiplication RL Environment

Train a model to multiply 3-digit numbers - something LLMs actually struggle with!

Example:
    "What is 847 * 293?" → "248171"

This is a task where RL can genuinely improve model capabilities,
unlike simple addition which models already know.
"""

import re
from functools import partial
from typing import Literal, Sequence

import chz
import numpy as np
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Unified regex for integer extraction and format validation
# Matches: "248171", "The answer is 248171", "248,171", "The answer is 248,171"
_INT_RE = re.compile(r"^\s*(?:The answer is\s*)?(-?(?:\d{1,3}(?:,\d{3})+|\d+))\s*$", re.IGNORECASE)


class MultiplicationEnv(ProblemEnv):
    """
    An environment for solving multiplication problems.

    Unlike addition, LLMs genuinely struggle with multi-digit multiplication,
    making this a task where RL training can show real improvement.
    """

    def __init__(
        self,
        x: int,
        y: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        super().__init__(renderer, convo_prefix)
        self.x = x
        self.y = y

    def get_question(self) -> str:
        return f"What is {self.x} * {self.y}? Answer with only the integer."

    def _extract_candidate_int(self, sample_str: str) -> int | None:
        """Extract integer from response using unified regex.

        Only extracts if the response matches the expected format exactly.
        This prevents issues like "Confidence: 0.9" extracting 9.
        """
        m = _INT_RE.fullmatch(sample_str.strip())
        if not m:
            return None
        return int(m.group(1).replace(",", ""))

    def check_answer(self, sample_str: str) -> bool:
        candidate = self._extract_candidate_int(sample_str)
        return candidate == self.x * self.y

    def answer_reward(self, sample_str: str) -> tuple[float, float]:
        """Reward correct suffix digits to provide dense feedback.

        Returns (shaped_reward, correctness_metric):
        - Perfect answer: (1.0, 1.0)
        - Partial match: (0.0 to 0.95, 0.0) - smooth scaling by suffix match
        - No match / invalid: (0.0, 0.0)
        """
        correct_value = self.x * self.y
        candidate = self._extract_candidate_int(sample_str)
        if candidate is None:
            return 0.0, 0.0

        # Reject negative numbers (impossible for positive inputs x, y)
        if candidate < 0:
            return 0.0, 0.0

        if candidate == correct_value:
            return 1.0, 1.0

        correct_str = str(correct_value)
        candidate_str = str(candidate)

        # Count contiguous matching suffix digits
        k = 0
        for i in range(1, min(len(correct_str), len(candidate_str)) + 1):
            if candidate_str[-i:] == correct_str[-i:]:
                k = i
            else:
                break

        # Smooth reward up to 0.95 (leaves gap for perfect answer)
        # This removes the old 0.8 cap that prevented signal for 5th/6th digits
        dense = 0.95 * (k / len(correct_str))
        return dense, 0.0

    def check_format(self, sample_str: str) -> bool:
        """Check if response matches expected integer format.

        Uses same regex as extraction for consistency.
        Accepts: "248171", "The answer is 248171", "248,171", etc.
        """
        return _INT_RE.fullmatch(sample_str.strip()) is not None

    def get_reference_answer(self) -> str:
        return str(self.x * self.y)

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        """Provide few-shot examples to show the expected format.

        Includes 2-digit and 3-digit examples to cover easy/medium difficulties.
        """
        return [
            {"role": "user", "content": "What is 12 * 15?"},
            {"role": "assistant", "content": "180"},
            {"role": "user", "content": "What is 34 * 27?"},
            {"role": "assistant", "content": "918"},
            {"role": "user", "content": "What is 847 * 293?"},
            {"role": "assistant", "content": "248171"},
        ]


class MultiplicationDataset(RLDataset):
    """Dataset that generates random multiplication problems."""

    def __init__(
        self,
        batch_size: int,
        renderer: renderers.Renderer,
        group_size: int,
        n_batches: int = 100,
        include_fewshot: bool = True,
        difficulty: Literal["easy", "medium", "hard"] = "medium",
    ):
        self._rng = np.random.RandomState(None)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.n_batches = n_batches
        self.include_fewshot = include_fewshot
        self.difficulty = difficulty

        # Set number ranges based on difficulty
        self.ranges = {
            "easy": (10, 100),      # 2-digit × 2-digit (10-99)
            "medium": (100, 1000),  # 3-digit × 3-digit (100-999)
            "hard": (1000, 10000),  # 4-digit × 4-digit (1000-9999)
        }

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        self._rng.seed(index)
        return [self._make_env_group_builder(self._rng) for _ in range(self.batch_size)]

    def _make_env_group_builder(self, rng: np.random.RandomState) -> ProblemGroupBuilder:
        low, high = self.ranges[self.difficulty]
        x = int(rng.randint(low, high))
        y = int(rng.randint(low, high))
        convo_prefix = MultiplicationEnv.standard_fewshot_prefix() if self.include_fewshot else None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MultiplicationEnv, x, y, convo_prefix=convo_prefix, renderer=self.renderer
            ),
            num_envs=self.group_size,
            dataset_name=f"multiplication_{self.difficulty}",
        )

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class MultiplicationDatasetBuilder(RLDatasetBuilder):
    """Builder for multiplication RL dataset."""

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    n_batches: int
    group_size: int
    include_fewshot: bool = True
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    async def __call__(self) -> tuple[MultiplicationDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return MultiplicationDataset(
            batch_size=self.batch_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            n_batches=self.n_batches,
            include_fewshot=self.include_fewshot,
            group_size=self.group_size,
            difficulty=self.difficulty,
        ), None
