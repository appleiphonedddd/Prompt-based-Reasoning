"""
Abstract base class for Benchmark Datasets.

This module defines the DatasetBase abstract class which enforces a
consistent interface for all benchmark dataset implementations.

Every concrete dataset must implement:
- load_dataset():       Download/load from Hugging Face
- get_problem(index):   Retrieve a single problem by index
- evaluate_answer():    Score a predicted answer against ground truth

Design Notes:
    Follows the same ABC pattern as models/base.py (BaseLLM),
    keeping the project architecture consistent.

Author: Egor Morozov
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Problem:
    """Standardised container for a single benchmark problem.

    Attributes:
        index:        Position in the dataset split.
        question:     The raw question / puzzle string shown to the model.
        ground_truth: The expected correct answer (string or structured data).
        metadata:     Any dataset-specific extras (e.g. language, difficulty).
    """
    index: int
    question: str
    ground_truth: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.question[:60] + "..." if len(self.question) > 60 else self.question
        return f"Problem(index={self.index}, question='{preview}')"


@dataclass
class EvaluationResult:
    """Standardised container for a single evaluation outcome.

    Attributes:
        is_correct:   Whether the prediction is considered correct.
        score:        Numeric score in [0.0, 1.0].
        prediction:   The model's raw answer string.
        ground_truth: The reference answer used for comparison.
        details:      Dataset-specific diagnostic information.
    """
    is_correct: bool
    score: float
    prediction: str
    ground_truth: Any
    details: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBase(ABC):
    """Abstract base class for all benchmark dataset implementations.

    Subclasses must implement the three abstract methods below.
    Optional hook methods (``get_instruction``, ``get_system_prompt``,
    ``__len__``) may be overridden to provide richer behaviour.

    Attributes:
        dataset_name: Human-readable name shown in logs and reports.
        split:        HuggingFace dataset split to load (e.g. "test").
        _data:        The raw HuggingFace dataset object (set by load_dataset).

    Example::

        class MyDataset(DatasetBase):
            def load_dataset(self): ...
            def get_problem(self, index): ...
            def evaluate_answer(self, prediction, ground_truth): ...

        ds = MyDataset()
        problem = ds.get_problem(0)
        result  = ds.evaluate_answer("42", problem.ground_truth)
    """

    def __init__(self, split: str = "test", dataset_name: str = "DatasetBase"):
        """Initialise the dataset wrapper.

        Args:
            split:        HuggingFace split to load (default: ``"test"``).
            dataset_name: Identifier used in logs and reports.
        """
        self.split = split
        self.dataset_name = dataset_name
        self._data = None           # populated by load_dataset()

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def load_dataset(self) -> None:
        """Download and cache the dataset from Hugging Face.

        Must assign the loaded data to ``self._data``.

        Raises:
            ImportError:  If the ``datasets`` package is unavailable.
            RuntimeError: If the download fails.
        """

    @abstractmethod
    def get_problem(self, index: int) -> Problem:
        """Return the problem at the given position.

        Args:
            index: Zero-based index into the current split.

        Returns:
            A ``Problem`` instance with ``question`` and ``ground_truth``.

        Raises:
            IndexError:   If ``index`` is out of range.
            RuntimeError: If ``load_dataset()`` has not been called yet.
        """

    @abstractmethod
    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Score a model prediction against the reference answer.

        Args:
            prediction:   The model's raw output string.
            ground_truth: The reference answer (type may vary per dataset).

        Returns:
            An ``EvaluationResult`` with ``is_correct`` and ``score``.
        """

    # ── Optional hook methods ──────────────────────────────────────────────

    def get_instruction(self) -> Optional[str]:
        """Return a task-specific instruction prepended to every prompt.

        Override in subclasses to inject dataset-specific guidance.
        Returns ``None`` by default (no extra instruction).
        """
        return None

    def get_system_prompt(self) -> Optional[str]:
        """Return an optional system-level prompt for the LLM.

        Override in subclasses to set a persona or constraint.
        Returns ``None`` by default.
        """
        return None

    def get_demonstrations(self, n_shot: int = 2) -> Optional[str]:
        """Return input-output demonstrations for RoT reverse-reasoning warm-up.

        Reversal of Thought (RoT) reverse-engineers the task definition from a
        small set of input-output examples ``D`` (the paper uses 1- or 2-shot).
        This hook supplies those examples, formatted as one block per shot::

            Input: <question>
            Output: <expected output>

        (blocks separated by a blank line).

        The default implementation derives demonstrations from the first
        ``n_shot`` loaded problems, rendering each expected output via
        :meth:`_demo_output`. It is well-suited to tasks whose ground truth is
        a clean scalar/string answer (e.g. MGSM, BIG-Bench Hard). Datasets
        whose ground truth is *not* the literal expected output (e.g. Game of
        24, code-generation tasks) should override this method with
        hand-crafted canonical demonstrations.

        Args:
            n_shot: Number of demonstration examples to include.

        Returns:
            A formatted demonstration string, or ``None`` when no
            demonstrations can be produced (RoT then warms up without
            grounding examples).
        """
        if self._data is None or len(self) == 0 or n_shot < 1:
            return None

        blocks: List[str] = []
        for i in range(min(n_shot, len(self))):
            problem = self.get_problem(i)
            output = self._demo_output(problem)
            if output is None:
                # Cannot derive a clean output — subclass should override.
                return None
            blocks.append(f"Input: {problem.question}\nOutput: {output}")

        return "\n\n".join(blocks) if blocks else None

    def _demo_output(self, problem: "Problem") -> Optional[str]:
        """Render a problem's expected output for a demonstration block.

        Default: stringify scalar/string ground truths. Returns ``None`` for
        structured ground truths (dicts, lists, …), signalling that the
        dataset should override :meth:`get_demonstrations` with hand-crafted
        examples instead of relying on the default derivation.

        Args:
            problem: The problem whose output should be rendered.

        Returns:
            The output string, or ``None`` if it cannot be derived cleanly.
        """
        gt = problem.ground_truth
        if isinstance(gt, bool):  # bool is an int subclass — keep as True/False
            return str(gt)
        if isinstance(gt, (str, int, float)):
            return str(gt)
        return None


    def __len__(self) -> int:
        """Return the number of problems in the loaded split.

        Returns:
            0 if ``load_dataset()`` has not yet been called.
        """
        return len(self._data) if self._data is not None else 0

    def _ensure_loaded(self) -> None:
        """Guard helper: raise RuntimeError if data is not yet loaded."""
        if self._data is None:
            raise RuntimeError(
                f"[{self.dataset_name}] Dataset not loaded. "
                "Call load_dataset() before accessing problems."
            )

    def __repr__(self) -> str:
        size = len(self) if self._data is not None else "not loaded"
        return (
            f"{self.__class__.__name__}("
            f"dataset_name='{self.dataset_name}', "
            f"split='{self.split}', "
            f"size={size})"
        )