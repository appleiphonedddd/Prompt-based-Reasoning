"""MBPP Benchmark Dataset.

Based on Google's Mostly Basic Python Problems (MBPP) benchmark:
https://github.com/google-research/google-research/tree/master/mbpp

Each problem provides a natural-language task description. Models must
generate a complete Python function that satisfies a set of assert-style
test cases.

Evaluation pipeline:
    1. Extract executable Python from the model's response.
    2. ``exec`` the model's code in a sandboxed namespace (with timeout).
    3. ``exec`` the optional ``test_setup_code`` in the same namespace.
    4. Run each assertion in ``test_list`` with a per-assert timeout.
    5. score = fraction of passing tests; is_correct = all tests pass.

Dataset:
    974 programming tasks.
    Source: ``benchmark/MBPP/data/mbpp.jsonl``

Author: Egor Morozov
"""

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_EXEC_TIMEOUT: float = 10.0   # wall-clock seconds for exec() of model code
_ASSERT_TIMEOUT: float = 5.0  # wall-clock seconds per individual assert


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_exec_namespace() -> Dict[str, Any]:
    """Return an exec namespace pre-populated with common standard-library names.

    Returns:
        A dict suitable for the ``globals`` argument of ``exec``.
    """
    import math
    import collections
    import itertools
    import functools
    import string
    import re as _re
    import heapq
    import bisect
    import operator
    from typing import (
        Any as _Any, Callable, Dict as _Dict, FrozenSet, Iterator,
        List as _List, Optional as _Optional, Set as _Set,
        Sequence, Tuple, Union,
    )

    return {
        "__builtins__": __builtins__,
        "math": math,
        "collections": collections,
        "itertools": itertools,
        "functools": functools,
        "string": string,
        "re": _re,
        "heapq": heapq,
        "bisect": bisect,
        "operator": operator,
        "List": _List,
        "Dict": _Dict,
        "Set": _Set,
        "Tuple": Tuple,
        "Optional": _Optional,
        "Any": _Any,
        "Union": Union,
        "Callable": Callable,
        "Iterator": Iterator,
        "Sequence": Sequence,
        "FrozenSet": FrozenSet,
    }


def _run_with_timeout(fn: Any, timeout: float) -> tuple:
    """Run ``fn()`` in a daemon thread with a wall-clock timeout.

    Args:
        fn:      Zero-argument callable.
        timeout: Maximum seconds to wait.

    Returns:
        ``(result, error)`` — *error* is ``None`` on success, an exception or
        ``TimeoutError`` otherwise.
    """
    container: List[Any] = [None, None]

    def _target() -> None:
        try:
            container[0] = fn()
        except Exception as exc:  # noqa: BLE001
            container[1] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None, TimeoutError(f"Execution timed out after {timeout}s")
    return container[0], container[1]


def _extract_code(response: str) -> str:
    """Return the executable portion of a model response.

    Prefer the first fenced code block; fall back to the raw text.

    Args:
        response: Raw model output.

    Returns:
        Cleaned code string (no trailing whitespace).
    """
    match = re.search(r"```(?:python)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.rstrip()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MBPP(DatasetBase):
    """Benchmark for evaluating LLMs on Python function generation (MBPP).

    Each problem presents a natural-language description and the model must
    write a complete Python function.  Correctness is determined by running
    assert-style tests from the dataset against the model's implementation.

    Args:
        split: Unused; kept for interface consistency (default ``"test"``).

    Example::

        ds = MBPP()
        ds.load_dataset()
        problem = ds.get_problem(0)
        result = ds.evaluate_answer("def min_cost(): return 0", problem.ground_truth)
        print(result.is_correct)  # False
    """

    def __init__(self, split: str = "test") -> None:
        super().__init__(split=split, dataset_name="MBPP")

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Load MBPP problems from the local JSONL file.

        Populates ``self._data`` with a list of problem dicts, each containing
        ``task_id``, ``text``, ``code``, ``test_list``, and ``test_setup_code``.

        Raises:
            RuntimeError: If ``mbpp.jsonl`` is missing or malformed.
        """
        data_file = Path(__file__).parent / "data" / "mbpp.jsonl"
        if not data_file.exists():
            raise RuntimeError(
                f"[{self.dataset_name}] Data file not found: {data_file}\n"
                "Expected mbpp.jsonl in benchmark/MBPP/data/."
            )

        problems: List[Dict[str, Any]] = []
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        problems.append(json.loads(line))
        except Exception as exc:
            raise RuntimeError(
                f"[{self.dataset_name}] Failed to load '{data_file}': {exc}"
            ) from exc

        self._data = problems
        print(f"[{self.dataset_name}] Loaded {len(self._data)} problems.")

    def get_problem(self, index: int) -> Problem:
        """Return the MBPP problem at position *index*.

        ``Problem.question`` is the task description plus example test cases so
        the model understands the expected function signature and behaviour.

        ``Problem.ground_truth`` is a dict with keys:

        - ``task_id``         — integer problem ID
        - ``test_list``       — list of assert strings
        - ``test_setup_code`` — optional setup code that runs before tests

        Args:
            index: Zero-based position in the dataset.

        Returns:
            A ``Problem`` instance ready for baseline evaluation.

        Raises:
            RuntimeError: If ``load_dataset()`` has not been called.
            IndexError:   If *index* is out of range.
        """
        self._ensure_loaded()

        if not (0 <= index < len(self._data)):
            raise IndexError(
                f"[{self.dataset_name}] Index {index} out of range "
                f"[0, {len(self._data) - 1}]."
            )

        row = self._data[index]
        tests_str = "\n".join(row["test_list"])
        question = (
            f"{row['text']}\n\n"
            "Your function must pass the following tests:\n"
            f"{tests_str}"
        )

        return Problem(
            index=index,
            question=question,
            ground_truth={
                "task_id":         row["task_id"],
                "test_list":       row["test_list"],
                "test_setup_code": row.get("test_setup_code", ""),
            },
            metadata={"task_id": row["task_id"]},
        )

    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Evaluate the model's function against the MBPP test suite.

        Pipeline:

        1. Extract code from *prediction* (markdown fence or raw text).
        2. ``exec`` the code in a sandboxed namespace (``_EXEC_TIMEOUT`` s).
        3. ``exec`` ``test_setup_code`` in the same namespace if non-empty.
        4. Run each assert in ``test_list`` (``_ASSERT_TIMEOUT`` s each).
        5. ``score`` = fraction of passing assertions;
           ``is_correct`` = all assertions pass.

        Args:
            prediction:   The model's raw text output.
            ground_truth: Dict with keys ``test_list`` and ``test_setup_code``.

        Returns:
            ``EvaluationResult`` with ``is_correct``, ``score``, and
            diagnostic ``details``.
        """
        details: Dict[str, Any] = {"raw_prediction": prediction}

        if not isinstance(ground_truth, dict) or "test_list" not in ground_truth:
            details["error"] = "ground_truth must be a dict with a 'test_list' key."
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        test_list: List[str] = ground_truth["test_list"]
        setup_code: str = ground_truth.get("test_setup_code", "") or ""

        # ── Step 1: extract code ───────────────────────────────────────────
        code = _extract_code(prediction)
        details["code_preview"] = code[:400]

        # ── Step 2: exec model code ────────────────────────────────────────
        namespace = _build_exec_namespace()
        _, error = _run_with_timeout(
            lambda: exec(code, namespace),  # noqa: S102
            _EXEC_TIMEOUT,
        )
        if error is not None:
            details["error"] = f"Code exec failed: {error}"
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        # ── Step 3: exec setup code (uses symbols from model code) ─────────
        if setup_code.strip():
            _, error = _run_with_timeout(
                lambda: exec(setup_code, namespace),  # noqa: S102
                _EXEC_TIMEOUT,
            )
            if error is not None:
                details["error"] = f"Setup code exec failed: {error}"
                return EvaluationResult(
                    is_correct=False, score=0.0,
                    prediction=prediction, ground_truth=ground_truth,
                    details=details,
                )

        # ── Step 4: run each assertion ────────────────────────────────────
        passed = 0
        test_results: List[Dict[str, Any]] = []
        for assertion in test_list:
            _, error = _run_with_timeout(
                lambda a=assertion: exec(a, namespace),  # noqa: S102
                _ASSERT_TIMEOUT,
            )
            ok = error is None
            if ok:
                passed += 1
            test_results.append({"assertion": assertion, "passed": ok,
                                  "error": str(error) if error else None})

        details["test_results"] = test_results
        total = len(test_list)
        score = passed / total if total > 0 else 0.0
        is_correct = passed == total

        return EvaluationResult(
            is_correct=is_correct,
            score=score,
            prediction=prediction,
            ground_truth=ground_truth,
            details=details,
        )

    # ── Optional hook overrides ────────────────────────────────────────────

    def get_instruction(self) -> str:
        """Return task-specific instruction for MBPP function generation."""
        return (
            "Write a complete Python function that solves the described task.\n"
            "Include the full function definition (with `def`).\n"
            "Do not include test code or example calls — only the function."
        )

    def get_system_prompt(self) -> str:
        """Return system prompt for MBPP evaluation."""
        return (
            "You are an expert Python programmer. "
            "Write a correct, complete Python function that solves the given task. "
            "Output only the function definition — no explanations, no test code."
        )

    def get_demonstrations(self, n_shot: int = 1) -> str:
        """Hand-crafted MBPP demonstration for RoT warm-up.

        The ground truth holds the hidden test list, not a reference solution,
        so the demonstration is one canonical worked example: a natural-language
        task plus an assertion as input and a complete function definition as
        output, matching the task's required answer style.
        """
        return (
            "Input: Write a function to find the sum of squares of the first n "
            "natural numbers.\n"
            "assert sum_of_squares(3) == 14\n"
            "Output:\n"
            "def sum_of_squares(n):\n"
            "    return sum(i * i for i in range(1, n + 1))"
        )
