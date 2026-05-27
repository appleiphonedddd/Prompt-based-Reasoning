"""ClassEval Benchmark Dataset.

Based on the ClassEval benchmark:
https://github.com/FudanSELab/ClassEval

Each problem provides a Python class skeleton (class header, method signatures,
and docstrings).  Models must implement all method bodies so that a hidden
unittest.TestCase suite passes.

Evaluation pipeline:
    1. Extract executable Python from the model's response (markdown or raw).
    2. Prepend any declared import statements that are missing from the code.
    3. ``exec`` the combined code in a sandboxed namespace (with timeout).
    4. Verify the target class is present in the namespace.
    5. ``exec`` the unittest test code into the same namespace.
    6. Collect and run all test methods via ``unittest.TestLoader``.
    7. score  = passed_tests / total_tests
       is_correct = all tests pass.

Dataset:
    100 class-level Python programming tasks.
    Source: ``benchmark/ClassEval/data/ClassEval.json``

Author: Egor Morozov
"""

import io
import json
import re
import threading
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_EXEC_TIMEOUT: float = 15.0   # wall-clock seconds for exec() of model code
_TEST_TIMEOUT: float = 30.0   # wall-clock seconds for the full unittest run


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_exec_namespace() -> Dict[str, Any]:
    """Return an exec namespace pre-populated with common standard-library names.

    ClassEval problems use a wider variety of modules than HumanEval/MBPP, so
    the namespace is more comprehensive.

    Returns:
        A dict suitable for the ``globals`` argument of ``exec``.
    """
    import abc
    import bisect
    import collections
    import copy
    import datetime
    import functools
    import heapq
    import itertools
    import json as _json
    import logging
    import math
    import operator
    import os
    import random
    import re as _re
    import string
    import sys
    import typing

    return {
        "__builtins__": __builtins__,
        # test framework (test code does `import unittest` via exec, but pre-seeding
        # avoids NameError if the test class inherits from unittest.TestCase
        # before that import line is processed)
        "unittest": unittest,
        # standard library
        "abc": abc,
        "bisect": bisect,
        "collections": collections,
        "copy": copy,
        "datetime": datetime,
        "functools": functools,
        "heapq": heapq,
        "itertools": itertools,
        "json": _json,
        "logging": logging,
        "math": math,
        "operator": operator,
        "os": os,
        "random": random,
        "re": _re,
        "string": string,
        "sys": sys,
        "typing": typing,
        # typing aliases
        "Any": typing.Any,
        "Callable": typing.Callable,
        "Dict": typing.Dict,
        "FrozenSet": typing.FrozenSet,
        "Iterator": typing.Iterator,
        "List": typing.List,
        "Optional": typing.Optional,
        "Sequence": typing.Sequence,
        "Set": typing.Set,
        "Tuple": typing.Tuple,
        "Union": typing.Union,
    }


def _run_with_timeout(fn: Any, timeout: float) -> tuple:
    """Run ``fn()`` in a daemon thread with a wall-clock timeout.

    Args:
        fn:      Zero-argument callable to run.
        timeout: Maximum seconds to wait.

    Returns:
        ``(result, error)`` — *error* is ``None`` on success, an exception or
        ``TimeoutError`` on failure.
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

    Prefer the content of the first fenced code block; fall back to the raw
    text so that models that do not use markdown fences are still handled.

    Args:
        response: Raw model output.

    Returns:
        Cleaned code string (no trailing whitespace).
    """
    match = re.search(r"```(?:python)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.rstrip()


def _merge_imports(import_stmts: List[str], code: str) -> str:
    """Prepend any declared imports that are missing from *code*.

    Rather than blindly prepending all imports, only the ones absent from the
    code are added.  This avoids duplicate ``import`` lines when the model
    already included them in its output.

    Args:
        import_stmts: List of import strings from the dataset (e.g.
                      ``["import logging", "import datetime"]``).
        code:         Executable code produced by the model.

    Returns:
        ``code`` with any missing imports prepended.
    """
    if not import_stmts:
        return code

    missing = [stmt for stmt in import_stmts if stmt.strip() not in code]
    if not missing:
        return code

    return "\n".join(missing) + "\n\n" + code


def _run_unittest_suite(
    namespace: Dict[str, Any],
    test_code: str,
    test_class_names: List[str],
) -> Tuple[int, int, str]:
    """Execute test code and run the unittest suite.

    Args:
        namespace:        Shared exec namespace already containing the model's
                          class implementation.
        test_code:        Python source defining one or more
                          ``unittest.TestCase`` subclasses.
        test_class_names: Names of the test classes to collect and run.

    Returns:
        ``(passed, total, output)`` — *passed* and *total* are integer counts;
        *output* is the captured ``TextTestRunner`` output string.
    """
    exec(test_code, namespace)  # noqa: S102

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls_name in test_class_names:
        cls = namespace.get(cls_name)
        if cls is not None and isinstance(cls, type):
            suite.addTests(loader.loadTestsFromTestCase(cls))

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)

    passed = result.testsRun - len(result.failures) - len(result.errors)
    return passed, result.testsRun, stream.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ClassEval(DatasetBase):
    """Benchmark for evaluating LLMs on Python class implementation (ClassEval).

    Each problem gives the model a class skeleton — the class header,
    constructor, and method signatures with docstrings — and asks it to
    implement all method bodies.  Correctness is determined by running a
    ``unittest.TestCase`` suite against the model's implementation.

    The score is the fraction of individual test methods that pass;
    ``is_correct`` is ``True`` only if every test method passes.

    Args:
        split: Unused; kept for interface consistency (default ``"test"``).

    Example::

        ds = ClassEval()
        ds.load_dataset()
        problem = ds.get_problem(0)
        result = ds.evaluate_answer("class Foo:\\n    pass", problem.ground_truth)
        print(result.score)   # 0.0  (no methods implemented)
    """

    def __init__(self, split: str = "test") -> None:
        super().__init__(split=split, dataset_name="ClassEval")

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Load ClassEval problems from the local JSON file.

        Populates ``self._data`` with a list of problem dicts, each containing
        ``task_id``, ``skeleton``, ``test``, ``class_name``,
        ``import_statement``, and ``test_classes``.

        Raises:
            RuntimeError: If ``ClassEval.json`` is missing or malformed.
        """
        data_file = Path(__file__).parent / "data" / "ClassEval.json"
        if not data_file.exists():
            raise RuntimeError(
                f"[{self.dataset_name}] Data file not found: {data_file}\n"
                "Expected ClassEval.json in benchmark/ClassEval/data/."
            )

        try:
            with open(data_file, "r", encoding="utf-8") as f:
                problems = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"[{self.dataset_name}] Failed to load '{data_file}': {exc}"
            ) from exc

        if not isinstance(problems, list):
            raise RuntimeError(
                f"[{self.dataset_name}] Expected a JSON array in '{data_file}', "
                f"got {type(problems).__name__}."
            )

        self._data = problems
        print(f"[{self.dataset_name}] Loaded {len(self._data)} problems.")

    def get_problem(self, index: int) -> Problem:
        """Return the ClassEval problem at position *index*.

        ``Problem.question`` is the class skeleton formatted to prompt the
        model for a full implementation.

        ``Problem.ground_truth`` is a dict with keys:

        - ``task_id``          — problem identifier (e.g. ``"ClassEval_0"``)
        - ``class_name``       — name of the class to implement
        - ``skeleton``         — class skeleton (header + signatures + docstrings)
        - ``test``             — Python source defining the unittest test classes
        - ``import_statement`` — list of import strings required by the class
        - ``test_classes``     — names of the unittest.TestCase subclasses

        Args:
            index: Zero-based index into the dataset.

        Returns:
            A ``Problem`` instance ready for baseline evaluation.

        Raises:
            RuntimeError: If ``load_dataset()`` has not been called.
            IndexError:   If ``index`` is out of range.
        """
        self._ensure_loaded()

        if not (0 <= index < len(self._data)):
            raise IndexError(
                f"[{self.dataset_name}] Index {index} out of range "
                f"[0, {len(self._data) - 1}]."
            )

        row = self._data[index]
        skeleton: str = row["skeleton"]

        question = (
            "Implement the following Python class by writing all method bodies.\n"
            "Provide the complete class implementation including all necessary imports.\n\n"
            f"{skeleton}"
        )

        return Problem(
            index=index,
            question=question,
            ground_truth={
                "task_id":          row["task_id"],
                "class_name":       row["class_name"],
                "skeleton":         skeleton,
                "test":             row["test"],
                "import_statement": row.get("import_statement", []),
                "test_classes":     row.get("test_classes", []),
            },
            metadata={
                "task_id":    row["task_id"],
                "class_name": row["class_name"],
            },
        )

    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Evaluate the model's class implementation against the unittest suite.

        Pipeline:

        1. Extract code from *prediction* (prefer markdown fence, else raw text).
        2. Prepend any declared import statements absent from the extracted code.
        3. ``exec`` the combined code in a sandboxed namespace (``_EXEC_TIMEOUT`` s).
        4. Verify the target class exists in the namespace.
        5. ``exec`` the test code into the same namespace (``_TEST_TIMEOUT`` s).
        6. Collect all listed test classes and run them via ``TestLoader``.
        7. ``score`` = passed / total; ``is_correct`` = all tests pass.

        Args:
            prediction:   The model's raw text output
                          (``BaselineResponse.final_answer``).
            ground_truth: Dict with keys ``class_name``, ``test``,
                          ``import_statement``, and ``test_classes``.

        Returns:
            ``EvaluationResult`` with ``is_correct``, ``score``, and
            diagnostic ``details``.
        """
        details: Dict[str, Any] = {"raw_prediction": prediction}

        if not isinstance(ground_truth, dict) or "test" not in ground_truth:
            details["error"] = "ground_truth must be a dict with a 'test' key."
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        class_name:       str       = ground_truth["class_name"]
        test_code:        str       = ground_truth["test"]
        import_stmts:     List[str] = ground_truth.get("import_statement", [])
        test_class_names: List[str] = ground_truth.get("test_classes", [])
        details["class_name"] = class_name

        # ── Step 1: extract code ───────────────────────────────────────────
        code = _extract_code(prediction)
        details["code_preview"] = code[:400]

        # ── Step 2: merge missing imports ─────────────────────────────────
        full_code = _merge_imports(import_stmts, code)

        # ── Step 3: exec implementation ────────────────────────────────────
        namespace = _build_exec_namespace()
        _, error = _run_with_timeout(
            lambda: exec(full_code, namespace),  # noqa: S102
            _EXEC_TIMEOUT,
        )
        if error is not None:
            details["error"] = f"Implementation exec failed: {error}"
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        # ── Step 4: verify class present ──────────────────────────────────
        if class_name not in namespace or not isinstance(namespace[class_name], type):
            details["error"] = (
                f"Class '{class_name}' not found in namespace after exec. "
                "The model may have omitted the class definition."
            )
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        # ── Step 5: run unittest suite ────────────────────────────────────
        result, error = _run_with_timeout(
            lambda: _run_unittest_suite(namespace, test_code, test_class_names),
            _TEST_TIMEOUT,
        )

        if error is not None:
            details["error"] = (
                f"Test runner timed out: {error}"
                if isinstance(error, TimeoutError)
                else f"Test runner error: {error}"
            )
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        passed, total, output = result
        details["test_output"] = output[:1500]
        details["passed"] = passed
        details["total"] = total

        score = passed / total if total > 0 else 0.0
        is_correct = (passed == total) and (total > 0)

        return EvaluationResult(
            is_correct=is_correct,
            score=score,
            prediction=prediction,
            ground_truth=ground_truth,
            details=details,
        )

    # ── Optional hook overrides ────────────────────────────────────────────

    def get_instruction(self) -> str:
        """Return task-specific instruction for class implementation."""
        return (
            "Implement the Python class below by writing all method bodies.\n"
            "Include any necessary imports and provide the complete class implementation.\n"
            "Your code will be tested against a hidden unittest test suite."
        )

    def get_system_prompt(self) -> str:
        """Return system prompt for ClassEval evaluation."""
        return (
            "You are an expert Python programmer. "
            "Implement the given Python class by filling in all method bodies. "
            "Write clean, correct Python that satisfies all constraints described in the docstrings. "
            "Output only the complete class implementation with all imports — no explanations."
        )

    def get_demonstrations(self, n_shot: int = 1) -> str:
        """Hand-crafted ClassEval demonstration for RoT warm-up.

        The ground truth holds the hidden unittest suite, not a reference
        implementation, so the demonstration is one canonical worked example:
        a class skeleton as input and the complete class implementation as
        output, matching the task's required answer style.
        """
        return (
            "Input:\n"
            "class Calculator:\n"
            '    """A simple calculator."""\n'
            "    def add(self, a, b):\n"
            '        """Return the sum of a and b."""\n'
            "        pass\n"
            "Output:\n"
            "class Calculator:\n"
            '    """A simple calculator."""\n'
            "    def add(self, a, b):\n"
            '        """Return the sum of a and b."""\n'
            "        return a + b"
        )
