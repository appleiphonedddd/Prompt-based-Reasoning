"""APPS Benchmark Dataset.

Based on the APPS (Automated Programming Progress Standard) benchmark:
https://github.com/hendrycks/apps

Two problem formats are present in the dataset:

- **stdin/stdout** (~99%): The model writes a Python program that reads from
  standard input via ``input()`` or ``sys.stdin`` and prints results.
- **fn_name** (~1%): LeetCode-style — the model implements a named function or
  a ``Solution`` class method; inputs and outputs are structured Python objects.

Evaluation pipeline (stdin/stdout):
    1. Extract the last Python code block from the model's response.
    2. For each test case: mock ``sys.stdin`` / ``sys.stdout`` / ``input()``,
       ``exec`` the code (timeout: ``_EXEC_TIMEOUT`` s), capture stdout.
    3. Normalise and compare output; score = passing / total test cases.

Evaluation pipeline (fn_name):
    1. Extract code as above.
    2. ``exec`` the code once in a sandboxed namespace (timeout).
    3. Resolve the callable: direct function **or** ``Solution`` class method.
    4. Call the function for each test case (timeout per call).
    5. Compare return values; score = passing / total test cases.

Answer extraction note:
    ``_extract_code`` picks the **last** fenced code block so that chain-of-
    thought baselines — which may emit preliminary/wrong code before the
    corrected final solution — are handled correctly.  The ``get_instruction``
    return value contains both ``"write"`` and ``"line"`` which triggers the
    ``is_generative_task`` guard inside ``ZeroShotCoT``, causing it to skip
    the second-pass answer-extraction step and return the full reasoning text
    as ``final_answer``; the code extractor then recovers the solution from it.

Dataset:
    5000 programming problems.
    Source: ``benchmark/APPS/data/test/`` (one sub-directory per problem).

Author: Egor Morozov
"""

import io
import json
import re
import sys as _sys
import threading
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_EXEC_TIMEOUT: float = 5.0    # seconds before a single exec is TLE
_MAX_TEST_CASES: int = 10     # test cases evaluated per problem (performance cap)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: pre-populated exec namespace
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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: timeout-guarded execution
# ─────────────────────────────────────────────────────────────────────────────

def _run_with_timeout(fn: Any, timeout: float) -> Tuple[Any, Optional[Exception]]:
    """Run ``fn()`` in a daemon thread with a wall-clock timeout.

    Args:
        fn:      Zero-argument callable.
        timeout: Maximum seconds to wait.

    Returns:
        ``(result, error)`` — *error* is ``None`` on success, an exception or
        ``TimeoutError`` otherwise.
    """
    container: List[Any] = [None, None]  # [result, error]

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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: code extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_code(response: str) -> str:
    """Return the **last** fenced code block from a model response.

    Picking the last block ensures chain-of-thought outputs — which often emit
    an initial (possibly wrong) solution followed by a corrected one — yield
    the final version.  Falls back to the raw text when no fenced block is
    found.

    Args:
        response: Raw model output (may include markdown fences).

    Returns:
        Stripped code string suitable for ``exec``.
    """
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return response.rstrip()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: output normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_output(s: str) -> str:
    """Normalise program output for comparison.

    Strips trailing whitespace from every line and drops a single trailing
    newline, matching standard competitive-programming judge behaviour.

    Args:
        s: Raw captured stdout string.

    Returns:
        Normalised string for equality comparison.
    """
    lines = s.rstrip("\n").split("\n")
    return "\n".join(line.rstrip() for line in lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: stdin/stdout execution
# ─────────────────────────────────────────────────────────────────────────────

def _run_stdin_stdout(
    code: str,
    input_str: str,
    timeout: float,
) -> Tuple[str, Optional[Exception]]:
    """Execute ``code`` with ``input_str`` as stdin, capturing stdout.

    Both ``input()`` (namespace override) and ``sys.stdin`` / ``sys.stdout``
    (global replacement) are mocked so that solutions using either idiom
    are handled correctly.  The global replacement is performed inside the
    worker thread and restored in a ``finally`` block.

    Args:
        code:      Python source code to execute.
        input_str: String to supply as standard input.
        timeout:   Wall-clock seconds before a ``TimeoutError`` is returned.

    Returns:
        ``(captured_stdout, error)`` — *error* is ``None`` on success.
    """
    outputs: List[str] = []
    error_container: List[Optional[Exception]] = [None]

    def _target() -> None:
        stdin_buf = io.StringIO(input_str)
        stdout_buf = io.StringIO()

        # Override input() in the exec namespace so solutions that call
        # input() directly work without touching sys.stdin.
        def mock_input(prompt: str = "") -> str:
            line = stdin_buf.readline()
            if not line:
                raise EOFError("No more input")
            return line.rstrip("\n")

        namespace = _build_exec_namespace()
        namespace["input"] = mock_input

        # Replace the real sys.stdin / sys.stdout so that solutions which do
        # `import sys; input = sys.stdin.readline` also work correctly.
        # Restored in the finally block regardless of outcome.
        old_stdin, old_stdout = _sys.stdin, _sys.stdout
        _sys.stdin = stdin_buf
        _sys.stdout = stdout_buf

        try:
            exec(code, namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            error_container[0] = exc
        finally:
            _sys.stdin = old_stdin
            _sys.stdout = old_stdout
            outputs.append(stdout_buf.getvalue())

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return "", TimeoutError(f"Execution timed out after {timeout}s")
    return outputs[0] if outputs else "", error_container[0]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: function resolution for fn_name problems
# ─────────────────────────────────────────────────────────────────────────────

def _find_callable(namespace: Dict[str, Any], fn_name: str) -> Optional[Any]:
    """Resolve a callable from an exec namespace.

    Checks two patterns in order:

    1. **Direct function**: ``namespace[fn_name]`` is callable.
    2. **Solution class method**: ``namespace["Solution"]`` is a class and has
       a callable attribute ``fn_name`` (LeetCode-style).

    Args:
        namespace: The ``globals`` dict after ``exec``-ing the model code.
        fn_name:   Name of the target function.

    Returns:
        A bound method or plain function, or ``None`` if not found.
    """
    direct = namespace.get(fn_name)
    if callable(direct):
        return direct

    solution_cls = namespace.get("Solution")
    if solution_cls is not None:
        try:
            instance = solution_cls()
            method = getattr(instance, fn_name, None)
            if callable(method):
                return method
        except Exception:  # noqa: BLE001
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class APPS(DatasetBase):
    """Benchmark for evaluating LLMs on competitive-programming problems (APPS).

    Problems are read from ``benchmark/APPS/data/test/``.  Each sub-directory
    contains a ``question.txt`` and an ``input_output.json`` file.  The JSON
    file either has a ``fn_name`` key (LeetCode-style) or lists plain
    stdin/stdout string pairs.

    Args:
        split:          Dataset split sub-directory (default: ``"test"``).
        max_test_cases: Number of test cases evaluated per problem (default:
                        ``_MAX_TEST_CASES``).  Capped to avoid excessive
                        runtime on problems with hundreds of cases.

    Example::

        ds = APPS()
        ds.load_dataset()
        problem = ds.get_problem(0)
        result = ds.evaluate_answer("print('hello')", problem.ground_truth)
        print(result.score)
    """

    def __init__(
        self,
        split: str = "test",
        max_test_cases: int = _MAX_TEST_CASES,
    ) -> None:
        super().__init__(split=split, dataset_name="APPS")
        self.max_test_cases = max_test_cases

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Load APPS problems from the local directory tree.

        Scans ``benchmark/APPS/data/{split}/`` for numbered sub-directories,
        reading ``question.txt``, ``input_output.json``, and (if present)
        ``metadata.json`` from each.  Populates ``self._data`` with a list
        of problem dicts.

        Raises:
            RuntimeError: If the data directory is missing or a required file
                          cannot be parsed.
        """
        data_dir = Path(__file__).parent / "data" / self.split
        if not data_dir.exists():
            raise RuntimeError(
                f"[{self.dataset_name}] Data directory not found: {data_dir}\n"
                f"Expected problem directories in benchmark/APPS/data/{self.split}/."
            )

        problems: List[Dict[str, Any]] = []
        for problem_dir in sorted(data_dir.iterdir()):
            if not problem_dir.is_dir():
                continue

            question_file = problem_dir / "question.txt"
            io_file = problem_dir / "input_output.json"

            if not question_file.exists() or not io_file.exists():
                continue

            try:
                question = question_file.read_text(encoding="utf-8").strip()
                with open(io_file, "r", encoding="utf-8") as f:
                    io_data = json.load(f)
            except Exception as exc:
                raise RuntimeError(
                    f"[{self.dataset_name}] Failed to load '{problem_dir.name}': {exc}"
                ) from exc

            meta_file = problem_dir / "metadata.json"
            metadata: Dict[str, Any] = {}
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                except Exception:  # noqa: BLE001
                    pass

            problems.append({
                "id":       problem_dir.name,
                "question": question,
                "fn_name":  io_data.get("fn_name"),         # None for stdin/stdout
                "inputs":   io_data.get("inputs", []),
                "outputs":  io_data.get("outputs", []),
                "metadata": metadata,
            })

        self._data = problems
        print(f"[{self.dataset_name}] Loaded {len(self._data)} problems.")

    def get_problem(self, index: int) -> Problem:
        """Return the APPS problem at position *index*.

        For **stdin/stdout** problems, ``Problem.question`` is the raw problem
        statement (which already includes input/output format descriptions).

        For **fn_name** problems, ``Problem.question`` appends an explicit
        instruction to implement the named function.

        ``Problem.ground_truth`` is a dict with keys:

        - ``problem_id`` — zero-padded directory name (e.g. ``"0042"``)
        - ``fn_name``    — function name or ``None`` for stdin/stdout problems
        - ``inputs``     — list of test inputs
        - ``outputs``    — list of expected outputs

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
        fn_name: Optional[str] = row["fn_name"]

        if fn_name:
            question = (
                f"{row['question']}\n\n"
                f"Implement a Python function named `{fn_name}` that solves this problem."
            )
        else:
            question = row["question"]

        return Problem(
            index=index,
            question=question,
            ground_truth={
                "problem_id": row["id"],
                "fn_name":    fn_name,
                "inputs":     row["inputs"],
                "outputs":    row["outputs"],
            },
            metadata={
                "problem_id": row["id"],
                "fn_name":    fn_name,
                "difficulty": row["metadata"].get("difficulty", "unknown"),
            },
        )

    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Evaluate the model's code against APPS test cases.

        Dispatches to :meth:`_evaluate_stdin_stdout` or
        :meth:`_evaluate_fn_name` depending on whether the problem has a
        ``fn_name`` key.  Always evaluates at most ``self.max_test_cases``
        test cases for performance.

        Args:
            prediction:   The model's raw text output
                          (``BaselineResponse.final_answer``).
            ground_truth: Dict with keys ``fn_name``, ``inputs``, ``outputs``.

        Returns:
            ``EvaluationResult`` with ``is_correct``, ``score`` in [0, 1],
            and diagnostic ``details``.
        """
        details: Dict[str, Any] = {"raw_prediction": prediction}

        if not isinstance(ground_truth, dict) or "inputs" not in ground_truth:
            details["error"] = "ground_truth must be a dict with 'inputs' and 'outputs' keys."
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        fn_name: Optional[str] = ground_truth.get("fn_name")
        inputs:  List[Any]     = ground_truth["inputs"]
        outputs: List[Any]     = ground_truth["outputs"]

        # ── Extract code from prediction (last code block wins) ───────────
        code = _extract_code(prediction)
        details["code_preview"] = code[:400]

        # ── Cap test cases for runtime ────────────────────────────────────
        test_inputs  = inputs[: self.max_test_cases]
        test_outputs = outputs[: self.max_test_cases]

        if fn_name:
            return self._evaluate_fn_name(
                code, fn_name, test_inputs, test_outputs,
                prediction, ground_truth, details,
            )
        return self._evaluate_stdin_stdout(
            code, test_inputs, test_outputs,
            prediction, ground_truth, details,
        )

    # ── Private evaluation helpers ─────────────────────────────────────────

    def _evaluate_stdin_stdout(
        self,
        code: str,
        inputs: List[str],
        outputs: List[str],
        prediction: str,
        ground_truth: Any,
        details: Dict[str, Any],
    ) -> EvaluationResult:
        """Run stdin/stdout test cases and score the solution.

        Args:
            code:    Extracted Python source.
            inputs:  List of stdin strings (one per test case).
            outputs: List of expected stdout strings.

        Returns:
            ``EvaluationResult`` with partial scores supported.
        """
        passed = 0
        test_results: List[Dict[str, Any]] = []

        for inp, expected in zip(inputs, outputs):
            actual, error = _run_stdin_stdout(code, inp, _EXEC_TIMEOUT)

            if error is not None:
                test_results.append({"passed": False, "error": str(error)})
                continue

            if _normalize_output(actual) == _normalize_output(expected):
                passed += 1
                test_results.append({"passed": True})
            else:
                test_results.append({
                    "passed":   False,
                    "expected": expected.strip()[:120],
                    "got":      actual.strip()[:120],
                })

        total = len(inputs)
        score = passed / total if total > 0 else 0.0
        details["test_results"] = test_results

        return EvaluationResult(
            is_correct=passed == total and total > 0,
            score=score,
            prediction=prediction,
            ground_truth=ground_truth,
            details=details,
        )

    def _evaluate_fn_name(
        self,
        code: str,
        fn_name: str,
        inputs: List[Any],
        outputs: List[Any],
        prediction: str,
        ground_truth: Any,
        details: Dict[str, Any],
    ) -> EvaluationResult:
        """Exec code, resolve the target function, and run structured test cases.

        Args:
            code:    Extracted Python source.
            fn_name: Name of the function to call.
            inputs:  List of argument lists (one per test case).
            outputs: List of expected return values.

        Returns:
            ``EvaluationResult`` with partial scores supported.
        """
        # ── Exec the model's code once ────────────────────────────────────
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

        # ── Resolve the callable ──────────────────────────────────────────
        func = _find_callable(namespace, fn_name)
        if func is None:
            details["error"] = (
                f"Function '{fn_name}' not found — expected a top-level function "
                f"or a method of a `Solution` class."
            )
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        # ── Run each test case ────────────────────────────────────────────
        passed = 0
        test_results: List[Dict[str, Any]] = []

        for inp, expected in zip(inputs, outputs):
            args = inp if isinstance(inp, list) else [inp]
            result, error = _run_with_timeout(
                lambda a=args: func(*a),
                _EXEC_TIMEOUT,
            )
            if error is not None:
                test_results.append({"passed": False, "error": str(error)})
                continue

            if result == expected:
                passed += 1
                test_results.append({"passed": True})
            else:
                test_results.append({
                    "passed":   False,
                    "expected": repr(expected)[:120],
                    "got":      repr(result)[:120],
                })

        total = len(inputs)
        score = passed / total if total > 0 else 0.0
        details["test_results"] = test_results

        return EvaluationResult(
            is_correct=passed == total and total > 0,
            score=score,
            prediction=prediction,
            ground_truth=ground_truth,
            details=details,
        )

    # ── Optional hook overrides ────────────────────────────────────────────

    def get_instruction(self) -> str:
        """Return task-specific instruction for APPS problems.

        The phrase ``"write"`` + ``"line"`` intentionally satisfies the
        ``is_generative_task`` condition in ``ZeroShotCoT``, which suppresses
        the second-pass answer-extraction step.  This causes ``ZeroShotCoT``
        to return the full reasoning as ``final_answer``, from which the
        evaluator can extract the code via ``_extract_code``.
        """
        return (
            "Write a complete Python program, line by line, that solves the problem below.\n"
            "Read input using input() and write output using print().\n"
            "Output only the Python code — no explanations outside the code."
        )

    def get_system_prompt(self) -> str:
        """Return system prompt for APPS evaluation."""
        return (
            "You are an expert Python programmer specializing in competitive programming. "
            "Write a correct, complete Python solution. "
            "Read from standard input with input() and write to standard output with print(). "
            "Output only the Python code."
        )

    def get_demonstrations(self, n_shot: int = 1) -> str:
        """Hand-crafted APPS demonstration for RoT warm-up.

        The ground truth holds the hidden I/O test cases, not a reference
        program, so the demonstration is one canonical worked example: a
        problem statement as input and a complete stdin/stdout program as
        output, matching the task's required answer style.
        """
        return (
            "Input: Read two integers A and B on a single line separated by a "
            "space and print their sum.\n"
            "Output:\n"
            "a, b = map(int, input().split())\n"
            "print(a + b)"
        )
