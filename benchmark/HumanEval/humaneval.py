"""HumanEval Benchmark Dataset.

Based on OpenAI's HumanEval benchmark:
https://github.com/openai/human-eval

Each problem provides a Python function signature with a docstring (the
"prompt").  Models must complete the function body so that all assertions in
the hidden ``check(candidate)`` test suite pass.

Evaluation pipeline:
    1. Extract / reconstruct executable Python from the model's response.
    2. ``exec`` the combined code (prompt imports + function) in a sandboxed
       namespace.
    3. Retrieve the implemented function by ``entry_point`` name.
    4. ``exec`` the test code (defines ``check``), then call
       ``check(candidate)`` with a 10-second wall-clock timeout.
    5. Score 1.0 if ``check`` completes without raising; 0.0 otherwise.

Dataset:
    164 programming tasks.
    Source: ``benchmark/HumanEval/data/humaneval.jsonl``

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

_EXEC_TIMEOUT: float = 10.0  # wall-clock seconds before execution is TLE


# ─────────────────────────────────────────────────────────────────────────────
# Helper: pre-populated exec namespace
# ─────────────────────────────────────────────────────────────────────────────

def _build_exec_namespace() -> Dict[str, Any]:
    """Return an exec namespace pre-populated with common standard library names.

    Many HumanEval prompts import from ``typing`` or ``math`` at the top of the
    function signature.  Executing the full code handles those imports, but
    pre-populating the namespace acts as a safety net for any edge case.

    Returns:
        A dict suitable for the ``globals`` argument of ``exec``.
    """
    import math
    import collections
    import itertools
    import functools
    import string
    import re as _re
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

def _run_with_timeout(fn: Any, timeout: float = _EXEC_TIMEOUT) -> tuple:
    """Run ``fn()`` with a wall-clock timeout via a daemon thread.

    Args:
        fn:      Zero-argument callable to run.
        timeout: Maximum seconds to wait (default: ``_EXEC_TIMEOUT``).

    Returns:
        ``(result, error)`` — *result* is ``fn()``'s return value (or ``None``),
        *error* is the raised exception (or ``None`` on success).  If the thread
        is still alive after *timeout* seconds, returns
        ``(None, TimeoutError(...))``.
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
# Helper: code extraction and solution assembly
# ─────────────────────────────────────────────────────────────────────────────

def _extract_markdown_code_block(text: str) -> Optional[str]:
    """Extract the content of the first fenced code block, if present.

    Args:
        text: Raw model output string.

    Returns:
        Stripped content inside the first ```...``` block, or ``None`` if no
        fenced block was found.
    """
    match = re.search(r"```(?:python)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _build_full_solution(response: str, entry_point: str, prompt: str) -> str:
    """Assemble executable Python from a model response and the original prompt.

    Handles two common model output styles:

    - **Complete function** — response contains ``def entry_point(``.
      Import lines are extracted from *prompt*; the model's function is used
      verbatim.  This tolerates models that re-state the signature.

    - **Completion / body only** — response does not contain a ``def`` for
      *entry_point*.  The response is appended directly to *prompt* (which
      already ends with the docstring closing ``\"\"\"\\n``).  Leading
      whitespace is preserved so that indented function bodies remain valid.

    Args:
        response:    Raw model output (may include markdown fences).
        entry_point: Name of the function the model must implement.
        prompt:      Original function signature + docstring from the dataset.

    Returns:
        A complete Python code string suitable for ``exec``.
    """
    from_block = _extract_markdown_code_block(response)
    # When a markdown block is found, use its stripped content (the model likely
    # wrote the complete function inside it).  Otherwise preserve leading
    # whitespace so an indented function body can be appended directly to prompt.
    code = from_block if from_block is not None else response.rstrip()

    fn_def_pattern = re.compile(
        rf"(?:^|\n)[ \t]*def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE
    )

    if fn_def_pattern.search(code):
        # Extract any import/from lines that appear before the first def in prompt
        import_lines: List[str] = []
        for line in prompt.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                import_lines.append(line)
            elif stripped.startswith("def "):
                break
        header = "\n".join(import_lines)
        return f"{header}\n\n{code}" if header else code

    # Body-only path: separate any extra imports from the actual function body.
    # Models sometimes write top-level `from X import Y` followed by unindented
    # body code (e.g. a bare `return` statement).  We hoist the imports above
    # the function definition and auto-indent body lines that start at column 0.
    extra_imports: List[str] = []
    body_lines: List[str] = []
    for line in code.split("\n"):
        if line.strip().startswith(("import ", "from ")):
            extra_imports.append(line.strip())
        else:
            body_lines.append(line)

    # Auto-indent lines that start at column 0 — they belong inside the function.
    fixed_body: List[str] = []
    for line in body_lines:
        if line and not line[0].isspace():
            fixed_body.append("    " + line)
        else:
            fixed_body.append(line)
    body_code = "\n".join(fixed_body).rstrip()

    if not extra_imports:
        return f"{prompt}{body_code}"

    # Merge extra imports with those already in the prompt (deduplicated).
    prompt_lines = prompt.split("\n")
    existing_imports: List[str] = [
        ln.strip() for ln in prompt_lines
        if ln.strip().startswith(("import ", "from "))
    ]
    all_imports = list(dict.fromkeys(existing_imports + extra_imports))
    imports_str = "\n".join(all_imports)

    prompt_no_imports = "\n".join(
        ln for ln in prompt_lines
        if not ln.strip().startswith(("import ", "from "))
    ).lstrip("\n")

    return f"{imports_str}\n\n{prompt_no_imports}{body_code}"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset implementation
# ─────────────────────────────────────────────────────────────────────────────

class HumanEval(DatasetBase):
    """Benchmark for evaluating LLMs on Python function completion (HumanEval).

    Each problem gives the model a function signature and docstring (the
    original HumanEval "prompt") and asks it to complete the implementation.
    Correctness is determined by running the dataset's ``check(candidate)``
    test suite against the model's implementation.

    Args:
        split: Unused; kept for interface consistency (default ``"test"``).

    Example::

        ds = HumanEval()
        ds.load_dataset()
        problem = ds.get_problem(0)
        # Provide a trivial (wrong) completion to test evaluation machinery
        result = ds.evaluate_answer("    return False", problem.ground_truth)
        print(result.is_correct)   # False
    """

    def __init__(self, split: str = "test") -> None:
        super().__init__(split=split, dataset_name="HumanEval")

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Load HumanEval problems from the local JSONL file.

        Populates ``self._data`` with a list of problem dicts, each containing
        ``task_id``, ``prompt``, ``entry_point``, ``canonical_solution``,
        and ``test``.

        Raises:
            RuntimeError: If ``humaneval.jsonl`` is missing or malformed.
        """
        data_file = Path(__file__).parent / "data" / "humaneval.jsonl"
        if not data_file.exists():
            raise RuntimeError(
                f"[{self.dataset_name}] Data file not found: {data_file}\n"
                "Expected humaneval.jsonl in benchmark/HumanEval/data/."
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
        """Return the HumanEval problem at position *index*.

        ``Problem.question`` is the function signature and docstring,
        formatted to prompt the model for a completion.

        ``Problem.ground_truth`` is a dict with keys:

        - ``task_id``     — problem identifier (e.g. ``"HumanEval/0"``)
        - ``prompt``      — function signature + docstring (used when assembling
                            the full solution for execution)
        - ``entry_point`` — name of the function to test
        - ``test``        — Python source defining ``check(candidate)``

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
        prompt: str = row["prompt"]
        entry_point: str = row["entry_point"]

        question = (
            "Complete the following Python function implementation:\n\n"
            f"{prompt}"
        )

        return Problem(
            index=index,
            question=question,
            ground_truth={
                "task_id":     row["task_id"],
                "prompt":      prompt,
                "entry_point": entry_point,
                "test":        row["test"],
            },
            metadata={
                "task_id":     row["task_id"],
                "entry_point": entry_point,
            },
        )

    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Evaluate the model's function completion against the test suite.

        Pipeline:

        1. Assemble executable Python from *prediction* and the original prompt.
        2. ``exec`` the combined code in a sandboxed namespace (with timeout).
        3. Retrieve the implemented function by ``entry_point`` name.
        4. ``exec`` the test code to define ``check``.
        5. Call ``check(candidate)`` with a 10-second timeout.

        A problem is correct if and only if ``check`` completes without raising
        any exception (``AssertionError`` or otherwise).

        Args:
            prediction:   The model's raw text output
                          (``BaselineResponse.final_answer``).
            ground_truth: Dict with keys ``task_id``, ``prompt``,
                          ``entry_point``, and ``test``.

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

        prompt: str = ground_truth["prompt"]
        entry_point: str = ground_truth["entry_point"]
        test_code: str = ground_truth["test"]
        details["entry_point"] = entry_point

        # ── Step 1: assemble full solution code ───────────────────────────
        full_code = _build_full_solution(prediction, entry_point, prompt)
        details["full_code_preview"] = full_code[:400]

        # ── Step 2: exec solution with timeout ────────────────────────────
        namespace = _build_exec_namespace()
        _, error = _run_with_timeout(lambda: exec(full_code, namespace))  # noqa: S102
        if error is not None:
            details["error"] = f"Solution exec failed: {error}"
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        # ── Step 3: retrieve the implemented function ─────────────────────
        candidate_fn = namespace.get(entry_point)
        if not callable(candidate_fn):
            details["error"] = (
                f"Function '{entry_point}' not found in namespace after exec."
            )
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        # ── Step 4: define check() from the test code ─────────────────────
        try:
            exec(test_code, namespace)  # noqa: S102
        except Exception as exc:
            details["error"] = f"Test code exec failed: {exc}"
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        check_fn = namespace.get("check")
        if not callable(check_fn):
            details["error"] = "'check' function not found in test code."
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        # ── Step 5: run check(candidate) with timeout ─────────────────────
        _, error = _run_with_timeout(lambda: check_fn(candidate_fn))

        if error is not None:
            details["error"] = f"check() failed: {error}"
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        return EvaluationResult(
            is_correct=True, score=1.0,
            prediction=prediction, ground_truth=ground_truth,
            details=details,
        )

    # ── Optional hook overrides ────────────────────────────────────────────

    def get_instruction(self) -> str:
        """Return task-specific instruction for function completion."""
        return (
            "Complete the Python function below by writing the function body.\n"
            "Do not repeat the function signature or docstring — provide only the implementation.\n"
            "Your code will be tested against a hidden test suite."
        )

    def get_system_prompt(self) -> str:
        """Return system prompt for HumanEval evaluation."""
        return (
            "You are an expert Python programmer. "
            "Complete the given Python function by writing a correct implementation. "
            "Write clean, idiomatic Python that satisfies all constraints described in the docstring. "
            "Output only the function body code — do not include the function signature, "
            "docstring, or any explanation."
        )

    def get_demonstrations(self, n_shot: int = 1) -> str:
        """Hand-crafted HumanEval demonstration for RoT warm-up.

        The ground truth holds the hidden test suite, not a reference body, so
        the demonstration is one canonical worked example: a signature with a
        docstring as input and the function body (only) as output, matching
        the task's required answer style.
        """
        return (
            "Input:\n"
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """ Check if in a given list of numbers, any two are closer to '
            'each other than the given threshold. """\n'
            "Output:\n"
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False"
        )
