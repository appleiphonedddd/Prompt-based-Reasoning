"""Programming Puzzles Benchmark Dataset.

Based on Microsoft's PythonProgrammingPuzzles repository:
https://github.com/microsoft/PythonProgrammingPuzzles

Each puzzle defines a ``sat`` function — a Python predicate that accepts
a candidate answer and returns ``True`` if and only if it is a valid solution.
Models are asked to produce a Python literal of the specified type that
satisfies the predicate.

Evaluation:
    The model's text output is parsed into a Python literal via
    ``ast.literal_eval``, then passed to the ``sat`` function.
    A puzzle is scored 1.0 (correct) if ``sat(answer)`` returns ``True``
    within the time limit (5 s).

Dataset:
    1715 puzzles across 18 modules (study, basic, algebra, number_theory,
    classic_puzzles, codeforces, human_eval, IMO, ICPC, …).
    Source: ``benchmark/ProgrammingPuzzles/puzzles.json``.

Author: Egor Morozov
"""

import ast
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_SAT_TIMEOUT: float = 5.0  # wall-clock seconds before sat() is considered TLE


# ─────────────────────────────────────────────────────────────────────────────
# Helper: typing namespace for exec'ing sat functions
# ─────────────────────────────────────────────────────────────────────────────

def _build_sat_namespace() -> Dict[str, Any]:
    """Return an exec namespace containing the typing aliases used by sat functions.

    The puzzle ``sat`` functions use PEP 484 generic annotations such as
    ``List[int]`` or ``Dict[str, int]``.  Python 3.11 evaluates annotations
    eagerly at function-definition time, so these names must be present in
    the namespace passed to ``exec``.

    Returns:
        A dict suitable for the ``globals`` argument of ``exec``.
    """
    from typing import (
        Any as _Any, Callable, Counter, DefaultDict, Dict as _Dict,
        FrozenSet, Iterator, List as _List, Optional as _Optional,
        Sequence, Set as _Set, Tuple, Union,
    )
    return {
        "__builtins__": __builtins__,
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
        "DefaultDict": DefaultDict,
        "Counter": Counter,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: timeout-guarded sat invocation
# ─────────────────────────────────────────────────────────────────────────────

def _call_sat_with_timeout(
    sat_fn: Any,
    answer: Any,
    timeout: float = _SAT_TIMEOUT,
) -> tuple:
    """Call ``sat_fn(answer)`` with a wall-clock timeout.

    Uses a daemon thread so a runaway sat function does not block forever.

    Args:
        sat_fn:  The compiled ``sat`` predicate callable.
        answer:  Parsed candidate answer to evaluate.
        timeout: Maximum seconds to wait (default: ``_SAT_TIMEOUT``).

    Returns:
        ``(result, error)`` where *result* is the bool returned by sat (or
        ``None`` on timeout/exception) and *error* is the raised exception
        (or ``None`` on success).
    """
    container: List[Any] = [None, None]  # [result, error]

    def _target() -> None:
        try:
            container[0] = sat_fn(answer)
        except Exception as exc:  # noqa: BLE001
            container[1] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None, TimeoutError(f"sat() timed out after {timeout}s")
    return container[0], container[1]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Python literal extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_python_literal(text: str, ans_type: str = "") -> Optional[str]:
    """Extract the first valid Python literal string from model output.

    Tries several extraction strategies in priority order, validating each
    candidate with ``ast.literal_eval``.  Returns the raw string of the
    first candidate that parses successfully, or ``None`` if none do.

    Args:
        text:     Raw model output string.
        ans_type: Expected type annotation (e.g. ``"List[int]"``).
                  Used as a hint for type-specific pattern matching.

    Returns:
        A raw string suitable for ``ast.literal_eval``, or ``None``.
    """
    text = text.strip()
    candidates: List[str] = []

    # 1. Whole text (model may output the literal directly)
    candidates.append(text)

    # 2. Inside fenced code blocks  ```python ... ``` or ``` ... ```
    for m in re.finditer(r"```(?:python)?\s*\n?(.*?)\n?```", text, re.DOTALL):
        block = m.group(1).strip()
        ret = re.search(r"^\s*return\s+(.+)$", block, re.MULTILINE)
        if ret:
            candidates.append(ret.group(1).strip())
        candidates.append(block)

    # 3. Inline backtick spans
    for m in re.finditer(r"`([^`\n]+)`", text):
        candidates.append(m.group(1).strip())

    # 4. After explicit answer / return markers
    for pattern in (
        r"(?:^|\n)\s*(?:answer|result|solution)\s*[:=]\s*(.+?)(?:\n|$)",
        r"(?:^|\n)\s*return\s+(.+?)(?:\n|$)",
        r"(?:answer is|answer:|output is|output:)\s+(.+?)(?:\n|$)",
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidates.append(m.group(1).strip())

    # 5. Type-specific heuristics
    base_type = ans_type.split("[")[0].strip() if ans_type else ""
    if base_type == "str":
        m = re.search(r'(["\'])(?:\\.|(?!\1).)*\1', text)
        if m:
            candidates.append(m.group(0))
    elif base_type in ("int", "float"):
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        if m:
            candidates.append(m.group(0))
    elif base_type == "bool":
        m = re.search(r"\b(True|False)\b", text)
        if m:
            candidates.append(m.group(0))
    elif base_type == "List":
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            candidates.append(m.group(0))

    # 6. Last non-empty line as final fallback
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        candidates.append(lines[-1].rstrip(".,;"))

    # Validate with ast.literal_eval; return first that succeeds
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            ast.literal_eval(candidate)
            return candidate
        except (ValueError, SyntaxError):
            continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: docstring cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _clean_docstring(docstring: str) -> str:
    """Strip indentation and triple-quote delimiters from a sol_docstring.

    Args:
        docstring: Raw ``sol_docstring`` value from ``puzzles.json``.

    Returns:
        Clean human-readable description string.
    """
    s = docstring.strip()
    for quote in ('"""', "'''"):
        if s.startswith(quote):
            s = s[len(quote):]
            if s.endswith(quote):
                s = s[: -len(quote)]
            break
    return s.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset implementation
# ─────────────────────────────────────────────────────────────────────────────

class ProgrammingPuzzles(DatasetBase):
    """Benchmark for evaluating LLMs on Python programming puzzle solving.

    Each puzzle provides a ``sat`` function — a Python predicate that returns
    ``True`` if and only if the supplied value is a valid solution.  Models
    must produce a Python literal of the specified type that satisfies the
    predicate.

    Puzzles span 18 difficulty categories, from trivial string / number tasks
    (``study.py``, ``basic.py``) to competition problems (``ICPC.py``,
    ``IMO.py``) and HumanEval-style coding problems (``human_eval.py``).

    Args:
        split:       Unused; kept for interface consistency (default ``"test"``).
        num_samples: Maximum number of puzzles to load.  ``None`` loads all
                     1715 puzzles.
        module:      If provided, restrict to puzzles whose ``module`` field
                     matches exactly (e.g. ``"study.py"``, ``"IMO.py"``).

    Example::

        ds = ProgrammingPuzzles(num_samples=20, module="basic.py")
        ds.load_dataset()
        print(len(ds))          # ≤ 20

        problem = ds.get_problem(0)
        result  = ds.evaluate_answer('"' + 'ho' * 1000 + '"', problem.ground_truth)
        print(result.is_correct)
    """

    def __init__(
        self,
        split: str = "test",
        num_samples: Optional[int] = None,
        module: Optional[str] = None,
    ) -> None:
        """Initialise the ProgrammingPuzzles benchmark.

        Args:
            split:       Dataset split (kept for interface consistency).
            num_samples: Maximum puzzles to load; ``None`` means all.
            module:      Filter by puzzle module filename
                         (e.g. ``"study.py"``, ``"basic.py"``).
        """
        super().__init__(split=split, dataset_name="ProgrammingPuzzles")
        self.num_samples = num_samples
        self.module = module

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Load puzzles from the local ``puzzles.json`` file.

        Populates ``self._data`` with a filtered, optionally truncated list
        of puzzle dicts.  Each dict contains at minimum the keys ``name``,
        ``sat``, ``ans_type``, ``sol_docstring``, ``module``, and ``weight``.

        Raises:
            RuntimeError: If ``puzzles.json`` is missing or cannot be parsed.
        """
        import json

        data_file = Path(__file__).parent / "puzzles.json"
        if not data_file.exists():
            raise RuntimeError(
                f"[{self.dataset_name}] Data file not found: {data_file}\n"
                "Expected puzzles.json in benchmark/ProgrammingPuzzles/."
            )

        try:
            with open(data_file, "r", encoding="utf-8") as f:
                raw: List[Dict[str, Any]] = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"[{self.dataset_name}] Failed to load '{data_file}': {exc}"
            ) from exc

        if self.module:
            raw = [p for p in raw if p.get("module") == self.module]
        if self.num_samples is not None:
            raw = raw[: self.num_samples]

        self._data = raw

        suffix_parts = []
        if self.module:
            suffix_parts.append(f"module='{self.module}'")
        if self.num_samples is not None:
            suffix_parts.append(f"num_samples={self.num_samples}")
        suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""

        print(f"[{self.dataset_name}] Loaded {len(self._data)} puzzles{suffix}.")

    def get_problem(self, index: int) -> Problem:
        """Return the puzzle at the given index.

        The ``Problem.question`` contains a natural-language task description
        (from ``sol_docstring``) together with the ``sat`` function so the
        model can see the exact constraints it must satisfy.

        The ``Problem.ground_truth`` is a dict with keys:

        - ``sat``      — source code of the ``sat`` function (``str``)
        - ``ans_type`` — expected return type (e.g. ``"List[int]"``)
        - ``name``     — puzzle identifier

        Args:
            index: Zero-based index into the loaded puzzle list.

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
        sat_src: str = row["sat"]
        ans_type: str = row.get("ans_type", "")
        description: str = _clean_docstring(row.get("sol_docstring", "")) or "(no description)"

        question = (
            f"Task: {description}\n\n"
            f"Return type: {ans_type}\n\n"
            "Your answer must make the following Python function return True:\n\n"
            f"{sat_src}\n\n"
            f"Provide only the answer as a Python literal of type {ans_type}."
        )

        return Problem(
            index=index,
            question=question,
            ground_truth={"sat": sat_src, "ans_type": ans_type, "name": row["name"]},
            metadata={
                "name": row["name"],
                "module": row.get("module", ""),
                "ans_type": ans_type,
                "weight": row.get("weight", 1.0),
            },
        )

    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Evaluate whether the model's prediction satisfies the sat predicate.

        Evaluation pipeline:

        1. Extract a Python literal candidate from ``prediction`` text.
        2. Parse it safely with ``ast.literal_eval``.
        3. Compile and ``exec`` the ``sat`` function in a restricted namespace.
        4. Call ``sat(parsed_answer)`` with a 5-second timeout.

        A puzzle is correct if and only if ``sat`` returns ``True``.

        Args:
            prediction:   The model's raw text output (``BaselineResponse.final_answer``).
            ground_truth: Dict with keys ``sat``, ``ans_type``, and ``name``.

        Returns:
            ``EvaluationResult`` with ``is_correct``, ``score``, and
            diagnostic ``details``.
        """
        details: Dict[str, Any] = {"raw_prediction": prediction}

        if not isinstance(ground_truth, dict) or "sat" not in ground_truth:
            details["error"] = "ground_truth must be a dict with a 'sat' key."
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                prediction=prediction,
                ground_truth=ground_truth,
                details=details,
            )

        sat_src: str = ground_truth["sat"]
        ans_type: str = ground_truth.get("ans_type", "")
        details["ans_type"] = ans_type

        # ── Step 1: extract literal candidate ─────────────────────────────
        literal_str = _extract_python_literal(prediction, ans_type)
        if literal_str is None:
            details["error"] = "Could not extract a valid Python literal from the prediction."
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                prediction=prediction,
                ground_truth=ground_truth,
                details=details,
            )
        details["extracted_literal"] = literal_str

        # ── Step 2: parse safely ───────────────────────────────────────────
        try:
            parsed_answer = ast.literal_eval(literal_str)
        except (ValueError, SyntaxError) as exc:
            details["error"] = f"ast.literal_eval failed: {exc}"
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                prediction=prediction,
                ground_truth=ground_truth,
                details=details,
            )
        details["parsed_value_repr"] = repr(parsed_answer)[:200]

        # ── Step 3: compile sat function ───────────────────────────────────
        namespace = _build_sat_namespace()
        try:
            exec(sat_src, namespace)  # noqa: S102
        except Exception as exc:
            details["error"] = f"Failed to compile sat function: {exc}"
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                prediction=prediction,
                ground_truth=ground_truth,
                details=details,
            )

        sat_fn = namespace.get("sat")
        if not callable(sat_fn):
            details["error"] = "sat function not found in namespace after exec."
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                prediction=prediction,
                ground_truth=ground_truth,
                details=details,
            )

        # ── Step 4: call sat with timeout ──────────────────────────────────
        result, error = _call_sat_with_timeout(sat_fn, parsed_answer)

        if error is not None:
            details["error"] = f"sat() raised: {error}"
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                prediction=prediction,
                ground_truth=ground_truth,
                details=details,
            )

        is_correct = bool(result)
        details["sat_result"] = result
        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            prediction=prediction,
            ground_truth=ground_truth,
            details=details,
        )

    # ── Optional hook overrides ────────────────────────────────────────────

    def get_instruction(self) -> str:
        """Return a task-specific instruction for programming puzzle solving."""
        return (
            "Solve the following Python programming puzzle.\n"
            "You are given a `sat` function — a Python predicate that returns True\n"
            "if and only if the input is a correct solution.\n"
            "Output ONLY a valid Python literal (e.g., 42, \"hello\", [1, 2, 3], True).\n"
            "Do not include explanations, function definitions, or any other text."
        )

    def get_system_prompt(self) -> str:
        """Return a system prompt for programming puzzle evaluation."""
        return (
            "You are an expert Python programmer solving programming puzzles. "
            "Each puzzle provides a `sat` function: a Python predicate that returns "
            "True if and only if the input is a correct solution. "
            "Your goal is to find a value that makes `sat` return True. "
            "Respond with only the Python literal answer — no explanations, "
            "no code blocks, no additional text."
        )

    def get_demonstrations(self, n_shot: int = 1) -> str:
        """Hand-crafted Programming Puzzles demonstration for RoT warm-up.

        The ground truth holds the ``sat`` predicate and answer type rather
        than a literal solution, so the demonstration is one canonical worked
        example: a ``sat`` function as input and the Python literal that
        satisfies it as output, matching the task's required answer style.
        """
        return (
            "Input:\n"
            "def sat(x: int):\n"
            "    return x + 1 == 42\n"
            "Output: 41"
        )
