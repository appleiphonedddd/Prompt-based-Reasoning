"""CRUXEval Benchmark: Code Reasoning, Understanding, and Execution Evaluation.

Each problem presents a Python function (always named ``f``) and a string
representation of its input arguments.  The model must predict the exact
Python literal that ``f(*args)`` returns.

Evaluation compares Python values via ``ast.literal_eval`` when possible,
falling back to normalised string comparison when either side cannot be parsed.

Dataset: 799 examples loaded from ``benchmark/CRUXEval/data/cruxeval.jsonl``

Each JSONL record contains:
    code:   Python function definition (always ``def f(...):``)
    input:  String representation of arguments passed to ``f``
    output: String representation of the expected return value
    id:     Unique identifier (e.g. ``"sample_0"``)

Reference:
    Gu et al., "CRUXEval: A Benchmark for Code Reasoning, Understanding
    and Execution" (https://crux-eval.github.io)

Author: Egor Morozov
"""

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: Python literal extraction and comparison
# ─────────────────────────────────────────────────────────────────────────────

def _eval_literal(text: str) -> tuple[Any, bool]:
    """Try to parse *text* as a Python literal with ``ast.literal_eval``.

    Args:
        text: String to evaluate.

    Returns:
        ``(value, True)`` on success; ``(None, False)`` on any error.
    """
    try:
        return ast.literal_eval(text.strip()), True
    except Exception:
        return None, False


def _extract_literal_from_line(line: str) -> Optional[str]:
    """Find the first valid Python literal within *line*.

    Two-pass strategy:
    1. Scan for unambiguous literal-start characters (``[``, ``{``, ``(``,
       ``'``, ``"``, digits, ``-digit``).  For the first match, try
       progressively shorter suffixes until one evaluates successfully.
    2. Look for the words ``True``, ``False``, ``None`` as whole words
       (``T``/``F``/``N`` are skipped in pass 1 to avoid matching "The",
       "For", "No" etc.).

    Args:
        line: A single line of model output.

    Returns:
        The extracted literal string, or ``None`` if no literal was found.
    """
    line = line.strip()
    if not line:
        return None

    val, ok = _eval_literal(line)
    if ok:
        return line

    # Pass 1: unambiguous compound-literal starters
    compound_starts = set('[{(\'"0123456789')
    for i, ch in enumerate(line):
        if ch in compound_starts:
            suffix = line[i:]
            for end in range(len(suffix), 0, -1):
                candidate = suffix[:end]
                val, ok = _eval_literal(candidate)
                if ok:
                    return candidate
            break
        # '-' only counts when followed by a digit (negative number)
        if ch == '-' and i + 1 < len(line) and line[i + 1].isdigit():
            suffix = line[i:]
            for end in range(len(suffix), 0, -1):
                candidate = suffix[:end]
                val, ok = _eval_literal(candidate)
                if ok:
                    return candidate
            break

    # Pass 2: singleton word literals True / False / None
    for word in ('True', 'False', 'None'):
        if re.search(rf'\b{word}\b', line):
            return word

    return None


def _extract_python_literal(text: str) -> str:
    """Extract the most likely Python literal value from *text*.

    Applied in order — returns on the first successful extraction:

    1. Direct ``ast.literal_eval`` of the full stripped text.
    2. Content of the first Markdown fenced code block (`` ``` … ``` ``).
    3. Capture group of "answer / output / result is X" patterns
       (handles zerocot's ``"Therefore, the answer is …"`` format and
       standard-baseline free-form responses).
    4. First line that is itself a valid Python literal.
    5. First Python literal found within any line (character-level scan).
    6. Fallback: last non-empty line (preserves the model's stated intent
       even if it cannot be parsed — the caller handles gracefully).

    Args:
        text: Raw model response (``BaselineResponse.final_answer``).

    Returns:
        The extracted literal string.  May still be unparseable Python if
        all strategies fail — ``evaluate_answer`` handles that case.
    """
    text = text.strip()

    # 1. Direct eval
    val, ok = _eval_literal(text)
    if ok:
        return text

    # 2. Markdown code block
    code_block = re.search(r"```(?:python)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block:
        content = code_block.group(1).strip()
        val, ok = _eval_literal(content)
        if ok:
            return content
        found = _extract_literal_from_line(content)
        if found is not None:
            return found

    # 3. "answer / output / result is X" patterns — check ALL matches, last wins
    #    (`\s*` before the separator handles both "answer is X" and "Answer: X")
    _value_re = re.compile(
        r"(?:the\s+)?(?:output|answer|result|return(?:s|ed)?(?:\s+value)?|value)"
        r"\s*(?:is|=|:)\s*(.+)$",
        re.IGNORECASE | re.MULTILINE,
    )
    _therefore_re = re.compile(
        r"therefore,?\s+the\s+answer\s+is\s*:?\s*(.+)$",
        re.IGNORECASE | re.MULTILINE,
    )
    for pattern in (_value_re, _therefore_re):
        for m in pattern.finditer(text):
            candidate = m.group(1).strip().rstrip(".")
            val, ok = _eval_literal(candidate)
            if ok:
                return candidate
            found = _extract_literal_from_line(candidate)
            if found is not None:
                return found

    # 4 & 5. Line-by-line scan — reversed so the answer (usually last) is
    #         tried before intermediate reasoning steps that may contain stray
    #         numbers (e.g. "Step 1: …" would otherwise match digit literals).
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for line in reversed(lines):
        val, ok = _eval_literal(line)
        if ok:
            return line
        found = _extract_literal_from_line(line)
        if found is not None:
            return found

    # 6. Fallback: last non-empty line
    return lines[-1] if lines else text


# ─────────────────────────────────────────────────────────────────────────────
# Dataset implementation
# ─────────────────────────────────────────────────────────────────────────────

class CRUXEval(DatasetBase):
    """CRUXEval benchmark: predict the return value of a Python function.

    Each problem gives the model the source of a function ``f`` and the
    string representation of its arguments.  The model must output the exact
    Python literal that ``f(*args)`` evaluates to.

    Evaluation logic (``evaluate_answer``):
        1. Extract a Python literal from the raw prediction using
           ``_extract_python_literal`` — handles CoT reasoning traces,
           "The answer is …" prefixes, Markdown code blocks, and plain
           literal responses from all seven baselines.
        2. Compare extracted answer and reference via ``ast.literal_eval``
           (Python-value equality).
        3. Fall back to normalised string comparison if either side cannot
           be parsed as a Python literal.

    Args:
        split: Unused; kept for interface consistency (default ``"test"``).

    Example::

        ds = CRUXEval()
        ds.load_dataset()
        problem = ds.get_problem(0)
        result = ds.evaluate_answer(
            "[(4, 1), (4, 1), (4, 1), (4, 1), (2, 3), (2, 3)]",
            problem.ground_truth,
        )
        print(result.is_correct)   # True
    """

    def __init__(self, split: str = "test") -> None:
        super().__init__(split=split, dataset_name="CRUXEval")

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Load CRUXEval problems from the local JSONL file.

        Populates ``self._data`` with a list of problem dicts, each containing
        ``code``, ``input``, ``output``, and ``id``.

        Raises:
            RuntimeError: If ``cruxeval.jsonl`` is missing or malformed.
        """
        data_file = Path(__file__).parent / "data" / "cruxeval.jsonl"
        if not data_file.exists():
            raise RuntimeError(
                f"[{self.dataset_name}] Data file not found: {data_file}\n"
                "Expected cruxeval.jsonl in benchmark/CRUXEval/data/."
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
        """Return the CRUXEval problem at position *index*.

        ``Problem.question`` shows the function source and the call expression,
        asking the model to predict the return value.

        ``Problem.ground_truth`` is a dict with keys:

        - ``expected_output`` — expected return value as a Python literal string
        - ``code``            — function source (for diagnostic details)
        - ``input``           — argument string (for diagnostic details)
        - ``id``              — sample identifier (e.g. ``"sample_0"``)

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
        code: str = row["code"]
        inp: str = row["input"]
        expected_output: str = row["output"]
        sample_id: str = row["id"]

        question = (
            f"```python\n{code}\n```\n\n"
            f"What does `f({inp})` return?"
        )

        return Problem(
            index=index,
            question=question,
            ground_truth={
                "expected_output": expected_output,
                "code": code,
                "input": inp,
                "id": sample_id,
            },
            metadata={
                "id": sample_id,
                "code": code,
                "input": inp,
            },
        )

    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Score the model's predicted return value against the reference.

        Pipeline:

        1. Extract a Python literal from *prediction* via
           ``_extract_python_literal`` (strips reasoning traces, "The answer
           is …" prefixes, Markdown fences, etc. — works for all baselines).
        2. Evaluate both extracted answer and reference with
           ``ast.literal_eval``; compare Python objects for equality.
        3. If either side fails to parse, fall back to stripped string
           comparison.

        Args:
            prediction:   The model's raw text output
                          (``BaselineResponse.final_answer``).
            ground_truth: Dict with key ``expected_output`` (Python literal
                          string) and optional ``code``, ``input``, ``id``.

        Returns:
            ``EvaluationResult`` with ``is_correct``, ``score`` ∈ {0.0, 1.0},
            and diagnostic ``details``.
        """
        details: Dict[str, Any] = {"raw_prediction": prediction}

        if not isinstance(ground_truth, dict) or "expected_output" not in ground_truth:
            details["error"] = "ground_truth must be a dict with 'expected_output' key."
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        expected_str: str = ground_truth["expected_output"]
        details["expected_output"] = expected_str

        # ── Step 1: extract literal from model's response ─────────────────
        extracted = _extract_python_literal(prediction)
        details["extracted_answer"] = extracted

        # ── Step 2: compare as Python values ─────────────────────────────
        pred_val, pred_ok = _eval_literal(extracted)
        gt_val, gt_ok = _eval_literal(expected_str)

        if pred_ok and gt_ok:
            is_correct = pred_val == gt_val
            details["comparison_method"] = "python_value"
            details["predicted_value"] = repr(pred_val)
            details["expected_value"] = repr(gt_val)
        else:
            # ── Step 3: normalised string fallback ────────────────────────
            is_correct = extracted.strip() == expected_str.strip()
            details["comparison_method"] = "string"
            details["pred_eval_ok"] = pred_ok
            details["gt_eval_ok"] = gt_ok

        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            prediction=prediction,
            ground_truth=ground_truth,
            details=details,
        )

    # ── Optional hook overrides ────────────────────────────────────────────

    def get_instruction(self) -> str:
        """Return task-specific instruction for output prediction."""
        return (
            "You are given a Python function and a function call.\n"
            "Predict the exact return value.\n"
            "Write ONLY the Python literal as your final answer — "
            "for example: 42, 'hello', [1, 2, 3], {'a': 1}, True, None.\n"
            "Do not include any explanation or surrounding text."
        )

    def get_system_prompt(self) -> str:
        """Return system prompt for CRUXEval evaluation."""
        return (
            "You are an expert Python interpreter. "
            "Given a Python function and its input, predict the exact return value. "
            "Your answer must be a valid Python literal — no explanation, no code, "
            "just the value."
        )

    def _demo_output(self, problem) -> Optional[str]:
        """Render the expected return value for an RoT demonstration.

        CRUXEval ground truth is a dict; the demonstration output is the
        ``expected_output`` literal. The default :meth:`get_demonstrations`
        derivation (Input = question, Output = this value) then applies.
        """
        gt = problem.ground_truth
        if isinstance(gt, dict) and gt.get("expected_output") is not None:
            return str(gt["expected_output"])
        return None

