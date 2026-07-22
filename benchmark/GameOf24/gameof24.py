"""
Game of 24 Benchmark Dataset.

The "Game of 24" is a classic arithmetic puzzle:
given four numbers, combine them using +, -, *, / and parentheses
to produce exactly 24.  Every number must be used exactly once.

HuggingFace source:
    dataset: "nlile/24-game"   (https://huggingface.co/datasets/nlile/24-game)
    split:   "train" (the dataset ships one split that contains all puzzles)

Evaluation strategy:
    1. Try to parse the prediction as a Python arithmetic expression.
    2. Evaluate it safely — result must equal 24 (within float tolerance).
    3. Verify that each of the four puzzle numbers is used exactly once
       in the expression.

Author: Egor Morozov
"""

import re
import ast
import operator
from typing import Any

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


# Allowed AST node types for the safe evaluator
_SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.USub,
    ast.UAdd,
)

_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    """Evaluate a pure arithmetic expression without using eval().

    Raises:
        ValueError: If the expression contains disallowed constructs.
        ZeroDivisionError: If division by zero occurs.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {expr!r}") from exc

    def _lookup(op_node):
        # The _SAFE_NODES check below only sees the BinOp/UnaryOp wrapper, not
        # the operator it carries, so an unsupported operator (%, **, //, ...)
        # reaches here.  Report it as a ValueError like any other malformed
        # expression instead of letting a KeyError escape evaluate_answer().
        op = _OPS.get(type(op_node))
        if op is None:
            raise ValueError(f"Disallowed operator: {type(op_node).__name__}")
        return op

    def _visit(node):
        if not isinstance(node, _SAFE_NODES):
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Expression):
            return _visit(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            left  = _visit(node.left)
            right = _visit(node.right)
            op    = _lookup(node.op)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("Division by zero in expression.")
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            return _lookup(node.op)(_visit(node.operand))
        raise ValueError(f"Unexpected node: {node}")

    return _visit(tree)


def _extract_numbers_from_expr(expr: str) -> list[int]:
    """Return all integer literals found in an expression string."""
    return [int(m) for m in re.findall(r"\d+", expr)]


class GameOf24(DatasetBase):
    """Benchmark wrapper for the Game of 24 arithmetic puzzle dataset.

    Each problem provides four numbers; the model must produce an
    arithmetic expression that equals exactly 24.

    Args:
        split: HuggingFace dataset split to load (default: ``"train"``).
               The nlile/24-game dataset only has a "train" split.

    Example::

        ds = GameOf24()
        ds.load_dataset()
        print(len(ds))                          # e.g. 1362

        problem = ds.get_problem(0)
        print(problem.question)                 # "1 1 1 1"

        result = ds.evaluate_answer("(1+1+1)*1+21", problem.ground_truth)
        print(result.is_correct, result.score)
    """

    HF_DATASET_ID = "nlile/24-game"

    def __init__(self, split: str = "train"):
        super().__init__(split=split, dataset_name="GameOf24")

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Download the Game of 24 dataset from Hugging Face.

        Populates ``self._data`` with the HuggingFace Dataset object.

        Raises:
            ImportError:  If the ``datasets`` package is not installed.
            RuntimeError: If the download or validation fails.
        """
        try:
            from datasets import load_dataset as hf_load
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. "
                "Install it with: pip install datasets"
            ) from exc

        try:
            raw = hf_load(self.HF_DATASET_ID, split=self.split)
        except Exception as exc:
            raise RuntimeError(
                f"[{self.dataset_name}] Failed to load '{self.HF_DATASET_ID}' "
                f"(split='{self.split}'): {exc}"
            ) from exc

        self._data = raw
        print(
            f"[{self.dataset_name}] Loaded {len(self._data)} problems "
            f"from '{self.HF_DATASET_ID}' (split='{self.split}')."
        )

    def get_problem(self, index: int) -> Problem:
        """Return the Game of 24 problem at the given index.

        Args:
            index: Zero-based index into the dataset split.

        Returns:
            Problem whose ``question`` is the four numbers (e.g. "2 5 8 11")
            and ``ground_truth`` is the same four numbers as a sorted list
            of ints.

        Raises:
            RuntimeError: If the dataset has not been loaded yet.
            IndexError:   If ``index`` is out of range.
        """
        self._ensure_loaded()

        if index < 0 or index >= len(self._data):
            raise IndexError(
                f"[{self.dataset_name}] Index {index} out of range "
                f"[0, {len(self._data) - 1}]."
            )

        row = self._data[index]

        # The nlile/24-game dataset has a "numbers" column (list of ints)
        numbers = row.get("numbers", [])
        puzzle_str = " ".join(str(n) for n in numbers)

        return Problem(
            index=index,
            question=puzzle_str.strip(),
            ground_truth=sorted(numbers),
            metadata={
                "numbers": numbers,
                "raw_row": dict(row),
            },
        )

    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Evaluate whether the predicted expression equals 24.

        The prediction is correct if:
        1. It is a valid arithmetic expression that evaluates to 24
           (within 1e-6 tolerance), AND
        2. The integers used in the expression match the puzzle numbers
           exactly (same multiset).

        Args:
            prediction:   The model's raw answer string (e.g. "(2+5)*8/11").
            ground_truth: The sorted list of ints from ``get_problem()``.

        Returns:
            EvaluationResult with ``is_correct`` and ``score`` in {0.0, 1.0}.
        """
        # Strip markdown code blocks and surrounding whitespace
        clean = re.sub(r"```[a-z]*", "", prediction).strip().strip("`").strip()

        # Convert LaTeX math symbols and delimiters
        # Handle \(...\), \[...\], and $...$ → content
        clean = re.sub(r"\\\(", "(", clean)
        clean = re.sub(r"\\\)", ")", clean)
        clean = re.sub(r"\\\[", "", clean)
        clean = re.sub(r"\\\]", "", clean)
        clean = re.sub(r"\$", "", clean)

        # Replace LaTeX commands with ASCII equivalents
        # \frac{a}{b} → (a/b)
        clean = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"(\1/\2)", clean)
        # \times, \cdot → *
        clean = re.sub(r"\\(times|cdot)", "*", clean)
        # \div → /
        clean = re.sub(r"\\div", "/", clean)
        # \left and \right → nothing (just remove them)
        clean = re.sub(r"\\left\s*", "", clean)
        clean = re.sub(r"\\right\s*", "", clean)

        # Try to isolate the arithmetic expression.
        # Models often write "The answer is (2+5)*8/11 = 24" or include
        # natural-language text before the expression.  Rather than taking
        # the first regex match (which tends to be a whitespace fragment),
        # collect ALL candidate segments and try each one from last to first,
        # preferring the one that evaluates to 24 with the right numbers.
        candidates = re.findall(r"[\d\s()+\-*/%.]+", clean)
        # Try candidates from last to first, preferring segments that contain
        # arithmetic operators over bare numbers.  This avoids picking "24"
        # when the model appends "= 24" to an otherwise correct expression.
        expr = clean  # fallback
        fallback_number: str | None = None
        for candidate in reversed(candidates):
            stripped = candidate.strip()
            if not stripped:
                continue
            if re.search(r"[+\-*/()]", stripped):
                expr = stripped
                break
            if fallback_number is None:
                fallback_number = stripped
        else:
            if expr == clean and fallback_number is not None:
                expr = fallback_number

        details: dict = {"raw_prediction": prediction, "parsed_expression": expr}

        try:
            result = _safe_eval(expr)
            details["evaluated_result"] = result
        except (ValueError, ZeroDivisionError, TypeError) as exc:
            details["error"] = str(exc)
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                prediction=prediction,
                ground_truth=ground_truth,
                details=details,
            )

        # Check numeric result
        reaches_24 = abs(result - 24.0) < 1e-6
        details["reaches_24"] = reaches_24

        # Check number usage (same multiset as puzzle numbers)
        used_numbers = sorted(_extract_numbers_from_expr(expr))
        details["used_numbers"]   = used_numbers
        details["puzzle_numbers"] = ground_truth
        numbers_match = used_numbers == list(ground_truth)
        details["numbers_match"] = numbers_match

        is_correct = reaches_24 and numbers_match

        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            prediction=prediction,
            ground_truth=ground_truth,
            details=details,
        )

    # ── Optional hook overrides ────────────────────────────────────────────

    def get_instruction(self) -> str:
        return (
            "Use the four given numbers and the operations +, -, *, / "
            "with parentheses to form an expression that equals exactly 24. "
            "Each number must be used exactly once."
        )

    def get_system_prompt(self) -> str:
        return (
            "You are an expert mathematical puzzle solver. "
            "Respond with a single arithmetic expression only."
        )

    def get_demonstrations(self, n_shot: int = 2) -> str:
        """Hand-crafted Game-of-24 demonstrations for RoT warm-up.

        The dataset ground truth holds only the four numbers, not a solution
        expression, so demonstrations are canonical worked examples (the
        output is the arithmetic expression the task expects), matching the
        official RoT release and the paper's Game-of-24 case study.
        """
        demos = [
            "Input: 1 5 5 5\nOutput: 5 * (5 - 1 / 5) = 24",
            "Input: 3 3 8 8\nOutput: 8 / (3 - 8 / 3) = 24",
        ]
        return "\n\n".join(demos[: max(1, n_shot)])
