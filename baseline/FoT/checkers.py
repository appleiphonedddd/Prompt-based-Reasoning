"""
Trusted external checkers c_q for Falsification-of-Thought (FoT).

In the executable regime of FoT (Algorithm 2), the model only *proposes* where to
probe (pi_probe); the verdict is supplied by an EXTERNAL, task-specific, trusted
checker c_q(a, w_hat) — never by the model. This is exactly what makes an
executable witness *sound*: c_q never reports a false failure, so a correct
candidate is never discarded on a spurious witness.

This module provides those trusted checkers for the benchmarks whose candidate
answers admit a cheap, decisive check (the paper's "cheap-to-falsify" task
profile):

  * ``gameof24``           — arithmetic evaluation: does the expression equal 24
                             using each of the four puzzle numbers exactly once?
  * ``cruxeval``           — program execution: run the reference function on the
                             given input and compare with the predicted literal.
  * ``programmingpuzzles`` — predicate execution: does ``sat(answer)`` return True?
  * ``bigbenchhard:multistep_arithmetic_two``
                           — arithmetic evaluation: recompute the expression the
                             question states and compare with the answer.

Each checker is deterministic and derives its verdict ONLY from the problem
statement (which carries the reference function / puzzle numbers / sat predicate)
and the candidate answer — the model is not involved in the decision. Tasks not
registered here have ``HasChecker(q) = False`` and fall to the relational regime.

The mapping from benchmark to checker realises the paper's ``HasChecker(q)``
predicate: it is FIXED PER BENCHMARK (keyed by the benchmark name), not decided
at run time by the model.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

# Reuse BoT's hardened, timeout-bounded subprocess executor to run the *reference*
# code embedded in the problem (CRUXEval functions, Python-Puzzles ``sat``). The
# code that runs is the benchmark's own, not model-authored verdict logic, so the
# verdict it yields is sound.
from baseline.BoT.bot import run_code


@dataclass
class CheckResult:
    """Outcome of a trusted checker c_q.

    Attributes:
        verdict: "fail"  — the candidate is provably wrong (a SOUND witness),
                 "pass"  — the candidate passes this decisive check (survives),
                 "undecided" — the checker could not decide (treated as survival,
                               so a correct candidate is never discarded).
        detail: a concrete, self-contained description of the failure (the witness
                detail ``d``) when ``verdict == "fail"``, else a diagnostic note.
    """

    verdict: str
    detail: str


# A checker maps (question, candidate_answer, probe) -> CheckResult. ``probe`` is
# the model's pi_probe suggestion ("where to look"); decisive checkers compute
# their canonical verdict regardless and use it only for diagnostics.
Checker = Callable[..., CheckResult]


# ── Game of 24 ──────────────────────────────────────────────────────────────────

# Only the four arithmetic operations (and unary sign) are legal in Game of 24.
_ARITH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_arith_eval(expr: str) -> float:
    """Evaluate a pure +,-,*,/ arithmetic expression via AST (no ``eval``).

    Raises ValueError/ZeroDivisionError on anything that is not a legal Game-of-24
    arithmetic expression, so a non-evaluable candidate is treated as failing.
    """

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("non-numeric constant")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
            return _ARITH_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_OPS:
            return _ARITH_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"disallowed expression node: {type(node).__name__}")

    return _eval(ast.parse(expr.strip(), mode="eval"))


def _clean_g24(text: str) -> str:
    """Strip code fences / LaTeX / unicode operators from a Game-of-24 answer."""
    s = re.sub(r"```[a-zA-Z]*", "", text)
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-")
    s = re.sub(r"\\(times|cdot)", "*", s)
    s = re.sub(r"\\div", "/", s)
    s = s.replace("$", "").replace("`", "")
    return s


def gameof24_checker(
    question: str, candidate: str, probe: str = "", *, timeout: float = 10.0
) -> CheckResult:
    """c_q for Game of 24: the expression must equal 24 using each number once.

    Sound: returns "fail" ONLY when no interpretation of the candidate reaches 24
    with the exact puzzle multiset, so a correct expression (even wrapped in prose)
    is never falsely refuted.
    """
    numbers = sorted(int(n) for n in re.findall(r"-?\d+", question))
    cleaned = _clean_g24(candidate)

    # Collect candidate arithmetic segments; prefer operator-bearing ones, scanning
    # last-to-first (the final expression in a trace), and accept on the first that
    # decisively reaches 24 with the right numbers.
    segments = re.findall(r"[\d\s()+\-*/.]+", cleaned)
    best: Optional[Tuple[str, str]] = None  # (segment, reason) for the diagnostic
    for seg in reversed(segments):
        seg = seg.strip().rstrip("=").strip()
        if not seg or not re.search(r"[+\-*/()]", seg):
            continue  # skip bare numbers
        try:
            value = _safe_arith_eval(seg)
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
            continue
        used = sorted(int(n) for n in re.findall(r"\d+", seg))
        if used != numbers:
            if best is None:
                best = (seg, f"it uses numbers {used}, but the puzzle requires "
                             f"exactly {numbers} (each used once)")
            continue
        if abs(value - 24.0) < 1e-6:
            return CheckResult("pass", f"{seg} = 24 using {numbers}")
        if best is None:
            best = (seg, f"it evaluates to {value:g}, not 24")

    if best is None:
        return CheckResult(
            "fail",
            f"no valid arithmetic expression over the numbers {numbers} could be "
            f"found in the answer, so it cannot be verified to equal 24",
        )
    seg, reason = best
    return CheckResult("fail", f"the expression {seg!r} is wrong: {reason}")


# ── CRUXEval ────────────────────────────────────────────────────────────────────

def _clean_literal(text: str) -> str:
    """Strip fences / common prefixes so a predicted Python literal stands alone."""
    s = text.strip()
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s).strip().strip("`").strip()
    s = re.sub(r"^(the answer is|answer|output|returns?|result)\s*[:=]?\s*",
               "", s, flags=re.IGNORECASE).strip()
    return s


def _literal_equal(a: str, b: str) -> bool:
    """Compare two literal strings as Python values, falling back to text equality."""
    try:
        return ast.literal_eval(a) == ast.literal_eval(b)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return a.strip() == b.strip()


def cruxeval_checker(
    question: str, candidate: str, probe: str = "", *, timeout: float = 10.0
) -> CheckResult:
    """c_q for CRUXEval: execute the reference function and compare its return value.

    Sound: the verdict comes from running the benchmark's own function on the given
    input; the model contributes only the candidate value, which is parsed as data.
    """
    cm = re.search(r"```(?:python)?\s*\n(.*?)```", question, re.DOTALL)
    if not cm:
        return CheckResult("undecided", "no reference function found in the problem")
    code = cm.group(1)
    tail = question[cm.end():]
    im = re.search(r"f\((.*)\)`?\s*return", tail, re.DOTALL)
    if im is None:
        return CheckResult("undecided", "no f(<input>) call found in the problem")
    inp = im.group(1).strip()

    harness = f"{code}\n\nprint(repr(f({inp})))\n"
    res = run_code(harness, timeout=timeout)
    if not res.success or not res.output.strip():
        return CheckResult(
            "undecided",
            f"the reference function could not be executed ({res.error or 'no output'})",
        )
    true_repr = res.output.strip().splitlines()[-1].strip()
    cand = _clean_literal(candidate)
    if _literal_equal(cand, true_repr):
        return CheckResult("pass", f"f({inp}) returns {true_repr}")
    return CheckResult(
        "fail",
        f"executing the function on the given input, f({inp}) returns {true_repr}, "
        f"not {cand}",
    )


# ── Python Programming Puzzles ───────────────────────────────────────────────────

# sat() predicates use bare PEP 484 generics (List[int], Dict[str, int], ...); make
# the typing names available without import in the execution harness.
_PP_PREAMBLE = "from typing import *\n"


def _extract_sat_src(question: str) -> Optional[str]:
    """Pull the ``sat`` function source out of a Programming-Puzzles question."""
    m = re.search(r"return True:\s*\n+(.*?)\n+Provide only the answer",
                  question, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(def sat\(.*)", question, re.DOTALL)
    return m2.group(1).strip() if m2 else None


def programmingpuzzles_checker(
    question: str, candidate: str, probe: str = "", *, timeout: float = 10.0
) -> CheckResult:
    """c_q for Python Puzzles: does ``sat(answer)`` return True?

    Sound: the verdict comes from running the puzzle's own ``sat`` predicate; the
    candidate is parsed with ``ast.literal_eval`` (never executed as code).
    """
    sat_src = _extract_sat_src(question)
    if not sat_src:
        return CheckResult("undecided", "no sat() predicate found in the problem")
    cand = _clean_literal(candidate)

    harness = (
        f"{_PP_PREAMBLE}{sat_src}\n\n"
        "import ast\n"
        f"__ans = ast.literal_eval({cand!r})\n"
        "print('FOT_PP_PASS' if sat(__ans) else 'FOT_PP_FAIL')\n"
    )
    res = run_code(harness, timeout=timeout)
    if not res.success:
        return CheckResult("undecided", f"sat() could not be evaluated ({res.error})")
    if "FOT_PP_FAIL" in res.output:
        return CheckResult(
            "fail",
            "sat(answer) returns False — the answer does not satisfy the puzzle "
            "predicate derived from the problem",
        )
    if "FOT_PP_PASS" in res.output:
        return CheckResult("pass", "sat(answer) returns True")
    return CheckResult("undecided", f"sat() produced no decisive verdict ({res.output!r})")


# ── BigBenchHard: multi-step arithmetic ─────────────────────────────────────────

def multistep_arithmetic_checker(
    question: str, candidate: str, probe: str = "", *, timeout: float = 10.0
) -> CheckResult:
    """c_q for BBH ``multistep_arithmetic_two``: recompute the stated expression.

    Sound: the expression is taken verbatim from the question and evaluated with
    the AST arithmetic evaluator, so the verdict never depends on the model.
    """
    expr = question.strip().rstrip("=").strip()
    try:
        value = _safe_arith_eval(expr)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
        return CheckResult("undecided", "the question is not a plain arithmetic expression")

    m = re.search(r"-?\d+(?:\.\d+)?", candidate.replace(",", ""))
    if not m:
        return CheckResult("undecided", "no numeric value found in the candidate answer")
    given = float(m.group(0))
    if abs(given - value) < 1e-6:
        return CheckResult("pass", f"{expr} = {value:g}")
    return CheckResult(
        "fail",
        f"evaluating the expression gives {expr} = {value:g}, not {given:g}",
    )


# ── Registry: realises HasChecker(q), fixed per benchmark ────────────────────────

# Keys are ``benchmark`` or ``benchmark:subtask``; lookup is most-specific-first,
# so a BigBenchHard subtask can carry a checker while its siblings do not.
CHECKERS: Dict[str, Checker] = {
    "gameof24": gameof24_checker,
    "cruxeval": cruxeval_checker,
    "programmingpuzzles": programmingpuzzles_checker,
    "bigbenchhard:multistep_arithmetic_two": multistep_arithmetic_checker,
}


def has_checker(task: Optional[str], subtask: Optional[str] = None) -> bool:
    """HasChecker(q): True iff this benchmark admits a cheap, decisive checker.

    Fixed per benchmark (keyed by name), not decided at run time by the model.
    """
    return get_checker(task, subtask) is not None


def get_checker(task: Optional[str], subtask: Optional[str] = None) -> Optional[Checker]:
    """Return the trusted checker c_q for ``task``, or None for the relational regime."""
    if not task:
        return None
    base = task.lower()
    if subtask:
        specific = CHECKERS.get(f"{base}:{subtask.lower()}")
        if specific is not None:
            return specific
        # A subtask of a benchmark whose checker is registered wholesale still
        # uses it; a subtask of an unregistered benchmark has none.
    return CHECKERS.get(base)
