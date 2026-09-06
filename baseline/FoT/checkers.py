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
  * ``checkmate``          — position replay: replay the PGN the question gives,
                             play the candidate move, and ask the board whether
                             it is checkmate.

Each checker is deterministic and derives its verdict ONLY from the problem
statement (which carries the reference function / puzzle numbers / sat predicate /
movetext) and the candidate answer — the model is not involved in the decision.
Tasks not registered here have ``HasChecker(q) = False`` and fall to the
relational regime.

One entry is conditional: ``checkmate`` needs the ``python-chess`` package to
replay a position, and is registered only when it imports. That is an
environment fact, not a run-time decision — for a given installation
``HasChecker(q)`` is still fixed per benchmark — but it does mean a run on a host
without ``python-chess`` falls back to the metamorphic regime, where the
checkmate benchmark has no catalogue and FoT degrades to FoT ≡ Solve. Check
``fot._has_checker`` (or the response metadata's ``regime``) if a result looks
unexpectedly cheap.

The mapping from benchmark to checker realises the paper's ``HasChecker(q)``
predicate: it is FIXED PER BENCHMARK (keyed by the benchmark name), not decided
at run time by the model.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# Reuse BoT's hardened, timeout-bounded subprocess executor to run the *reference*
# code embedded in the problem (CRUXEval functions, Python-Puzzles ``sat``). The
# code that runs is the benchmark's own, not model-authored verdict logic, so the
# verdict it yields is sound.
from baseline.BoT.bot import run_code

# The checkmate checker resolves the candidate with the benchmark's own move
# extractor, so the move the verifier plays is the move the grader will score.
from benchmark.Checkmate.checkmate import extract_move

# The sonnet checker re-runs the benchmark's own three constraints, so the
# verdict it issues is the verdict the grader will issue.
from benchmark.SonnetWriting.sonnetwriting import (
    _get_last_word, _rhyme_pairs, _scheme_length, _words_rhyme)


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


# ── Checkmate in one ────────────────────────────────────────────────────────────

# python-chess is what makes this checker sound; without it the benchmark simply
# has no checker (see the module docstring).
try:
    import chess as _chess

    _HAS_CHESS = True
except ImportError:  # pragma: no cover - depends on the installation
    _chess = None
    _HAS_CHESS = False

# PGN decorations that carry no move: comments, NAGs, game results, move numbers.
_PGN_COMMENT_RE = re.compile(r"\{[^}]*\}")
_PGN_NAG_RE = re.compile(r"\$\d+")
_PGN_RESULT_RE = re.compile(r"\b(?:1-0|0-1|1/2-1/2)\b|\*")
_PGN_MOVE_NUMBER_RE = re.compile(r"\d+\s*\.(?:\.\.)?")


def _replay_movetext(movetext: str):
    """Replay PGN movetext into the position it reaches.

    Returns the board, or None if any token fails to apply — an unreplayable
    game yields "undecided" rather than a verdict, so a correct answer is never
    discarded because the *question* could not be parsed.
    """
    text = _PGN_COMMENT_RE.sub(" ", movetext)
    text = _PGN_NAG_RE.sub(" ", text)
    text = _PGN_RESULT_RE.sub(" ", text)
    text = _PGN_MOVE_NUMBER_RE.sub(" ", text)

    board = _chess.Board()
    for token in text.split():
        try:
            board.push_san(token)
        except (ValueError, AssertionError):
            return None
    return board


def checkmate_checker(
    question: str, candidate: str, probe: str = "", *, timeout: float = 10.0
) -> CheckResult:
    """c_q for checkmate-in-one: play the candidate move and ask the board.

    Sound: the position comes from replaying the question's own movetext and the
    verdict from ``python-chess``'s rules engine, so a move that really does mate
    is never refuted. The candidate is resolved with the benchmark's own
    extractor, so the verifier plays exactly the move the grader will score.

    ``timeout`` is accepted for interface conformance and unused: replaying a
    game is pure in-process computation, with no subprocess to bound (a full
    3500-position sweep replays in ~2.5 s).

    The failure detail names one concrete escape — "after Qg8+, Black can reply
    Kxg8" — never the set of legal moves and never the mating move: the witness
    has to indict the candidate without handing the repair the answer the
    benchmark withholds.
    """
    if not _HAS_CHESS:  # pragma: no cover - depends on the installation
        return CheckResult("undecided", "python-chess is not installed")

    board = _replay_movetext(question)
    if board is None:
        return CheckResult("undecided", "the movetext could not be replayed")
    if board.is_game_over():
        return CheckResult("undecided", "the position given is already final")

    legal_san = [board.san(m) for m in board.legal_moves]
    written = extract_move(candidate, legal_san)
    if written is None:
        return CheckResult(
            "fail",
            "the answer names no move in standard algebraic notation, so no "
            "move can be played on the board (a single move such as Qxe7# is "
            "the whole answer)",
        )

    try:
        move = board.parse_san(written)
    except (ValueError, AssertionError):
        return CheckResult(
            "fail",
            f"{written} is not a legal move in the position the game reaches",
        )

    played = board.san(move)
    board.push(move)

    if board.is_checkmate():
        return CheckResult("pass", f"after {played} the opponent is checkmated")
    if board.is_stalemate():
        return CheckResult("fail", f"after {played} the position is stalemate, not checkmate")
    if not board.is_check():
        return CheckResult(
            "fail",
            f"{played} does not even give check — after it the opponent's king "
            f"is not attacked",
        )

    escape = next(iter(board.legal_moves), None)
    reply = board.san(escape) if escape is not None else ""
    return CheckResult(
        "fail",
        f"{played} gives check but is not mate: the opponent can reply {reply}"
        if reply else f"{played} gives check but is not mate",
    )


# ── Sonnet writing ──────────────────────────────────────────────────────────────

# The two gradeable constraints are stated in the prompt itself — the rhyme
# scheme and the required words — so the checker reads them off the question the
# way the checkmate checker replays the question's own movetext. Nothing is taken
# from the dataset's target, and nothing needs to be withheld from the repair:
# unlike checkmate, where the failure detail must not name the mating move, here
# the constraints are public and a concrete violation ("line 3 ends in 'sky',
# which does not rhyme with line 1's 'grass'") leaks nothing the prompt did not
# already say.
#
# The verdict is computed with the benchmark's own grading primitives, so this
# checker is not merely sound but *decisive*: it fails exactly the sonnets the
# benchmark would mark incorrect.

_SCHEME_RE = re.compile(r"rhyme scheme\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)\s*,", re.IGNORECASE)
_QUOTED_WORD_RE = re.compile(r"[\"“]([^\"“”]+)[\"”]")


def _sonnet_constraints(question: str) -> Optional[Tuple[str, List[str]]]:
    """Read the rhyme scheme and the required words out of the prompt itself."""
    scheme_match = _SCHEME_RE.search(question)
    words = [w.strip() for w in _QUOTED_WORD_RE.findall(question) if w.strip()]
    if scheme_match is None or not words:
        return None
    scheme = scheme_match.group(1).strip().upper()
    return (scheme, words) if _scheme_length(scheme) else None


def _sonnet_lines(candidate: str, expected: int) -> List[str]:
    """The lines the grader will score: non-empty, truncated to the scheme length."""
    lines = [ln.strip() for ln in candidate.split("\n") if ln.strip()]
    return lines[:expected] if len(lines) > expected else lines


def sonnet_checker(
    question: str, candidate: str, probe: str = "", *, timeout: float = 10.0
) -> CheckResult:
    """c_q for sonnet writing: re-run the benchmark's own three constraints.

    Sound *and* decisive: word inclusion, line count and the rhyme scheme are
    mechanical predicates, computed here with the benchmark's own helpers
    (``_rhyme_pairs``, ``_get_last_word``, ``_words_rhyme``), so a sonnet the
    grader would accept is never refuted and one it would reject never passes.

    ``timeout`` is accepted for interface conformance and unused — the check is
    pure string work with no subprocess to bound.

    The detail names *one* violation at a time, most structural first: a poem
    with the wrong number of lines has to be resized before its rhymes mean
    anything. Repairing against one concrete defect works better than against a
    list, and the remaining defects surface on the next round.
    """
    constraints = _sonnet_constraints(question)
    if constraints is None:
        return CheckResult("undecided",
                           "the prompt states no rhyme scheme and quoted words")
    scheme, required = constraints
    expected = _scheme_length(scheme)
    lines = _sonnet_lines(candidate, expected)

    if len(lines) != expected:
        return CheckResult(
            "fail",
            f"the poem has {len(lines)} non-empty lines but the scheme {scheme} "
            f"prescribes {expected}"
            + (" (blank lines between stanzas are fine; it is the count of "
               "verse lines that is short)" if len(lines) < expected else ""),
        )

    missing = [w for w in required
               if not re.search(rf"\b{re.escape(w)}\b", candidate, re.IGNORECASE)]
    if missing:
        quoted = ", ".join(f"'{w}'" for w in missing)
        return CheckResult(
            "fail",
            f"the required word{'s' if len(missing) > 1 else ''} {quoted} "
            f"do{'' if len(missing) > 1 else 'es'} not appear verbatim in the poem",
        )

    end_words = [_get_last_word(line) for line in lines]
    for i, j in _rhyme_pairs(scheme):
        if i >= len(end_words) or j >= len(end_words):
            continue
        if not _words_rhyme(end_words[i], end_words[j]):
            return CheckResult(
                "fail",
                f"the scheme {scheme} requires line {i + 1} and line {j + 1} to "
                f"rhyme, but they end in '{end_words[i]}' and '{end_words[j]}'",
            )

    return CheckResult(
        "pass",
        f"{expected} lines, every required word present, and every rhyme pair "
        f"of {scheme} matches",
    )


# ── Registry: realises HasChecker(q), fixed per benchmark ────────────────────────

# Keys are ``benchmark`` or ``benchmark:subtask``; lookup is most-specific-first,
# so a BigBenchHard subtask can carry a checker while its siblings do not.
CHECKERS: Dict[str, Checker] = {
    "gameof24": gameof24_checker,
    "cruxeval": cruxeval_checker,
    "programmingpuzzles": programmingpuzzles_checker,
    "sonnetwriting": sonnet_checker,
    "bigbenchhard:multistep_arithmetic_two": multistep_arithmetic_checker,
}

# Conditional: without python-chess there is no way to replay a position, and a
# checker that always answers "undecided" would be worse than none — it would
# suppress the metamorphic regime while deciding nothing.
if _HAS_CHESS:
    CHECKERS["checkmate"] = checkmate_checker


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
