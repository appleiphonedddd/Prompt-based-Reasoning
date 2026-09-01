"""Checkmate-in-One Benchmark Dataset.

BIG-bench task ``checkmate_in_one``: given a chess game as a PGN move list,
name the single move that delivers checkmate.

The task is a pure search-and-verify reasoning problem — the model must keep
track of a position built up over dozens of plies, enumerate the moves that are
legal in it, and recognise the one that leaves the opponent's king attacked with
no escape. That makes it a natural companion to Game of 24 in this suite: a
large, mechanically-checkable answer space where a plausible-looking answer is
almost always refutable.

Dataset:
    3500 examples loaded from ``benchmark/Checkmate/checkmate_in_one.json``
    (the BIG-bench task file, kept verbatim — its ``task_prefix`` supplies this
    benchmark's instruction).

Each example contains:
    input:         the game so far in PGN movetext, e.g. ``"1. d4 d5 2. Nf3 …"``
    target:        the mating move in standard algebraic notation, e.g. ``"Rg5#"``
    target_scores: every legal move in the position, mapped to 1 for the mating
                   move and 0 for the rest

Evaluation strategy:
    1. Extract a SAN move from the model's raw output (handles ``\\boxed{}``,
       "the answer is …", Markdown, and bare-move responses alike).
    2. Canonicalise it against the position's legal moves, so a move written
       ``Qe7#`` instead of ``Qxe7#`` is resolved to the move it uniquely names
       rather than scored wrong on notation alone.
    3. Compare with the reference move, ignoring the check/mate suffix and any
       ``!?`` annotation — the answer is the *move*, not its punctuation.

    Note that ``target_scores`` is deliberately NOT shown to the model: exactly
    one of its keys carries the ``#`` suffix, so the option list would give the
    answer away. This matches the BIG-bench task's own
    ``append_choices_to_input: False`` and its ``exact_str_match`` metric; the
    legal-move list is used only for grading.

Reference:
    BIG-bench, "checkmate_in_one"
    https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/checkmate_in_one

Author: Egor Morozov
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


# ─────────────────────────────────────────────────────────────────────────────
# Standard algebraic notation (SAN)
# ─────────────────────────────────────────────────────────────────────────────

# A single SAN move: castling, a piece move (with optional disambiguating file
# and/or rank), a pawn capture, or a pawn push (with optional promotion), plus an
# optional check/mate suffix and "!?"-style annotation.
#
# The alternation is ordered longest-first (O-O-O before O-O) and the boundaries
# keep the pattern from firing inside a word — "Nf3" in prose matches, "she4"
# does not.
_SAN_RE = re.compile(
    r"(?<![\w\-])"
    r"(?:"
    r"O-O-O|O-O"                                    # castling (queenside first)
    r"|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]"             # piece move, opt. disambiguation
    r"|[a-h]x[a-h][1-8](?:=[KQRBN])?"               # pawn capture, opt. promotion
    r"|[a-h][1-8](?:=[KQRBN])?"                     # pawn push, opt. promotion
    r")"
    r"[+#]?[!?]{0,2}"
    r"(?![\w])"
)

# Leading move number a model may keep in its answer: "32." or "31..." .
_MOVE_NUMBER_RE = re.compile(r"^\d+\s*\.+\s*")

# Answer-announcing phrases. The captured tail is where the move is stated, so
# the FIRST SAN token inside it is the answer.
_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer|mating\s+move|checkmate(?:-|\s)in(?:-|\s)one|"
    r"the\s+move|solution)"
    r"\s*(?:is|:|=|-)?\s*(.+)",
    re.IGNORECASE,
)

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _strip_markup(text: str) -> str:
    """Remove Markdown / LaTeX decoration that would hide a SAN move.

    Only decoration is dropped — never characters that are part of a move — so
    the token positions the extractor relies on stay meaningful.
    """
    text = re.sub(r"```[a-zA-Z0-9]*", "", text)
    text = text.replace("`", "").replace("**", "").replace("*", "")
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = text.replace("$", "")
    return text


def _normalize_san(move: str) -> str:
    """Canonical form of a SAN move for comparison.

    Strips a leading move number, normalises the digit-zero spelling of castling
    (``0-0`` → ``O-O``) and unicode dashes, and removes the trailing ``!?``
    annotation and the ``+``/``#`` suffix: check and mate marks describe the
    consequence of the move, not the move itself, so ``Rg5`` and ``Rg5#`` are
    the same answer.

    Case is preserved — SAN is case-sensitive (``Bd2`` is a bishop move, ``bd2``
    a pawn from the b-file).
    """
    s = move.strip()
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = _MOVE_NUMBER_RE.sub("", s).strip()
    s = re.sub(r"^0-0-0", "O-O-O", s)
    s = re.sub(r"^0-0", "O-O", s)
    s = s.rstrip("!?").rstrip()
    s = s.rstrip("+#").rstrip()
    return s


def _loose_san(move: str) -> str:
    """Notation-insensitive key for a SAN move.

    Drops the capture marker and the promotion ``=``, which are redundant given
    the position: within one position no two legal moves share a loose form
    unless SAN would have disambiguated them anyway. Used to recognise a move
    the model wrote as ``Qe7`` when the position calls it ``Qxe7`` — and only
    when the loose form names exactly one legal move, so the resolution is
    unique, never a guess.
    """
    return _normalize_san(move).replace("x", "").replace("=", "")


@dataclass(frozen=True)
class _LegalIndex:
    """Lookup from a written move to the legal move it names.

    Attributes:
        exact: normalised SAN -> the position's own spelling of that move.
        loose: loose SAN -> every legal move sharing that loose form. A form
               claimed by more than one legal move is ambiguous and is never
               resolved.
    """

    exact: Dict[str, str] = field(default_factory=dict)
    loose: Dict[str, List[str]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.exact)

    def is_legal(self, move: str) -> bool:
        """Whether *move* names a legal move of the position (either spelling)."""
        return _normalize_san(move) in self.exact or _loose_san(move) in self.loose

    def canonicalize(self, move: str) -> Optional[str]:
        """Return the position's spelling of *move*, or ``None`` if unresolved."""
        exact = self.exact.get(_normalize_san(move))
        if exact is not None:
            return exact
        candidates = self.loose.get(_loose_san(move))
        if candidates is not None and len(candidates) == 1:
            return candidates[0]
        return None


def _build_legal_index(legal_moves: Optional[Iterable[str]]) -> _LegalIndex:
    """Index a position's legal moves for canonicalisation."""
    index = _LegalIndex()
    for move in legal_moves or ():
        if not isinstance(move, str) or not move.strip():
            continue
        index.exact.setdefault(_normalize_san(move), move)
        index.loose.setdefault(_loose_san(move), []).append(move)
    return index


def _answer_regions(text: str) -> Iterator[Tuple[str, bool]]:
    """Yield ``(region, prefer_last)`` slices of *text*, most reliable first.

    A region is a stretch of the response that plausibly contains the final
    answer. ``prefer_last`` says which SAN token to take inside it: an
    answer-announcing region starts *at* the answer, so its first move wins,
    while a whole line or the whole response ends with the model's conclusion,
    so its last move wins.
    """
    boxed = _BOXED_RE.findall(text)
    if boxed:
        yield boxed[-1], False

    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        yield matches[-1].group(1), False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        yield lines[-1], True

    yield text, True


def _pick_move(region: str, index: _LegalIndex, prefer_last: bool) -> Optional[str]:
    """Choose one SAN move from *region*, preferring moves legal in the position."""
    tokens = [m.group(0) for m in _SAN_RE.finditer(region)]
    if not tokens:
        return None
    legal = [t for t in tokens if index.is_legal(t)]
    if legal:
        tokens = legal
    return tokens[-1] if prefer_last else tokens[0]


def extract_move(
    text: str,
    legal_moves: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Extract the move the model settled on from its raw output.

    Works across all baselines: a bare ``Qxe7#``, a CoT trace ending in
    "Therefore the answer is Qxe7#", a ``\\boxed{Qxe7\\#}``, or a Markdown-fenced
    move. Regions are tried in order of reliability, and within a region a move
    that is actually legal in the position beats one that is not — a CoT trace
    is full of squares and piece letters, and legality is what separates the
    move being proposed from the ones being discussed.

    Public because FoT's trusted checker (``baseline/FoT/checkers.py``) resolves
    the candidate with it too: the move the verifier plays on the board must be
    the same one this benchmark will grade, or the witness would be about a
    different answer than the score.

    Args:
        text: Raw model response.
        legal_moves: The position's legal moves, when known.

    Returns:
        The extracted SAN move, or ``None`` if the response contains no move.
    """
    if not text or not text.strip():
        return None

    cleaned = _strip_markup(text)
    index = _build_legal_index(legal_moves)

    for region, prefer_last in _answer_regions(cleaned):
        move = _pick_move(region, index, prefer_last)
        if move is not None:
            return move
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset implementation
# ─────────────────────────────────────────────────────────────────────────────

class Checkmate(DatasetBase):
    """Checkmate-in-one benchmark: name the mating move in a chess position.

    Args:
        split:       Unused; kept for interface consistency (default ``"test"``).
        num_samples: Maximum number of positions to load, taken from the front
                     of the file. ``None`` loads all 3500 — a full sweep is
                     expensive, so a few hundred is the usual setting.

    Example::

        ds = Checkmate(num_samples=100)
        ds.load_dataset()
        problem = ds.get_problem(0)
        result = ds.evaluate_answer("The mating move is Rg5#.", problem.ground_truth)
        print(result.is_correct)   # True
    """

    DATA_FILE = "checkmate_in_one.json"

    # Used until load_dataset() replaces it with the task file's own prefix.
    DEFAULT_TASK_PREFIX = (
        "In the following chess position, find a checkmate-in-one move."
    )

    def __init__(self, split: str = "test", num_samples: Optional[int] = None) -> None:
        """Initialise the Checkmate benchmark.

        Args:
            split:       Dataset split (kept for interface consistency).
            num_samples: Maximum positions to load; ``None`` means all.
        """
        super().__init__(split=split, dataset_name="Checkmate")
        self.num_samples = num_samples
        self.task_prefix = self.DEFAULT_TASK_PREFIX

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Load the positions from the local BIG-bench task file.

        Populates ``self._data`` with the (optionally truncated) list of example
        dicts and adopts the file's own ``task_prefix`` as this benchmark's
        instruction, so the prompt stays in step with the task definition.

        Raises:
            RuntimeError: If the task file is missing or malformed.
        """
        data_file = Path(__file__).parent / self.DATA_FILE
        if not data_file.exists():
            raise RuntimeError(
                f"[{self.dataset_name}] Data file not found: {data_file}\n"
                f"Expected {self.DATA_FILE} in benchmark/Checkmate/."
            )

        try:
            with open(data_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            examples: List[Dict[str, Any]] = raw["examples"]
        except Exception as exc:
            raise RuntimeError(
                f"[{self.dataset_name}] Failed to load '{data_file}': {exc}"
            ) from exc

        self.task_prefix = (raw.get("task_prefix") or self.DEFAULT_TASK_PREFIX).strip()

        if self.num_samples is not None:
            examples = examples[: self.num_samples]

        self._data = examples

        suffix = f" (num_samples={self.num_samples})" if self.num_samples is not None else ""
        print(f"[{self.dataset_name}] Loaded {len(self._data)} positions{suffix}.")

    def get_problem(self, index: int) -> Problem:
        """Return the position at *index*.

        ``Problem.question`` is the PGN movetext of the game so far — the legal
        moves are withheld, since the mating one is the only key marked ``#``.

        ``Problem.ground_truth`` is a dict with keys:

        - ``move``        — the mating move in SAN, e.g. ``"Rg5#"``
        - ``legal_moves`` — every legal move in the position, used to
                            canonicalise the model's notation while grading

        Args:
            index: Zero-based index into the loaded examples.

        Returns:
            A ``Problem`` ready for baseline evaluation.

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
        movetext: str = str(row.get("input", "")).strip()
        target: str = str(row.get("target", "")).strip()
        legal_moves: List[str] = list(row.get("target_scores") or {})

        return Problem(
            index=index,
            question=movetext,
            ground_truth={
                "move": target,
                "legal_moves": legal_moves,
            },
            metadata={
                "num_legal_moves": len(legal_moves),
                # A movetext ending in a move number ("… 31.") means White has
                # yet to move; otherwise it is Black's turn.
                "side_to_move": "white" if re.search(r"\d+\s*\.\s*$", movetext) else "black",
            },
        )

    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Score a predicted move against the mating move.

        Pipeline:

        1. Extract a SAN move from *prediction* (``extract_move``).
        2. Canonicalise it against the position's legal moves, so a move named
           with different-but-unambiguous notation is credited.
        3. Compare with the reference, ignoring check/mate suffix and
           annotations. If canonicalisation found nothing, fall back to a
           notation-insensitive comparison against the reference alone, so
           grading still works without a legal-move list.

        Args:
            prediction:   The model's raw output (``BaselineResponse.final_answer``).
            ground_truth: Dict with key ``move`` (and optionally ``legal_moves``),
                          or the mating move as a plain string.

        Returns:
            ``EvaluationResult`` with ``is_correct``, ``score`` ∈ {0.0, 1.0},
            and diagnostic ``details``.
        """
        details: Dict[str, Any] = {"raw_prediction": prediction}

        if isinstance(ground_truth, dict):
            target = str(ground_truth.get("move", ""))
            legal_moves = ground_truth.get("legal_moves") or []
        else:
            target = str(ground_truth or "")
            legal_moves = []

        if not target:
            details["error"] = "ground_truth carries no mating move."
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        details["target_move"] = target
        details["num_legal_moves"] = len(legal_moves)

        extracted = extract_move(prediction or "", legal_moves)
        details["extracted_move"] = extracted

        if extracted is None:
            details["comparison_method"] = "none"
            return EvaluationResult(
                is_correct=False, score=0.0,
                prediction=prediction, ground_truth=ground_truth,
                details=details,
            )

        index = _build_legal_index(legal_moves)
        canonical = index.canonicalize(extracted)
        if canonical is not None:
            details["canonical_move"] = canonical
            details["comparison_method"] = "legal_move"
            is_correct = _normalize_san(canonical) == _normalize_san(target)
        else:
            details["comparison_method"] = "notation"
            is_correct = (
                _normalize_san(extracted) == _normalize_san(target)
                or _loose_san(extracted) == _loose_san(target)
            )

        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            prediction=prediction,
            ground_truth=ground_truth,
            details=details,
        )

    # ── Optional hook overrides ────────────────────────────────────────────

    def get_instruction(self) -> str:
        """Return the task instruction, built on the task file's own prefix."""
        return (
            f"{self.task_prefix}\n"
            "The game so far is given in PGN movetext. Find the one legal move "
            "for the side to play that delivers immediate checkmate.\n"
            "Answer with that single move in standard algebraic notation "
            "(e.g. Qxe7#, Rg5#, O-O#) and nothing else — no move number, "
            "no commentary."
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for checkmate-in-one evaluation."""
        return (
            "You are a chess grandmaster. You are given a game in PGN movetext. "
            "Replay the moves to reconstruct the current position, then find the "
            "move that delivers checkmate in one. Respond with only that move in "
            "standard algebraic notation."
        )

    def _demo_output(self, problem: Problem) -> Optional[str]:
        """Render the mating move for an RoT demonstration.

        The ground truth is a dict, so the default derivation in
        :meth:`DatasetBase.get_demonstrations` needs this hook to know which
        field is the expected output.
        """
        gt = problem.ground_truth
        if isinstance(gt, dict) and gt.get("move"):
            return str(gt["move"])
        return None
