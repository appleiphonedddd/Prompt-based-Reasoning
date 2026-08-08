"""
The relation library R_q for Falsification-of-Thought (FoT), §3.3.2 of the paper.

A *relation schema* is a pair ``rho = (T, phi)`` where ``T`` constructs a check
instance from ``(q, a)`` and ``phi`` is the deterministic output relation the
instance must satisfy. ``rho`` is *valid* for a task if ``phi`` holds whenever
the answers involved are the ground-truth ones. Validity makes ``phi`` a
necessary property of the intended functionality — the role played by the
model-invented necessary conditions in the pilot, except that ``phi`` is fixed
offline and audited once by a human instead of being re-invented per query.

FoT uses the paper's three schema families, tagged by ``Relation.family``:

  * ``"bwd"`` — **backward substitution** ``rho_bwd``. The driver mechanically
    enumerates the quantities stated in ``q`` (numeric literals) and masks one as
    ``X``; the model re-derives it from the candidate through ``pi_mask``, and
    ``phi: x_hat = x``. This is the Self-Verification / FOBAR construction,
    repurposed from vote-weighting to witness generation.
  * ``"mr"`` — **metamorphic transformation** ``rho_mr = (T, phi)``. ``T``
    perturbs the query into a follow-up ``q_f``; the model solves ``q_f`` with
    ``pi_solve`` (never seeing the candidate), and ``phi(a, a_f)`` is checked
    mechanically. Semantics-preserving transforms have ``phi: a_f = a``;
    covariant transforms (scaling) have a task-declared ``phi``, e.g.
    ``a_f = c * a``.
  * ``"inv"`` — **mechanical invariant** ``rho_inv``. The comparator computes
    *both* sides from the problem data and the candidate, so the check costs no
    model call at all (e.g. the vertex count and path closure that an SVG path
    implies, versus the ones the named shape implies).

Soundness (§3.6, the ladder): a violated *valid* relation certifies that some
answer involved is wrong — as a matter of logic, with no appeal to the model's
opinion of its own work. The residual unsoundness is therefore confined to
(i) an invalid relation in this file, (ii) a systematic majority error in the
voted derivation, and (iii) attribution (for ``mr`` the defect may sit at
``a_f``, which is why ``pi_rep`` explicitly licenses the model to defend its
answer). Item (i) is a design-time property of the small table below; **this
module is the artifact to audit**.

Two properties are required of every entry: the relation must be valid, and its
image must be answered at least as reliably as its source. Distractor insertion
is a valid relation but is *excluded* on the second ground.

Most transformations here are purely programmatic (rescaling numbers, masking a
literal, permuting sentences or answer options, applying an affine map to an SVG
path, renaming identifiers), so they cost no model call and leave no room for
hallucination in ``T`` itself. Only paraphrase-style transformations (``apply is
None``) are delegated to the model through the ``pi_mr`` template; those are all
answer-preserving *and* mechanically validated afterwards to preserve every
numeric literal of the original (``preserve_numbers``), exactly as §3.3.2
requires of model-instantiated transforms.

Audit notes on the entries below:

  * ``scale_quantities`` (MGSM) is the paper's ``(T_scale, phi_lin)`` and is
    valid only when the answer is homogeneous of degree one in the scaled
    quantities. It is *not* valid when a problem multiplies two scaled quantities
    together (a price times a count scales as ``c^2``), so it can manufacture a
    spurious violation. Under the new driver a single violation is decisive, so
    the guards are now (a) library *ordering* — it sits last, behind the backward
    and semantics-preserving relations, and the driver returns on the first
    violation, so it is only ever reached once the sounder checks have abstained;
    (b) the voted derivation; and (c) ``pi_rep``'s licence to defend the
    candidate against a pair-implicating witness. Drop it entirely with
    ``--fot_relations`` if a stricter library is wanted.
  * ``mask_quantity`` is the one relation whose check instance deliberately
    contains the candidate (that is what backward substitution *is*: recover a
    masked premise *from* the conclusion). The independence requirement — the
    candidate must not be shown to the solver — applies to the ``mr`` family,
    where showing it would make the follow-up an echo; here the candidate is the
    input of a different question.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# ── Answer comparison (the driver evaluates rho, never the model) ───────────────

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(answer: str) -> Optional[float]:
    """Parse the first numeric value out of an answer string, or None."""
    s = answer.strip().replace("$", "").replace("%", "")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def normalize_answer(answer: str) -> str:
    """Normalise an answer for equality comparison (case, punctuation, spacing)."""
    s = answer.strip().lower()
    s = re.sub(r"^(the\s+)?(final\s+)?answer\s+is\s*", "", s)
    s = s.strip().strip("`\"'").strip()
    s = re.sub(r"[.。]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def option_letter(answer: str) -> Optional[str]:
    """Extract a multiple-choice option letter, e.g. '(D)' / 'D' / 'D) kite' → 'D'."""
    m = re.search(r"\(\s*([A-Z])\s*\)", answer)
    if m:
        return m.group(1)
    m = re.match(r"\s*([A-Z])\s*[).:]", answer)
    if m:
        return m.group(1)
    s = answer.strip()
    return s.upper() if len(s) == 1 and s.isalpha() else None


def numbers_equal(x: Optional[float], y: Optional[float], tol: float = 1e-6) -> bool:
    """Compare two parsed numbers with a relative tolerance."""
    if x is None or y is None:
        return False
    return math.isclose(x, y, rel_tol=tol, abs_tol=tol)


def answers_equal(a: str, b: str) -> bool:
    """phi for answer-preserving relations: numeric when both parse, else textual."""
    la, lb = option_letter(a), option_letter(b)
    if la and lb:
        return la == lb
    na, nb = parse_number(a), parse_number(b)
    if na is not None and nb is not None:
        return numbers_equal(na, nb)
    return normalize_answer(a) == normalize_answer(b)


_ANY_NUM = re.compile(r"\d+(?:\.\d+)?")


def numeric_literals(text: str) -> List[str]:
    """The multiset of numeric literals in a text, normalised for comparison.

    Used to validate model-instantiated transforms: §3.3.2 requires that a
    textual transform be "mechanically validated to preserve every numeric
    literal it is required to preserve", so a paraphrase that quietly drops or
    invents a quantity is rejected rather than checked.
    """
    return sorted(_fmt(float(m.group(0))) for m in _ANY_NUM.finditer(text))


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Variant:
    """One realised *check instance*: what to ask, and how to judge the reply.

    The same container serves all three schema families; ``family`` says how the
    driver must realise it.

    Attributes:
        relation: name of the relation rho that produced this instance.
        relation_text: phi, written out for the witness and the repair prompt.
        family: "bwd" (solve ``question`` with pi_mask, parse DERIVED:),
            "mr" (solve ``question`` with pi_solve, parse ANSWER:), or "inv"
            (no model call at all — ``expected`` and ``observed`` are already
            filled in by the comparator).
        question: the instance handed to the model — ``q`` with one quantity
            masked (bwd) or ``q_f = T(q)`` (mr). Empty for "inv".
        holds: phi — evaluated by the driver, never by the model. Called as
            ``holds(a, value)`` where ``value`` is the voted model output.
        expected: the value phi requires, when it does not depend on ``a``
            (bwd: the masked literal; inv: the comparator's left-hand side).
        observed: inv only — the comparator's right-hand side.
        detail: inv only — a sentence explaining what the two sides mean.
        expected_for: for "mr", phi(a): the answer the follow-up *must* receive
            given the candidate, used to render the witness.
        pullback: g^-1, mapping the instance's answer back into q's frame. None
            when the instance answers a *different* question (bwd).
        slot: bwd only — the stated quantity that was masked (NextSlot's choice).
        source: "programmatic" (T ran as code) or "model" (T came from pi_mr).
    """

    relation: str
    relation_text: str
    question: str
    holds: Callable[[str, str], bool]
    family: str = "mr"
    expected: Optional[str] = None
    observed: Optional[str] = None
    detail: str = ""
    expected_for: Optional[Callable[[str], Optional[str]]] = None
    pullback: Optional[Callable[[str], Optional[str]]] = None
    slot: Optional[str] = None
    source: str = "programmatic"

    def expected_value(self, answer: str) -> Optional[str]:
        """What phi requires the instance's answer to be, given the candidate."""
        if self.expected is not None:
            return self.expected
        return self.expected_for(answer) if self.expected_for is not None else None


@dataclass
class Relation:
    """A library entry rho = (T, phi).

    Attributes:
        name: stable identifier, usable with ``--fot_relations``.
        transformation: one-line description of T. Doubles as the instruction
            handed to pi_mr when T cannot be applied programmatically.
        direction: "symmetric" or "increasing" — the reliability requirement.
            Every library entry must be non-decreasing in reliability.
        relation_text: phi, in words.
        family: "bwd" | "mr" | "inv" (see :class:`Variant`).
        apply: programmatic T for the "mr" family. Returns a Variant, or None
            when the relation does not apply to this particular query (which is
            not a violation — the relation is simply skipped). When None *and*
            the family is "mr", the transformation is delegated to the model via
            pi_mr and phi is equality.
        apply_slot: T for the "bwd" family, taking the slot index chosen by
            NextSlot so that successive attempts mask *distinct* quantities.
        invariant: for the "inv" family — computes
            ``(expected, observed, detail)`` straight from ``(q, a)``, or None
            when it does not apply. The first two are compared verbatim.
        applicable: optional guard for model-applied relations.
        preserve_numbers: for model-applied T only — reject the model's variant
            unless it carries exactly the numeric literals of the original.
    """

    name: str
    transformation: str
    direction: str
    relation_text: str
    family: str = "mr"
    apply: Optional[Callable[[str, str], Optional[Variant]]] = None
    apply_slot: Optional[Callable[[str, str, int], Optional[Variant]]] = None
    invariant: Optional[
        Callable[[str, str], Optional[Tuple[str, str, str]]]] = None
    applicable: Optional[Callable[[str], bool]] = None
    preserve_numbers: bool = True

    @property
    def programmatic(self) -> bool:
        """True iff realising this relation costs no model call to *construct*."""
        return (self.apply is not None or self.apply_slot is not None
                or self.invariant is not None)


def equality_variant(relation: Relation, question: str, source: str = "model") -> Variant:
    """Build an answer-preserving instance (phi = equality, g = g^-1 = id)."""
    return Variant(
        relation=relation.name,
        relation_text=relation.relation_text,
        question=question,
        holds=answers_equal,
        family="mr",
        expected_for=lambda a: a,
        pullback=lambda a_prime: a_prime,
        source=source,
    )


# ── SVG path relations (BBH geometric_shapes) ──────────────────────────────────

_PATH_RE = re.compile(r'd\s*=\s*"([^"]*)"')
_POINT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


def _path_points(question: str) -> Optional[Tuple[str, List[Tuple[float, float]]]]:
    """Return the raw ``d`` attribute and its coordinates, for M/L-only paths.

    Anything containing a curve or arc command is rejected: an affine map on the
    raw numbers would corrupt an arc's radii/flags, which would make the relation
    invalid rather than merely inapplicable.
    """
    m = _PATH_RE.search(question)
    if not m:
        return None
    d = m.group(1)
    if re.search(r"[A-Za-z]", re.sub(r"[MLml]", "", d)):
        return None
    pts = [(float(x), float(y)) for x, y in _POINT_RE.findall(d)]
    return (d, pts) if len(pts) >= 3 else None


def _rewrite_path(question: str, d: str, new_pts: List[Tuple[float, float]]) -> str:
    """Substitute new coordinates into the path, keeping the command skeleton."""
    it = iter(new_pts)

    def _sub(m: re.Match) -> str:
        x, y = next(it)
        return f"{x:.2f},{y:.2f}"

    return question.replace(f'd="{d}"', f'd="{_POINT_RE.sub(_sub, d)}"', 1)


def _svg_affine(relation: Relation, question: str,
                fn: Callable[[float, float], Tuple[float, float]]) -> Optional[Variant]:
    parsed = _path_points(question)
    if parsed is None:
        return None
    d, pts = parsed
    return equality_variant(
        relation, _rewrite_path(question, d, [fn(x, y) for x, y in pts]),
        source="programmatic")


def _apply_translate_scale(question: str, candidate: str) -> Optional[Variant]:
    return _svg_affine(REL_SVG_TRANSLATE_SCALE, question,
                       lambda x, y: (1.5 * x + 13.0, 1.5 * y - 7.0))


def _apply_rotate(question: str, candidate: str) -> Optional[Variant]:
    parsed = _path_points(question)
    if parsed is None:
        return None
    _, pts = parsed
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    th = math.radians(37.0)
    cos_t, sin_t = math.cos(th), math.sin(th)

    def _rot(x: float, y: float) -> Tuple[float, float]:
        dx, dy = x - cx, y - cy
        return (cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t)

    return _svg_affine(REL_SVG_ROTATE, question, _rot)


def _apply_reverse(question: str, candidate: str) -> Optional[Variant]:
    """Reverse the traversal order of the vertices.

    The path is first canonicalised into a single ``M v1 L v2 ...`` polyline (BBH
    paths repeat ``M`` per segment), then emitted in reverse. Canonicalisation is
    itself answer-preserving, so the composite relation stays valid.
    """
    parsed = _path_points(question)
    if parsed is None:
        return None
    d, pts = parsed
    walk: List[Tuple[float, float]] = []
    for p in pts:
        if not walk or walk[-1] != p:
            walk.append(p)
    if len(walk) < 3:
        return None
    rev = list(reversed(walk))
    new_d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in rev)
    return equality_variant(
        REL_SVG_REVERSE, question.replace(f'd="{d}"', f'd="{new_d}"', 1),
        source="programmatic")


REL_SVG_TRANSLATE_SCALE = Relation(
    name="svg_translate_scale",
    transformation="translate and uniformly scale every control point of the path",
    direction="symmetric",
    relation_text="the shape is unchanged, so the answer must be identical",
    apply=_apply_translate_scale,
)
REL_SVG_ROTATE = Relation(
    name="svg_rotate",
    transformation="rotate every control point of the path by 37 degrees about its centroid",
    direction="symmetric",
    relation_text="the shape is unchanged, so the answer must be identical",
    apply=_apply_rotate,
)
REL_SVG_REVERSE = Relation(
    name="svg_reverse",
    transformation="reverse the order in which the vertices of the path are traversed",
    direction="symmetric",
    relation_text="the shape is unchanged, so the answer must be identical",
    apply=_apply_reverse,
)


# ── Multiple-choice option permutation (BBH letter-answer tasks) ────────────────

_OPTION_RE = re.compile(r"^\s*\(([A-Z])\)\s*(.*\S)\s*$")


def _permute_options(question: str, name: str, relation_text: str,
                     perm: Callable[[int, int], int], min_options: int = 2
                     ) -> Optional[Variant]:
    """Relabel the option bodies among the option letters.

    Answer-transforming with a known map: the variant's letter at position i
    carries the body that sat at position ``perm(i, L)`` in the original, so
    ``g^-1`` sends that letter back to the original one. A textbook metamorphic
    relation for multiple choice, and entirely programmatic.
    """
    lines = question.splitlines()
    idx = [i for i, ln in enumerate(lines) if _OPTION_RE.match(ln)]
    if len(idx) < min_options or idx != list(range(idx[0], idx[0] + len(idx))):
        return None
    parsed = [_OPTION_RE.match(lines[i]) for i in idx]
    letters = [m.group(1) for m in parsed]           # type: ignore[union-attr]
    bodies = [m.group(2) for m in parsed]            # type: ignore[union-attr]
    size = len(letters)
    if len(set(letters)) != size:
        return None

    new_lines = list(lines)
    for pos, i in enumerate(idx):
        new_lines[i] = f"({letters[pos]}) {bodies[perm(pos, size)]}"
    # letter in the variant -> letter in the original
    back = {letters[pos]: letters[perm(pos, size)] for pos in range(size)}
    # letter in the original -> letter in the variant (phi, for the witness)
    fwd = {original: variant for variant, original in back.items()}

    def _pull(a_prime: str) -> Optional[str]:
        letter = option_letter(a_prime)
        return f"({back[letter]})" if letter in back else None

    def _fwd(a: str) -> Optional[str]:
        letter = option_letter(a)
        return f"({fwd[letter]})" if letter in fwd else None

    def _holds(a: str, a_prime: str) -> bool:
        pulled = _pull(a_prime)
        return answers_equal(a, pulled) if pulled is not None else True

    return Variant(
        relation=name,
        relation_text=relation_text,
        question="\n".join(new_lines),
        holds=_holds,
        family="mr",
        expected_for=_fwd,
        pullback=_pull,
        source="programmatic",
    )


_OPTIONS_RELABELLED = ("the options were relabelled: the chosen letter must denote "
                       "the same option text as before")


def _apply_options_shift1(question: str, candidate: str) -> Optional[Variant]:
    return _permute_options(question, "options_shift1", _OPTIONS_RELABELLED,
                            lambda i, n: (i - 1) % n)


def _apply_options_shift2(question: str, candidate: str) -> Optional[Variant]:
    return _permute_options(question, "options_shift2", _OPTIONS_RELABELLED,
                            lambda i, n: (i - 2) % n, min_options=3)


def _apply_options_reverse(question: str, candidate: str) -> Optional[Variant]:
    return _permute_options(question, "options_reverse", _OPTIONS_RELABELLED,
                            lambda i, n: n - 1 - i)


REL_OPTIONS_SHIFT1 = Relation(
    name="options_shift1",
    transformation="cyclically relabel the answer options by one, keeping their texts intact",
    direction="symmetric",
    relation_text=_OPTIONS_RELABELLED,
    apply=_apply_options_shift1,
)
REL_OPTIONS_SHIFT2 = Relation(
    name="options_shift2",
    transformation="cyclically relabel the answer options by two, keeping their texts intact",
    direction="symmetric",
    relation_text=_OPTIONS_RELABELLED,
    apply=_apply_options_shift2,
)
REL_OPTIONS_REVERSE = Relation(
    name="options_reverse",
    transformation="reverse the order of the answer options, keeping their texts intact",
    direction="symmetric",
    relation_text=_OPTIONS_RELABELLED,
    apply=_apply_options_reverse,
)


# ── Mechanical invariant rho_inv (BBH geometric_shapes) ────────────────────────
#
# The paper's example of the "inv" family: the comparator computes the vertex
# count and path closure implied by the SVG path commands, and the ones implied
# by the shape the candidate names, straight from the problem data. No model call
# is involved on either side, so a false refutation would require a wrong entry in
# the table below — an offline, auditable bug rather than a run-time hallucination.

_SHAPE_VERTICES = {
    "triangle": 3,
    "quadrilateral": 4, "rectangle": 4, "square": 4, "kite": 4, "rhombus": 4,
    "trapezoid": 4,
    "pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8, "nonagon": 9,
    "decagon": 10,
}
# Shapes whose path is an open polyline rather than a closed polygon.
_SHAPE_OPEN_VERTICES = {"line": 2, "line segment": 2}


def _options_map(question: str) -> Dict[str, str]:
    """letter -> option body, for the multiple-choice block of a question."""
    out: Dict[str, str] = {}
    for line in question.splitlines():
        m = _OPTION_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _selected_option(question: str, answer: str) -> Optional[str]:
    """Resolve a candidate answer to the option body it selects."""
    letter = option_letter(answer)
    if letter is not None:
        body = _options_map(question).get(letter)
        if body is not None:
            return body.strip().lower()
    normalised = normalize_answer(answer)
    return normalised or None


def _polygon_walk(question: str) -> Optional[Tuple[int, bool]]:
    """(vertex count, closed?) for an M/L-only path, or None if not one."""
    parsed = _path_points(question)
    if parsed is None:
        return None
    _, pts = parsed
    walk: List[Tuple[float, float]] = []
    for p in pts:
        if not walk or walk[-1] != p:
            walk.append(p)
    if len(walk) < 2:
        return None
    closed = walk[0] == walk[-1]
    return (len(walk) - 1 if closed else len(walk)), closed


def _shape_invariant(question: str, candidate: str
                     ) -> Optional[Tuple[str, str, str]]:
    """Compare the vertex count the path draws with the one the answer names.

    Returns ``(expected, observed, detail)``: the first two are rendered in the
    same shape so that Compare is plain string equality — they differ exactly
    when the counts differ — and the third explains the check in the witness.

    Abstains (returns None) whenever the two sides are not directly comparable —
    an unparseable path, an answer naming a curved shape, or a polygon name
    against a path that does not close — so under-determination can never
    masquerade as a refutation.
    """
    walk = _polygon_walk(question)
    if walk is None:
        return None
    vertices, closed = walk
    name = _selected_option(question, candidate)
    if name is None:
        return None
    if name in _SHAPE_VERTICES:
        if not closed:
            return None
        n = _SHAPE_VERTICES[name]
        shape = "closed path"
    elif name in _SHAPE_OPEN_VERTICES:
        if closed:
            return None
        n = _SHAPE_OPEN_VERTICES[name]
        shape = "open path"
    else:
        return None                               # curved shape: not comparable
    return (f"{n} vertices in a {shape}",
            f"{vertices} vertices in a {shape}",
            f"the answer names a {name}, which has {n} vertices, while the SVG "
            f"path in the problem is a {shape} on {vertices} vertices")


REL_SHAPE_INVARIANT = Relation(
    name="shape_vertex_count",
    transformation="count the vertices the SVG path draws and the ones the named shape has",
    direction="symmetric",
    relation_text=("the number of vertices the path draws must equal the number the "
                   "named shape has"),
    family="inv",
    invariant=_shape_invariant,
)


# ── MGSM / word-problem relations ──────────────────────────────────────────────

_STANDALONE_NUM = re.compile(r"(?<![\w.,])\d+(?:\.\d+)?(?![\w.])")

# Spelled-out cardinals are quantities too: leaving them unscaled would silently
# turn the scaled problem into a *different* problem rather than a variant of the
# same one. "one" is excluded because it doubles as an article/pronoun ("one of
# the friends"), and hyphenated compounds ("twenty-one") are skipped by the word
# boundaries below rather than half-scaled.
_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_NUMBER_WORD_RE = re.compile(
    r"(?<![\w-])(" + "|".join(_NUMBER_WORDS) + r")(?![\w-])", re.IGNORECASE)


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _scale_quantities(question: str, factor: float) -> Optional[Variant]:
    """Multiply every quantity in the problem by c; rho: a' = c * a.

    Two factors are catalogued rather than one, which makes the relation
    self-checking: the pull-backs a'/2 and a'/3 agree only when the answer really
    is homogeneous of degree one in the scaled quantities, i.e. exactly when the
    relation is valid here. Where it is not — an answer that scales as c^2 (a
    price times a count) or as c^0 (a percentage) — the two pull-backs differ, no
    answer holds the orbit majority, and Remark 3's acceptance rule blocks the
    repair instead of acting on an invalid relation.
    """
    if not _STANDALONE_NUM.search(question):
        return None
    scaled = _STANDALONE_NUM.sub(
        lambda m: _fmt(float(m.group(0)) * factor), question)
    scaled = _NUMBER_WORD_RE.sub(
        lambda m: _fmt(_NUMBER_WORDS[m.group(1).lower()] * factor), scaled)
    if scaled == question:
        return None

    def _holds(a: str, a_prime: str) -> bool:
        na, nb = parse_number(a), parse_number(a_prime)
        if na is None or nb is None:
            return True          # not comparable → no violation is claimed
        return numbers_equal(nb, factor * na)

    def _pull(a_prime: str) -> Optional[str]:
        n = parse_number(a_prime)
        return None if n is None else _fmt(n / factor)

    def _fwd(a: str) -> Optional[str]:
        n = parse_number(a)
        return None if n is None else _fmt(n * factor)

    return Variant(
        relation=f"scale_quantities_x{_fmt(factor)}",
        relation_text=f"every quantity was multiplied by {_fmt(factor)}, so the answer "
                      f"must be exactly {_fmt(factor)} times the original answer",
        question=scaled,
        holds=_holds,
        family="mr",
        expected_for=_fwd,
        pullback=_pull,
        source="programmatic",
    )


# ── Backward substitution rho_bwd: slot enumeration + masking ──────────────────

@dataclass
class Slot:
    """One quantity stated in q, enumerated mechanically by the driver.

    Attributes:
        start, end: character span of the literal in the question.
        text: the literal as written.
        value: its numeric value, which phi requires the model to re-derive.
    """

    start: int
    end: int
    text: str
    value: float


def enumerate_slots(question: str) -> List[Slot]:
    """Enumerate the numeric literals stated in ``q``, left to right.

    This is the driver's mechanical stand-in for "the quantities stated in q":
    NextSlot walks this list so that successive backward checks mask *distinct*
    premises rather than re-rolling one check.
    """
    slots: List[Slot] = []
    for m in _STANDALONE_NUM.finditer(question):
        try:
            value = float(m.group(0))
        except ValueError:                        # pragma: no cover - regex-guarded
            continue
        slots.append(Slot(m.start(), m.end(), m.group(0), value))
    return slots


def _apply_mask_quantity(question: str, candidate: str,
                         slot_index: int = 0) -> Optional[Variant]:
    """Mask the ``slot_index``-th stated quantity for backward re-derivation.

    The instance carries only the *masked* problem; pi_mask supplies the
    candidate slot and the UNDETERMINED escape. phi holds iff the re-derived
    value equals the value that was masked out.
    """
    slots = enumerate_slots(question)
    if not slots or not candidate.strip():
        return None
    slot = slots[slot_index % len(slots)]
    masked = question[:slot.start] + "X" + question[slot.end:]

    def _holds(a: str, derived: str) -> bool:
        n = parse_number(derived)
        return True if n is None else numbers_equal(n, slot.value)

    return Variant(
        relation="mask_quantity",
        relation_text=(f"the hidden quantity X must re-derive to the value the "
                       f"problem states for it, {_fmt(slot.value)}"),
        question=masked,
        holds=_holds,
        family="bwd",
        expected=_fmt(slot.value),
        pullback=None,          # answers a different question, not q itself
        slot=slot.text,
        source="programmatic",
    )


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DEPENDENT_START = re.compile(
    r"^\s*(he|she|it|they|them|his|her|their|then|next|after|afterwards|finally|"
    r"so|therefore|thus|but|and|also|however|this|that|these|those|each|both|"
    r"the rest|the remainder)\b", re.IGNORECASE)


def _apply_permute_premises(question: str, candidate: str) -> Optional[Variant]:
    """Rotate the leading premise sentences, keeping the final question in place.

    Applied only when no premise opens with a pronoun or discourse connective —
    a cheap mechanical guard for the independence the relation presumes. When the
    guard fails the relation is skipped, never applied unsoundly.
    """
    parts = [p for p in _SENTENCE_SPLIT.split(question.strip()) if p.strip()]
    if len(parts) < 3:
        return None
    premises, tail = parts[:-1], parts[-1]
    if any(_DEPENDENT_START.match(p) for p in premises[1:]):
        return None
    rotated = premises[1:] + premises[:1]
    return equality_variant(
        REL_PERMUTE_PREMISES, " ".join(rotated + [tail]), source="programmatic")


_EN_STOPWORDS = {"the", "a", "an", "of", "and", "to", "in", "is", "are", "was",
                 "were", "how", "what", "many", "much", "does", "did", "she",
                 "he", "they", "for", "with", "each", "per", "if", "then"}


def _looks_english(text: str) -> bool:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in _EN_STOPWORDS)
    return hits >= 3 and hits / len(words) > 0.08


REL_SCALE_X2 = Relation(
    name="scale_quantities_x2",
    transformation="multiply every quantity in the problem by 2",
    direction="symmetric",
    relation_text="the answer must be exactly 2 times the original answer",
    apply=lambda q, a: _scale_quantities(q, 2.0),
)
REL_SCALE_X3 = Relation(
    name="scale_quantities_x3",
    transformation="multiply every quantity in the problem by 3",
    direction="symmetric",
    relation_text="the answer must be exactly 3 times the original answer",
    apply=lambda q, a: _scale_quantities(q, 3.0),
)
REL_MASK = Relation(
    name="mask_quantity",
    transformation="mask one stated quantity and re-derive it from the candidate answer",
    direction="increasing",
    relation_text="the masked quantity must re-derive exactly",
    family="bwd",
    apply_slot=_apply_mask_quantity,
)
REL_PERMUTE_PREMISES = Relation(
    name="permute_premises",
    transformation="reorder the independent premises of the problem",
    direction="symmetric",
    relation_text="the problem is unchanged, so the answer must be identical",
    apply=_apply_permute_premises,
)
REL_TRANSLATE_EN = Relation(
    name="translate_to_english",
    transformation=("translate the problem into English, keeping every quantity, "
                    "name and relation exactly as it is"),
    direction="increasing",
    relation_text="the problem is unchanged, so the answer must be identical",
    apply=None,                                   # needs pi_mr (0-1 model call)
    applicable=lambda q: not _looks_english(q),
)


# ── Code-reasoning relations ───────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _rewrite_code(question: str, transform: Callable[[ast.Module], ast.Module]
                  ) -> Optional[str]:
    m = _CODE_BLOCK_RE.search(question)
    if not m:
        return None
    try:
        tree = ast.parse(m.group(1))
        new_src = ast.unparse(ast.fix_missing_locations(transform(tree)))
    except (SyntaxError, ValueError, RecursionError, AttributeError):
        return None
    return question[:m.start(1)] + new_src + "\n" + question[m.end(1):]


class _LocalRenamer(ast.NodeTransformer):
    """Rename the parameters and local variables of every function, nothing else."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        locals_: List[str] = [a.arg for a in node.args.args]
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                locals_.append(sub.id)
        mapping = {n: f"v{i}_" for i, n in enumerate(dict.fromkeys(locals_))}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in mapping:
                sub.id = mapping[sub.id]
            elif isinstance(sub, ast.arg) and sub.arg in mapping:
                sub.arg = mapping[sub.arg]
        self.generic_visit(node)
        return node


class _DeadCodeInserter(ast.NodeTransformer):
    """Insert one unused assignment at the top of every function body."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        dead = ast.parse("_unused_fot = 0").body
        node.body = dead + node.body
        return node


def _apply_rename_identifiers(question: str, candidate: str) -> Optional[Variant]:
    src = _rewrite_code(question, lambda t: _LocalRenamer().visit(t))
    if src is None or src == question:
        return None
    return equality_variant(REL_RENAME, src, source="programmatic")


def _apply_insert_dead_code(question: str, candidate: str) -> Optional[Variant]:
    src = _rewrite_code(question, lambda t: _DeadCodeInserter().visit(t))
    if src is None or src == question:
        return None
    return equality_variant(REL_DEAD_CODE, src, source="programmatic")


REL_RENAME = Relation(
    name="rename_identifiers",
    transformation="rename the local variables and parameters of the program",
    direction="symmetric",
    relation_text="the program's behaviour is unchanged, so the answer must be identical",
    apply=_apply_rename_identifiers,
)
REL_DEAD_CODE = Relation(
    name="insert_dead_code",
    transformation="insert a statement whose value is never used",
    direction="symmetric",
    relation_text="the program's behaviour is unchanged, so the answer must be identical",
    apply=_apply_insert_dead_code,
)


# ── The library: task -> R_q ───────────────────────────────────────────────────
#
# ORDER IS PART OF THE DESIGN. Falsify draws one relation per attempt with Next()
# and returns on the *first* violation, so each library is sorted by its rung on
# the soundness ladder: mechanical invariants (no model call) first, then
# backward substitution, then semantics-preserving transforms (phi = equality),
# and covariant transforms (scaling, whose validity is conditional on
# homogeneity) last — they are only reached once the sounder checks abstain.

_CODE_RELATIONS = [REL_RENAME, REL_DEAD_CODE]
_OPTION_RELATIONS = [REL_OPTIONS_SHIFT1, REL_OPTIONS_REVERSE, REL_OPTIONS_SHIFT2]

CATALOGUES: Dict[str, List[Relation]] = {
    # §4.2's worked instantiation: R_q = {rho_bwd, (T_scale, phi_lin)}, plus the
    # two answer-preserving transforms that apply to word problems.
    "mgsm": [REL_MASK, REL_PERMUTE_PREMISES, REL_TRANSLATE_EN,
             REL_SCALE_X2, REL_SCALE_X3],
    "bigbenchhard:geometric_shapes": [REL_SHAPE_INVARIANT, REL_SVG_ROTATE,
                                      REL_SVG_TRANSLATE_SCALE, REL_SVG_REVERSE,
                                      REL_OPTIONS_SHIFT1],
    "bigbenchhard": _OPTION_RELATIONS,
    "cruxeval": _CODE_RELATIONS,
    "humaneval": _CODE_RELATIONS,
    "mbpp": _CODE_RELATIONS,
    "apps": _CODE_RELATIONS,
    "classeval": _CODE_RELATIONS,
    "programmingpuzzles": _CODE_RELATIONS,
}

# Tasks with no entry fall back to this: option relabelling applies to any
# multiple-choice query and is skipped (not misapplied) everywhere else, so an
# unregistered benchmark degrades to FoT == Solve rather than to a bad relation.
DEFAULT_CATALOGUE: List[Relation] = list(_OPTION_RELATIONS)


def catalogue_key(task: Optional[str], subtask: Optional[str] = None) -> str:
    """Build the library lookup key, e.g. 'bigbenchhard:geometric_shapes'."""
    base = (task or "").lower()
    return f"{base}:{subtask.lower()}" if subtask else base


def get_catalogue(task: Optional[str], subtask: Optional[str] = None,
                  names: Optional[List[str]] = None) -> List[Relation]:
    """Return R_q for a task, optionally restricted to the named relations.

    Lookup is most-specific-first (``task:subtask`` then ``task``), mirroring
    ``HasChecker``: the library is fixed per benchmark at construction time,
    never chosen at run time by the model. The returned order is the order in
    which ``Next(R_q)`` will draw from it.
    """
    base = (task or "").lower()
    cat = (CATALOGUES.get(catalogue_key(task, subtask))
           or CATALOGUES.get(base)
           or DEFAULT_CATALOGUE)
    if names:
        wanted = {n.strip().lower() for n in names if n.strip()}
        cat = [r for r in cat if r.name.lower() in wanted]
    return list(cat)


# ── pi_mr-gen: model-proposed catalogue (ablation only) ────────────────────────

_MR_LINE_RE = re.compile(
    r"^\s*MR\s*\d+\s*[:.)]\s*TRANSFORM\s*:\s*(.+?)\s*\|\s*RELATION\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE)

_ANSWER_PRESERVING_RE = re.compile(
    r"\b(unchanged|the same|identical|equal|same answer|not change|no change)\b",
    re.IGNORECASE)


def parse_generated_catalogue(text: str) -> List[Relation]:
    """Parse pi_mr-gen output into relations (the catalogue-construction ablation).

    Only answer-preserving proposals are kept: rho must be evaluable by the
    driver, and an arbitrary natural-language relation is not. A proposal whose
    stated relation is anything other than "the answer is unchanged" is dropped
    rather than guessed at.
    """
    relations: List[Relation] = []
    for i, (transform, rho) in enumerate(_MR_LINE_RE.findall(text), start=1):
        if not _ANSWER_PRESERVING_RE.search(rho):
            continue
        relations.append(Relation(
            name=f"generated_{i}",
            transformation=transform.strip(),
            direction="symmetric",
            relation_text=rho.strip(),
            apply=None,
        ))
    return relations
