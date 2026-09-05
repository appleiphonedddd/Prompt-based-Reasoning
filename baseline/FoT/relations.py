"""
The relation catalogue C for Falsification-of-Thought (FoT), §2.6 of the paper.

A *metamorphic relation* (Definition 1) is a pair ``R = (T, rho)`` where
``T : Q -> Q`` is an input transformation and ``rho`` is an output relation. ``R``
is *valid* for a task if ``rho(a*(q), a*(T(q)))`` holds for every query q of that
task, writing ``a*(.)`` for the ground-truth answer. Validity makes ``rho`` a
necessary property of the intended functionality — the role played by the
model-invented necessary conditions in the pilot, except that ``rho`` is fixed in
advance and inspected once by a human instead of being re-invented per query.

Many relations are answer-preserving (``rho`` is equality); others are
answer-transforming with a known map ``g``, so that ``rho(a, a') <=> a' = g(a)``.
Each :class:`Variant` therefore carries both ``g`` (``expected_for``, used to
render the witness) and ``g^-1`` (``pullback``, used to pull the variant's answer
back into the coordinate frame of q, where it joins the orbit O).

Soundness (Proposition 1): if ``R`` is valid and ``not rho(a, a')``, then ``a``
or ``a'`` is wrong — as a matter of logic, with no appeal to the model's opinion
of its own work. The residual unsoundness is confined to (i) an invalid relation
in this file and (ii) attribution (the defect may sit at ``a'``). Item (i) is a
design-time property of the small table below; **this module is the artifact to
audit**.

Two properties are required of every entry (Remark 2): the relation must be
valid, and its image must be answered at least as reliably as its source, which
is recorded in ``Relation.direction``. Distractor insertion is a valid relation
but is *excluded* on the second ground.

Most transformations here are purely programmatic (rescaling numbers, masking a
literal, permuting sentences or answer options, applying an affine map to an SVG
path, renaming identifiers), so they cost no model call and leave no room for
hallucination in ``T`` itself. Only paraphrase-style transformations (``apply is
None``) are delegated to the model through the ``pi_mr`` template; those are all
answer-preserving *and* mechanically validated afterwards to preserve every
numeric literal of the original (``preserve_numbers``).

Audit notes on the entries below:

  * ``scale_quantities`` (MGSM) is valid only when the answer is homogeneous of
    degree one in the scaled quantities. It is *not* valid when a problem
    multiplies two scaled quantities together (a price times a count scales as
    ``c^2``), so it can manufacture a spurious violation. Two factors are
    catalogued rather than one, which makes the relation self-checking: the
    pull-backs ``a'/2`` and ``a'/3`` land on the same orbit member exactly when
    the answer really is homogeneous of degree one, so where the relation does
    not hold the orbit fragments, no answer takes the majority and Remark 3's
    acceptance rule blocks the repair. Drop it entirely with ``--fot_relations``
    if a stricter catalogue is wanted.
  * The ``svg_*`` entries (geometric_shapes) are all answer-preserving, and all
    of them canonicalise the path first — the BBH generator emits one subpath per
    edge, so the encoding repeats every interior vertex and the vertex count,
    which *is* the answer, cannot be read off. Collapsing that is answer-
    preserving and strictly easier, which is why it leads the catalogue and is
    composed into the others. What was removed on Remark 2 grounds is the
    generic rotation (37 degrees) and the non-integral rescale (1.5x) the
    catalogue used to open with: both are valid relations but *decreasing* ones,
    since they tilt an axis-aligned figure off axis and replace the dataset's
    2-decimal grid with fresh decimals, so their variants are answered less
    reliably than the source and a disagreement indicts the variant. Only exact
    isometries of the grid are kept (integer translation, mirror about the
    bounding box mid-line, quarter turn).
  * ``mask_quantity`` is the backward-verification entry of §2.6: masking a
    quantity and requiring the candidate answer to recover it is an
    answer-transforming relation whose follow-up question is written by a fixed
    template. It is the one relation whose variant deliberately contains the
    candidate — that is what backward substitution *is* (recover a masked premise
    *from* the conclusion). Remark 1's independence requirement applies to the
    relations whose variant re-asks the *same* question, where showing the
    candidate would make the follow-up an echo; here the candidate is the input
    of a different question. Its answer lives in a different space, so it
    contributes a violation but no orbit member (``pullback is None``).
"""

from __future__ import annotations

import ast
import datetime
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
    """rho for answer-preserving relations: numeric when both parse, else textual."""
    la, lb = option_letter(a), option_letter(b)
    if la and lb:
        return la == lb
    na, nb = parse_number(a), parse_number(b)
    if na is not None and nb is not None:
        return numbers_equal(na, nb)
    return normalize_answer(a) == normalize_answer(b)


def answer_key(answer: str) -> str:
    """Canonical key for grouping answers, consistent with :func:`answers_equal`.

    The driver counts orbit members with it (Majority(O) in Algorithm 2, line 16),
    so two answers that ``answers_equal`` considers the same must land on the same
    key: an option letter first, then a numeric value, then normalised text.
    """
    letter = option_letter(answer)
    if letter is not None:
        return f"({letter})"
    number = parse_number(answer)
    if number is not None:
        return _fmt(number)
    return normalize_answer(answer)


_ANY_NUM = re.compile(r"\d+(?:\.\d+)?")


def numeric_literals(text: str) -> List[str]:
    """The multiset of numeric literals in a text, normalised for comparison.

    Used to validate model-instantiated transforms: a textual transform must be
    mechanically validated to preserve every numeric literal it is required to
    preserve, so a paraphrase that quietly drops or invents a quantity is rejected
    rather than checked against a relation it no longer satisfies.
    """
    return sorted(_fmt(float(m.group(0))) for m in _ANY_NUM.finditer(text))


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Variant:
    """One realised member of the orbit: ``q' = T(q)`` plus how to judge its answer.

    Attributes:
        relation: name of the relation R that produced this variant.
        relation_text: rho, written out for the witness and the repair prompt.
        question: the variant q' handed to SOLVE. Never carries the candidate,
            except for ``mask_quantity`` (see the module docstring).
        holds: rho — evaluated by the driver, never by the model. Called as
            ``holds(a, a_prime)``.
        expected_for: g, i.e. rho(a): the answer this variant *must* receive given
            the candidate. Rendered into the witness; None when not determined.
        pullback: g^-1, mapping the variant's answer back into q's frame so it can
            join the orbit O. None when the variant answers a *different* question
            (backward substitution), in which case it contributes no orbit member.
        slot: for ``mask_quantity`` — the stated quantity that was masked.
        source: "programmatic" (T ran as code) or "model" (T came from pi_mr).
    """

    relation: str
    relation_text: str
    question: str
    holds: Callable[[str, str], bool]
    expected_for: Optional[Callable[[str], Optional[str]]] = None
    pullback: Optional[Callable[[str], Optional[str]]] = None
    slot: Optional[str] = None
    source: str = "programmatic"

    def expected_value(self, answer: str) -> Optional[str]:
        """g(a): what rho requires this variant's answer to be, given ``a``."""
        return self.expected_for(answer) if self.expected_for is not None else None


@dataclass
class Relation:
    """A catalogue entry R = (T, rho).

    Attributes:
        name: stable identifier, usable with ``--fot_relations``.
        transformation: one-line description of T. Doubles as the instruction
            handed to pi_mr when T cannot be applied programmatically.
        direction: "symmetric" or "increasing" — the reliability requirement of
            Remark 2. Every entry must be non-decreasing in reliability.
        relation_text: rho, in words.
        apply: programmatic T. Returns a Variant, or None when the relation does
            not apply to this particular query (which is not a violation — the
            relation is simply skipped). Takes the candidate and a draw index,
            which only backward substitution uses (to mask a different quantity on
            successive rounds). When None, T is delegated to the model via pi_mr
            and rho is equality.
        applicable: optional guard for model-applied relations.
        preserve_numbers: for model-applied T only — reject the model's variant
            unless it carries exactly the numeric literals of the original.
    """

    name: str
    transformation: str
    direction: str
    relation_text: str
    apply: Optional[Callable[..., Optional[Variant]]] = None
    applicable: Optional[Callable[[str], bool]] = None
    preserve_numbers: bool = True

    @property
    def programmatic(self) -> bool:
        """True iff T costs no model call to apply."""
        return self.apply is not None


def equality_variant(relation: Relation, question: str,
                     source: str = "model") -> Variant:
    """Build an answer-preserving variant (rho = equality, g = g^-1 = id)."""
    return Variant(
        relation=relation.name,
        relation_text=relation.relation_text,
        question=question,
        holds=answers_equal,
        expected_for=lambda a: a,
        pullback=lambda a_prime: a_prime,
        source=source,
    )


# ── SVG path relations (BBH geometric_shapes) ──────────────────────────────────
#
# The answer to a geometric_shapes query is a combinatorial property of the
# path's vertex walk (how many distinct vertices, and their metric arrangement
# for kite/trapezoid/rectangle). Two consequences shape this block:
#
#   * The BBH encoding is *redundant*: the generator emits one subpath per edge,
#     "M v1 L v2 M v2 L v3 M v3 L v4 ...", so a hexagon is written as eleven
#     coordinate pairs. Collapsing that into the single polyline it denotes is
#     answer-preserving (the drawing is identical) and strictly easier to read —
#     it is the one transformation here that is genuinely *increasing* in
#     reliability, so it leads the catalogue and is composed into every other
#     entry (the composite stays valid because canonicalisation is itself a
#     valid answer-preserving relation).
#   * Every other transformation must be an *exact* isometry on the 2-decimal
#     coordinate grid. A generic rotation or a non-integral rescale is a valid
#     relation but a *decreasing* one: it tilts an axis-aligned rectangle and
#     litters the path with fresh decimals, so the variant is answered less
#     reliably than the source and a disagreement indicts the variant rather
#     than the candidate. Remark 2 excludes it, exactly as it excludes distractor
#     insertion. The isometries kept below (integer translation, mirror about the
#     bounding box's vertical mid-line, quarter-turn about the bounding box
#     centre) land back on the same grid and preserve axis-alignment.
#
# Composing canonicalisation into every entry makes the drawn relations partly
# correlated, which is deliberate rather than an oversight: Remark 3's majority
# rule then reads "the canonical form's verdict outvotes the redundant encoding's",
# which is precisely the reliability ordering Remark 2 asks the catalogue to
# encode. Where the query is already canonical (141 of the 250 questions carry no
# redundant subpath) the entries reduce to independent isometries and the rule
# reverts to ordinary corroboration.
#
# Paths are parsed into (command, args) segments rather than a flat coordinate
# list, so arcs are transformed correctly instead of being skipped: "A rx,ry phi
# fa,fs x,y" translates by moving only its endpoint, and rotates by turning its
# endpoint and adding to phi. That matters for coverage — the sector and ellipse
# questions (52 of 250) are exactly the arc-bearing ones, and under a
# coordinate-list parser they had no geometric relation at all.

_PATH_RE = re.compile(r'd\s*=\s*"([^"]*)"')
_TOKEN_RE = re.compile(r"[A-Za-z]|-?\d+(?:\.\d+)?")
_ARGC = {"M": 2, "L": 2, "A": 7}

Segment = Tuple[str, List[float]]


def _parse_path(question: str) -> Optional[Tuple[str, List[Segment]]]:
    """Return the raw ``d`` attribute and its segments, for M/L/A paths.

    Anything else — a curve, a close-path, a relative command — makes the whole
    relation inapplicable rather than misapplied: an affine map on the raw
    numbers would corrupt a cubic's control points or an arc's radii, which
    would make the relation invalid instead of merely unavailable.
    """
    m = _PATH_RE.search(question)
    if not m:
        return None
    d = m.group(1)
    tokens = _TOKEN_RE.findall(d)
    if "".join(tokens) != re.sub(r"[\s,]", "", d):
        return None                       # something in d we did not tokenise
    segments: List[Segment] = []
    cmd: Optional[str] = None
    i = 0
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
        if cmd not in _ARGC:              # unsupported or leading numbers
            return None
        n = _ARGC[cmd]
        args = tokens[i:i + n]
        if len(args) < n or any(a.isalpha() for a in args):
            return None
        segments.append((cmd, [float(a) for a in args]))
        i += n
        if cmd == "M":
            cmd = "L"                     # SVG: numbers after a moveto are linetos
    return (d, segments) if segments else None


def _fmt_coord(value: float) -> str:
    """2 decimals, as the dataset writes them — and never a signed zero."""
    return f"{0.0 if abs(value) < 5e-3 else value:.2f}"


def _emit_path(segments: List[Segment]) -> str:
    """Render segments back in the dataset's own notation."""
    out: List[str] = []
    for cmd, a in segments:
        if cmd == "A":
            out.append(f"A {_fmt_coord(a[0])},{_fmt_coord(a[1])} {_fmt_coord(a[2])} "
                       f"{int(a[3])},{int(a[4])} {_fmt_coord(a[5])},{_fmt_coord(a[6])}")
        else:
            out.append(f"{cmd} {_fmt_coord(a[0])},{_fmt_coord(a[1])}")
    return " ".join(out)


def _seg_endpoint(segment: Segment) -> Tuple[float, float]:
    cmd, a = segment
    return (a[5], a[6]) if cmd == "A" else (a[0], a[1])


def _same_point(p: Tuple[float, float], q: Tuple[float, float]) -> bool:
    return abs(p[0] - q[0]) < 5e-3 and abs(p[1] - q[1]) < 5e-3


def _canonicalise(segments: List[Segment]) -> List[Segment]:
    """Collapse the per-edge subpaths into the single walk they denote.

    Drops a ``M p`` whose target is where the pen already is, and a zero-length
    ``L``. Both are no-ops for the rendered figure, so the transformation is
    answer-preserving; what it removes is the duplicated coordinate that makes
    the vertex count hard to read off.
    """
    out: List[Segment] = []
    cur: Optional[Tuple[float, float]] = None
    for cmd, a in segments:
        point = _seg_endpoint((cmd, a))
        if cmd in ("M", "L") and cur is not None and _same_point(cur, point):
            continue
        out.append((cmd, list(a)))
        cur = point
    return out


def _has_arc(segments: List[Segment]) -> bool:
    return any(cmd == "A" for cmd, _ in segments)


def _bbox(segments: List[Segment]) -> Tuple[float, float, float, float]:
    xs = [_seg_endpoint(s)[0] for s in segments]
    ys = [_seg_endpoint(s)[1] for s in segments]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_centre(segments: List[Segment]) -> Tuple[float, float]:
    """Centre of the endpoint bounding box, snapped to the 2-decimal grid.

    Snapping keeps every rotated coordinate on the same grid the dataset uses,
    so the quarter-turn is exact rather than a rounding of an exact map.
    """
    lo_x, lo_y, hi_x, hi_y = _bbox(segments)
    return (round((lo_x + hi_x) / 2, 2), round((lo_y + hi_y) / 2, 2))


def _in_frame_offset(lo: float, hi: float, step: float = 5.0,
                     frame: float = 100.0) -> float:
    """A +/-``step`` shift that keeps ``[lo, hi]`` inside ``[0, frame]``, else 0."""
    if hi + step <= frame:
        return step
    if lo - step >= 0.0:
        return -step
    return 0.0


def _reanchor(target_lo: float, lo: float, hi: float,
              frame: float = 100.0) -> float:
    """Shift ``[lo, hi]`` back towards ``target_lo`` without leaving ``[0, frame]``."""
    placed = min(max(target_lo, 0.0), max(0.0, frame - (hi - lo)))
    return round(placed - lo, 2)


def _map_segments(segments: List[Segment],
                  point: Callable[[float, float], Tuple[float, float]],
                  phi: float = 0.0) -> List[Segment]:
    """Apply an isometry to every endpoint; ``phi`` is added to arc rotations."""
    out: List[Segment] = []
    for cmd, a in segments:
        if cmd == "A":
            x, y = point(a[5], a[6])
            out.append(("A", [a[0], a[1], (a[2] + phi) % 360.0, a[3], a[4], x, y]))
        else:
            out.append((cmd, list(point(a[0], a[1]))))
    return out


def _svg_variant(relation: Relation, question: str, d: str,
                 segments: List[Segment]) -> Optional[Variant]:
    """Wrap rewritten segments as an answer-preserving variant, or skip a no-op."""
    new_d = _emit_path(segments)
    if new_d == d.strip():
        return None
    return equality_variant(relation, question.replace(f'd="{d}"', f'd="{new_d}"', 1),
                            source="programmatic")


def _apply_canonicalise(question: str, candidate: str = "",
                        index: int = 0) -> Optional[Variant]:
    parsed = _parse_path(question)
    if parsed is None:
        return None
    d, segments = parsed
    return _svg_variant(REL_SVG_CANONICALISE, question, d, _canonicalise(segments))


def _apply_reverse(question: str, candidate: str = "",
                   index: int = 0) -> Optional[Variant]:
    """Reverse the traversal order of the vertices.

    Restricted to arc-free paths: reversing an arc also requires flipping its
    sweep flag, and a relation that is only nearly right is worse than one that
    is unavailable.
    """
    parsed = _parse_path(question)
    if parsed is None:
        return None
    d, segments = parsed
    walk = _canonicalise(segments)
    if _has_arc(walk) or len(walk) < 3:
        return None
    points = [_seg_endpoint(s) for s in walk][::-1]
    rev: List[Segment] = [("M", list(points[0]))]
    rev += [("L", list(p)) for p in points[1:]]
    return _svg_variant(REL_SVG_REVERSE, question, d, rev)


def _apply_translate(question: str, candidate: str = "",
                     index: int = 0) -> Optional[Variant]:
    """Translate by an integer offset — exact on the 2-decimal grid, arcs included.

    The offset is flipped where it would push the figure out of the dataset's own
    0-100 frame: an isometry is only reliability-neutral if its image still looks
    like a query from this task.
    """
    parsed = _parse_path(question)
    if parsed is None:
        return None
    d, segments = parsed
    walk = _canonicalise(segments)
    lo_x, lo_y, hi_x, hi_y = _bbox(walk)
    dx, dy = _in_frame_offset(lo_x, hi_x), _in_frame_offset(lo_y, hi_y)
    if dx == 0.0 and dy == 0.0:
        return None
    return _svg_variant(REL_SVG_TRANSLATE, question, d,
                        _map_segments(walk, lambda x, y: (x + dx, y + dy)))


def _apply_reflect(question: str, candidate: str = "",
                   index: int = 0) -> Optional[Variant]:
    """Mirror about the bounding box's vertical mid-line.

    Every shape class in this task's option lists is closed under reflection, so
    rho is equality. Arc-free only: a mirrored elliptical arc also needs its
    sweep flag flipped and its x-axis rotation negated.
    """
    parsed = _parse_path(question)
    if parsed is None:
        return None
    d, segments = parsed
    walk = _canonicalise(segments)
    if _has_arc(walk) or len(walk) < 2:
        return None
    axis = _bbox_centre(walk)[0]
    return _svg_variant(REL_SVG_REFLECT, question, d,
                        _map_segments(walk, lambda x, y: (2 * axis - x, y)))


def _apply_rotate90(question: str, candidate: str = "",
                    index: int = 0) -> Optional[Variant]:
    """Quarter-turn about the bounding box centre.

    A quarter-turn is the one rotation that keeps the path on the coordinate
    grid *and* keeps an axis-aligned figure axis-aligned, so it costs the reader
    nothing — unlike the generic rotation Remark 2 rules out.
    """
    parsed = _parse_path(question)
    if parsed is None:
        return None
    d, segments = parsed
    walk = _canonicalise(segments)
    cx, cy = _bbox_centre(walk)
    turned = _map_segments(walk, lambda x, y: (cx - (y - cy), cy + (x - cx)),
                           phi=90.0)
    # Re-anchor the bounding box inside the frame: a quarter turn makes a wide
    # figure tall, which walks off the top. The correcting translation is exact
    # on the grid (both corners are), so the composite is still an isometry.
    lo_x, lo_y, _, _ = _bbox(walk)
    new_lo_x, new_lo_y, new_hi_x, new_hi_y = _bbox(turned)
    shift_x = _reanchor(lo_x, new_lo_x, new_hi_x)
    shift_y = _reanchor(lo_y, new_lo_y, new_hi_y)
    turned = _map_segments(turned, lambda x, y: (x + shift_x, y + shift_y))
    return _svg_variant(REL_SVG_ROTATE90, question, d, turned)


_SHAPE_UNCHANGED = "the figure is unchanged, so the answer must be identical"

REL_SVG_CANONICALISE = Relation(
    name="svg_canonicalise",
    transformation="rewrite the per-edge subpaths as the single polyline they denote",
    direction="increasing",
    relation_text=_SHAPE_UNCHANGED,
    apply=_apply_canonicalise,
)
REL_SVG_REVERSE = Relation(
    name="svg_reverse",
    transformation="reverse the order in which the vertices of the path are traversed",
    direction="increasing",
    relation_text=_SHAPE_UNCHANGED,
    apply=_apply_reverse,
)
REL_SVG_TRANSLATE = Relation(
    name="svg_translate",
    transformation="translate every point of the path by a fixed integer offset",
    direction="symmetric",
    relation_text=_SHAPE_UNCHANGED,
    apply=_apply_translate,
)
REL_SVG_REFLECT = Relation(
    name="svg_reflect",
    transformation="mirror the path about the vertical mid-line of its bounding box",
    direction="symmetric",
    relation_text=_SHAPE_UNCHANGED,
    apply=_apply_reflect,
)
REL_SVG_ROTATE90 = Relation(
    name="svg_rotate90",
    transformation="rotate the path by a quarter turn about the centre of its bounding box",
    direction="symmetric",
    relation_text=_SHAPE_UNCHANGED,
    apply=_apply_rotate90,
)


# ── Multiple-choice option permutation (BBH letter-answer tasks) ────────────────

_OPTION_RE = re.compile(r"^\s*\(([A-Z])\)\s*(.*\S)\s*$")


def _option_block(question: str, min_options: int = 2
                  ) -> Optional[Tuple[List[str], List[int], List[str], List[str]]]:
    """Locate a contiguous, uniquely-lettered option block: lines, indices, letters, bodies."""
    lines = question.splitlines()
    idx = [i for i, ln in enumerate(lines) if _OPTION_RE.match(ln)]
    if len(idx) < min_options or idx != list(range(idx[0], idx[0] + len(idx))):
        return None
    parsed = [_OPTION_RE.match(lines[i]) for i in idx]
    letters = [m.group(1) for m in parsed]           # type: ignore[union-attr]
    bodies = [m.group(2) for m in parsed]            # type: ignore[union-attr]
    if len(set(letters)) != len(letters):
        return None
    return lines, idx, letters, bodies


def _permute_options(question: str, name: str, relation_text: str,
                     perm: Callable[[int, int], int], min_options: int = 2
                     ) -> Optional[Variant]:
    """Relabel the option bodies among the option letters.

    Answer-transforming with a known map: the variant's letter at position i
    carries the body that sat at position ``perm(i, L)`` in the original, so
    ``g^-1`` sends that letter back to the original one. A textbook metamorphic
    relation for multiple choice, and entirely programmatic.
    """
    block = _option_block(question, min_options)
    if block is None:
        return None
    lines, idx, letters, bodies = block
    size = len(letters)

    new_lines = list(lines)
    for pos, i in enumerate(idx):
        new_lines[i] = f"({letters[pos]}) {bodies[perm(pos, size)]}"
    # letter in the variant -> letter in the original
    back = {letters[pos]: letters[perm(pos, size)] for pos in range(size)}
    # letter in the original -> letter in the variant (g, for the witness)
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
        expected_for=_fwd,
        pullback=_pull,
        source="programmatic",
    )


_OPTIONS_RELABELLED = ("the options were relabelled: the chosen letter must denote "
                       "the same option text as before")


def _apply_options_shift1(question: str, candidate: str = "",
                          index: int = 0) -> Optional[Variant]:
    return _permute_options(question, "options_shift1", _OPTIONS_RELABELLED,
                            lambda i, n: (i - 1) % n)


def _apply_options_shift2(question: str, candidate: str = "",
                          index: int = 0) -> Optional[Variant]:
    return _permute_options(question, "options_shift2", _OPTIONS_RELABELLED,
                            lambda i, n: (i - 2) % n, min_options=3)


def _apply_options_reverse(question: str, candidate: str = "",
                           index: int = 0) -> Optional[Variant]:
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


# ── Date relations (BBH date_understanding) ────────────────────────────────────
#
# Every one of the 250 queries has the same shape: a premise that fixes "today"
# (often indirectly — "Christmas Eve of 1937", "her 5th visit", "the last day of
# the first quarter of 2008"), a requested offset, and six MM/DD/YYYY options.
# Two mechanical attacks follow from that shape, and neither needs a model call:
#
#   * **Reformatting the options.** MM/DD/YYYY is ambiguous — one of the task's
#     own question templates is built on exactly that ambiguity ("In the UK,
#     people usually put the day before the month") — so writing the options in a
#     form that cannot be misread is answer-preserving and strictly easier. These
#     are the reliability-*increasing* entries, and they lead the catalogue.
#   * **Relabelling the timeline.** Shifting every year by a multiple of 28 maps
#     the Gregorian calendar onto itself: 28 years is 1461 whole weeks, so the
#     day of the week is preserved, and the leap pattern is preserved because the
#     shift is a multiple of 4. Shifting the option years by the same amount
#     leaves the answer's *letter* unchanged, so rho stays equality. The one
#     thing that breaks the isomorphism is a century year that is not a leap year
#     (1900, 2100), so the shift is refused whenever the span crosses one — the
#     relation becomes unavailable rather than invalid.
#
# What is deliberately *not* catalogued: rewriting the options into DD/MM/YYYY
# (valid, but Remark 2 excludes it — it is the ambiguous direction), and having
# the model restate the premise with "today" given explicitly (that would require
# the falsifier to solve the anchor, and a wrong anchor silently changes the
# question rather than transforming it).
#
# Measured on 30 questions with qwen2.5:32b (Solve accuracy 90%), none of these
# relations turned out to be reliability-*increasing*: the deltas against the
# source were -3, -3, 0 and -9 points, every relation agreed with the source on
# about 90% of queries, and where a variant disagreed it was right about half the
# time. Two consequences are recorded here rather than papered over. First, the
# entries are labelled "symmetric": the reformatting relations were catalogued on
# the theory that removing the MM/DD ambiguity would help, and the measurement did
# not bear that out. Second, FoT has little to offer on this task — a repair
# fires on a coin flip, and an offline sweep over n in 2..5 and every tau was flat
# at the Solve baseline. The reason is visible in the residual errors: they are
# anchor misreadings ("Yesterday, Jan 21, 2011" read as "today is Jan 21"), and an
# anchor misreading is invariant under every transformation above, so the model
# reproduces it in each variant and the orbit corroborates the wrong answer. That
# is the attribution residual of Proposition 1 in its starkest form: self-
# refutation cannot see an error the model makes identically everywhere. What the
# catalogue does buy is that the falsifier is no longer inert — the inherited
# default (three option permutations) fired on 0 of 30 queries.

_MDY_RE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")
_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|2[01]\d{2})\b")
_ASK_FORMAT = "in MM/DD/YYYY"
_MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December")

_DATES_SAME_DAY = ("the options denote the same days, only written differently, "
                   "so the chosen letter must be the same")
_DATES_SHIFTED = ("every year in the problem and in the options moved by the same "
                  "whole number of 28-year calendar cycles, so the chosen letter "
                  "must be the same")


def _rewrite_option_bodies(question: str,
                           render: Callable[[str], Optional[str]]) -> Optional[str]:
    """Rewrite every option body through ``render``, or skip the relation.

    Skips rather than half-applies: if one body is not a date this relation does
    not fit the query, and if the rewrite would merge two options into one it is
    no longer answer-preserving.
    """
    block = _option_block(question)
    if block is None:
        return None
    lines, idx, letters, bodies = block
    rendered = [render(b) for b in bodies]
    if any(b is None for b in rendered) or rendered == bodies:
        return None
    if len(set(rendered)) != len(set(bodies)):
        return None
    out = list(lines)
    for pos, i in enumerate(idx):
        out[i] = f"({letters[pos]}) {rendered[pos]}"
    return "\n".join(out)


def _parse_mdy(body: str) -> Optional[Tuple[int, int, int]]:
    """Parse an MM/DD/YYYY option body into (year, month, day), or None."""
    m = _MDY_RE.fullmatch(body.strip())
    if not m:
        return None
    month, day, year = (int(g) for g in m.groups())
    try:
        datetime.date(year, month, day)
    except ValueError:
        return None                       # not a real calendar date
    return year, month, day


def _apply_dates_spell_out(question: str, candidate: str = "",
                           index: int = 0) -> Optional[Variant]:
    """Write the options as "December 25, 1937" — a form that cannot be misread."""
    def render(body: str) -> Optional[str]:
        parsed = _parse_mdy(body)
        if parsed is None:
            return None
        year, month, day = parsed
        return f"{_MONTH_NAMES[month - 1]} {day}, {year}"

    rewritten = _rewrite_option_bodies(question, render)
    if rewritten is None or _ASK_FORMAT not in rewritten:
        return None
    return equality_variant(REL_DATES_SPELL_OUT,
                            rewritten.replace(f" {_ASK_FORMAT}", "", 1),
                            source="programmatic")


def _apply_dates_iso(question: str, candidate: str = "",
                     index: int = 0) -> Optional[Variant]:
    """Write the options as ISO 8601 — the other unambiguous ordering."""
    def render(body: str) -> Optional[str]:
        parsed = _parse_mdy(body)
        return None if parsed is None else "%04d-%02d-%02d" % parsed

    rewritten = _rewrite_option_bodies(question, render)
    if rewritten is None or _ASK_FORMAT not in rewritten:
        return None
    return equality_variant(REL_DATES_ISO,
                            rewritten.replace(_ASK_FORMAT, "in YYYY-MM-DD", 1),
                            source="programmatic")


def _crosses_common_century(lo: int, hi: int) -> bool:
    """True if [lo, hi] contains a century year that is not a leap year."""
    return any(year % 400 != 0
               for year in range(lo + (-lo) % 100, hi + 1, 100))


def _shift_years(question: str, shift: int) -> Optional[str]:
    """Relabel every year by ``shift``, or None when the calendar is not preserved."""
    body = question.split("\nOptions:", 1)[0]
    body_years = [int(y) for y in _YEAR_RE.findall(body)]
    all_years = [int(y) for y in _YEAR_RE.findall(question)]
    if not body_years or not all_years:
        return None
    # The answer's own year may sit a year outside the stated span, so pad before
    # asking whether the shift keeps the calendar isomorphic.
    lo = min(min(all_years), min(all_years) + shift) - 2
    hi = max(max(all_years), max(all_years) + shift) + 2
    if _crosses_common_century(lo, hi):
        return None
    shifted = _YEAR_RE.sub(lambda m: str(int(m.group(0)) + shift), question)
    # Every date the variant mentions must still be a real calendar date.
    for month, day, year in _MDY_RE.findall(shifted):
        try:
            datetime.date(int(year), int(month), int(day))
        except ValueError:
            return None
    return shifted


def _apply_dates_shift_back(question: str, candidate: str = "",
                            index: int = 0) -> Optional[Variant]:
    shifted = _shift_years(question, -28)
    if shifted is None:
        return None
    return equality_variant(REL_DATES_SHIFT_BACK, shifted, source="programmatic")


def _apply_dates_shift_forward(question: str, candidate: str = "",
                               index: int = 0) -> Optional[Variant]:
    shifted = _shift_years(question, 28)
    if shifted is None:
        return None
    return equality_variant(REL_DATES_SHIFT_FORWARD, shifted, source="programmatic")


REL_DATES_SPELL_OUT = Relation(
    name="dates_spell_out_options",
    transformation="write every answer option as a spelled-out date such as "
                   "'December 25, 1937'",
    direction="symmetric",
    relation_text=_DATES_SAME_DAY,
    apply=_apply_dates_spell_out,
)
REL_DATES_ISO = Relation(
    name="dates_iso_options",
    transformation="write every answer option in ISO 8601 form, YYYY-MM-DD",
    direction="symmetric",
    relation_text=_DATES_SAME_DAY,
    apply=_apply_dates_iso,
)
REL_DATES_SHIFT_BACK = Relation(
    name="dates_shift_years_back28",
    transformation="move every year in the problem and its options back by 28 years",
    direction="symmetric",
    relation_text=_DATES_SHIFTED,
    apply=_apply_dates_shift_back,
)
REL_DATES_SHIFT_FORWARD = Relation(
    name="dates_shift_years_28",
    transformation="move every year in the problem and its options forward by 28 years",
    direction="symmetric",
    relation_text=_DATES_SHIFTED,
    apply=_apply_dates_shift_forward,
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
    """Multiply every quantity in the problem by c; rho: a' = c * a."""
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
        expected_for=_fwd,
        pullback=_pull,
        source="programmatic",
    )


# ── Backward substitution: mask a quantity and re-derive it from a ─────────────

@dataclass
class Slot:
    """One quantity stated in q, enumerated mechanically by the driver.

    Attributes:
        start, end: character span of the literal in the question.
        text: the literal as written.
        value: its numeric value, which rho requires the variant to recover.
    """

    start: int
    end: int
    text: str
    value: float


def enumerate_slots(question: str) -> List[Slot]:
    """Enumerate the numeric literals stated in ``q``, left to right.

    This is the driver's mechanical stand-in for "the quantities stated in q":
    successive draws of ``mask_quantity`` walk this list so that a repaired
    candidate is attacked on a *different* premise.
    """
    slots: List[Slot] = []
    for m in _STANDALONE_NUM.finditer(question):
        try:
            value = float(m.group(0))
        except ValueError:                        # pragma: no cover - regex-guarded
            continue
        slots.append(Slot(m.start(), m.end(), m.group(0), value))
    return slots


# The fixed template that turns "mask the i-th quantity" into a self-contained
# follow-up question (§2.6). It is solved by pi_solve like any other variant; the
# candidate appears because it is the *input* of this question, not a suggestion.
_MASK_TEMPLATE = (
    "In the problem below, one stated quantity has been replaced by X.\n"
    "It is given that the final answer to the problem is: {answer}\n"
    "Working from that answer and the remaining information, determine the "
    "value of X.\n\n"
    "Problem (one quantity hidden as X):\n{masked}"
)


def _apply_mask_quantity(question: str, candidate: str = "",
                         index: int = 0) -> Optional[Variant]:
    """Mask the ``index``-th stated quantity; rho: the recovered value = original."""
    slots = enumerate_slots(question)
    if not slots or not candidate.strip():
        return None
    slot = slots[index % len(slots)]
    masked = question[:slot.start] + "X" + question[slot.end:]

    def _holds(a: str, recovered: str) -> bool:
        n = parse_number(recovered)
        return True if n is None else numbers_equal(n, slot.value)

    return Variant(
        relation="mask_quantity",
        relation_text=(f"the hidden quantity X must come back out as the value the "
                       f"problem states for it, {_fmt(slot.value)}"),
        question=_MASK_TEMPLATE.format(answer=candidate.strip(), masked=masked),
        holds=_holds,
        expected_for=lambda a, _v=_fmt(slot.value): _v,
        pullback=None,          # answers a different question, not q itself
        slot=slot.text,
        source="programmatic",
    )


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DEPENDENT_START = re.compile(
    r"^\s*(he|she|it|they|them|his|her|their|then|next|after|afterwards|finally|"
    r"so|therefore|thus|but|and|also|however|this|that|these|those|each|both|"
    r"the rest|the remainder)\b", re.IGNORECASE)


def _apply_permute_premises(question: str, candidate: str = "",
                            index: int = 0) -> Optional[Variant]:
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


REL_MASK = Relation(
    name="mask_quantity",
    transformation="mask one stated quantity and re-derive it from the candidate answer",
    direction="increasing",
    relation_text="the masked quantity must be recovered exactly",
    apply=_apply_mask_quantity,
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
REL_SCALE_X2 = Relation(
    name="scale_quantities_x2",
    transformation="multiply every quantity in the problem by 2",
    direction="symmetric",
    relation_text="the answer must be exactly 2 times the original answer",
    apply=lambda q, a="", i=0: _scale_quantities(q, 2.0),
)
REL_SCALE_X3 = Relation(
    name="scale_quantities_x3",
    transformation="multiply every quantity in the problem by 3",
    direction="symmetric",
    relation_text="the answer must be exactly 3 times the original answer",
    apply=lambda q, a="", i=0: _scale_quantities(q, 3.0),
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


def _apply_rename_identifiers(question: str, candidate: str = "",
                              index: int = 0) -> Optional[Variant]:
    src = _rewrite_code(question, lambda t: _LocalRenamer().visit(t))
    if src is None or src == question:
        return None
    return equality_variant(REL_RENAME, src, source="programmatic")


def _apply_insert_dead_code(question: str, candidate: str = "",
                            index: int = 0) -> Optional[Variant]:
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


# ── The catalogue: task -> C ───────────────────────────────────────────────────
#
# Table 3 of the paper. Sample(C, n) draws the first n *applicable* entries in
# this order, so the order is part of the design: the relations whose image is
# answered at least as reliably as its source come first, and the covariant ones
# (scaling, whose validity is conditional on homogeneity) come last.

_CODE_RELATIONS = [REL_RENAME, REL_DEAD_CODE]
_OPTION_RELATIONS = [REL_OPTIONS_SHIFT1, REL_OPTIONS_REVERSE, REL_OPTIONS_SHIFT2]

CATALOGUES: Dict[str, List[Relation]] = {
    "mgsm": [REL_MASK, REL_PERMUTE_PREMISES, REL_TRANSLATE_EN,
             REL_SCALE_X2, REL_SCALE_X3],
    # Canonicalisation first (the only strictly reliability-increasing entry),
    # then the exact isometries, then option relabelling — which is all that
    # remains applicable to the arc-bearing sector/ellipse queries.
    "bigbenchhard:geometric_shapes": [REL_SVG_CANONICALISE, REL_SVG_REVERSE,
                                      REL_SVG_TRANSLATE, REL_SVG_REFLECT,
                                      REL_SVG_ROTATE90, REL_OPTIONS_SHIFT1,
                                      REL_OPTIONS_REVERSE],
    # Reformatting the options first (unambiguous dates are the reliability-
    # increasing direction), then the calendar relabellings, then plain option
    # permutation as the fallback that always applies.
    # One of each kind of attack first — reformat, relabel the calendar, relabel
    # the letters — so the three drawn probes are as uncorrelated as this task
    # allows; the second reformat and the forward shift come after.
    "bigbenchhard:date_understanding": [REL_DATES_SPELL_OUT, REL_DATES_SHIFT_BACK,
                                        REL_OPTIONS_SHIFT1, REL_DATES_ISO,
                                        REL_OPTIONS_REVERSE, REL_DATES_SHIFT_FORWARD,
                                        REL_OPTIONS_SHIFT2],
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
    """Build the catalogue lookup key, e.g. 'bigbenchhard:geometric_shapes'."""
    base = (task or "").lower()
    return f"{base}:{subtask.lower()}" if subtask else base


def get_catalogue(task: Optional[str], subtask: Optional[str] = None,
                  names: Optional[List[str]] = None) -> List[Relation]:
    """Return C for a task, optionally restricted to the named relations.

    Lookup is most-specific-first (``task:subtask`` then ``task``), mirroring
    ``HasChecker``: the catalogue is fixed per benchmark at construction time,
    never chosen at run time by the model. The returned order is the order in
    which ``Sample(C, n)`` will draw from it.
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
