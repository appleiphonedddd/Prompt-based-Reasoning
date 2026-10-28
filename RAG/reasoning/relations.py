"""
The relation catalogue C for FoT on knowledge-intensive QA (Table 4.6).

The benchmark catalogue (:mod:`baseline.FoT.relations`) has no entry for open-
domain QA, so FoT would fall back to option relabelling, apply to nothing, and
degrade to FoT ≡ Solve. This module is the QA counterpart, and like that one it
is **the artifact to audit**: every entry must be *valid* (rho holds between the
gold answers of q and T(q)) and *non-decreasing in reliability* (Remark 2).

All four relations are answer-preserving, so rho is equality and g = g⁻¹ = id.
Equality is :func:`RAG.metrics.answer_key` (official normalisation, yes/no
verdicts collapsed), never the benchmark catalogue's numeric reading, under which
"July 21, 1926" equals "July 21, 1930".

In RAG the orbit buys a second kind of independence: each variant goes through
the *whole* retrieval strategy again, so a reworded query retrieves its own
evidence. A disagreement can then expose a retrieval miss as well as a reasoning
slip.

Entries, in draw order:

  * ``qa_swap_operands`` (programmatic, symmetric) — list the two compared
    entities in the opposite order ("Who was born first, A or B?",
    "Were A and B of the same nationality?"). The answer names one of them, or
    is a verdict about a symmetric relation, so it is unchanged. Skipped (not
    violated) on every other question.
  * ``qa_decompose`` (pi_mr, *increasing*) — spell the question out as the chain
    of sub-questions it implies. Nothing is added or removed; the bridge entity
    the model has to find is only made explicit, which is the reliability-
    increasing direction (IRCoT's motivation in one rewrite).
  * ``qa_paraphrase`` (pi_mr, symmetric) — reword without changing what is asked.
  * ``qa_cloze`` (pi_mr, symmetric) — restate as a fill-in-the-blank statement
    (for a yes/no question, "True or false: ..."; true/false collapse to yes/no).

Excluded on Remark 2 grounds: negating a yes/no question (valid, with g = flip,
but negated questions are answered less reliably) and inserting distractor facts.

Model-instantiated variants are mechanically validated by
:func:`preserves_surface` before they are solved: every number and every
mid-sentence capitalised name of the question must survive, and the variant may
introduce no name of its own — so a rewrite that quietly answers a hop ("Who
directed X (Miguel Morayta)?") or drops a constraint is discarded, not solved.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import List, Optional

from baseline.FoT.relations import Relation, Variant
from RAG.metrics import answer_key

# ── rho ────────────────────────────────────────────────────────────────────────

def answers_agree(a: str, b: str) -> bool:
    return answer_key(a) == answer_key(b)


def qa_variant(relation: Relation, question: str, source: str) -> Variant:
    """An answer-preserving variant whose rho is QA equality."""
    return Variant(relation=relation.name, relation_text=relation.relation_text,
                   question=question, holds=answers_agree,
                   expected_for=lambda a: a, pullback=lambda a_prime: a_prime, source=source)


def with_qa_rho(variant: Variant) -> Variant:
    """Re-bind a variant built by FoT's pi_mr (rho = numeric equality) to QA equality."""
    return replace(variant, holds=answers_agree)


# ── mechanical validation of model-written variants ────────────────────────────

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_CAP_RE = re.compile(r"\b[A-ZÀ-ÖØ-Þ][\w'’\-]*")
_BOUNDARY = re.compile(r"(^|[.?!:;\n]|\d[.)]|#\d+|[-*•])\s*$")
_ALLOWED_NEW = {"true", "false", "first", "then", "finally", "next", "step", "question",
                "sub-question", "subquestion", "answer", "what", "who", "which", "when",
                "where", "how", "why", "is", "was", "are", "were", "did", "does", "do", "fill",
                "blank", "the", "in", "of", "i"}


_MARKER_RE = re.compile(r"#\d+|(?:^|(?<=\s))[1-9][.)](?=\s)")


def numbers(text: str) -> set:
    """Numeric literals, ignoring a decomposition's list markers ("1.") and back-
    references ("#1")."""
    return {n.replace(",", "") for n in _NUM_RE.findall(_MARKER_RE.sub(" ", text))}


def names(text: str) -> set:
    """Mid-sentence capitalised words — the entity names a rewrite must keep."""
    out = set()
    for m in _CAP_RE.finditer(text):
        if not _BOUNDARY.search(text[:m.start()]):
            out.add(m.group().lower())
    return out


def preserves_surface(original: str, variant: str) -> bool:
    if not variant.strip() or answer_key(variant) == answer_key(original):
        return False
    if re.search(r"answer\s+is", variant, re.IGNORECASE):
        return False
    if numbers(variant) != numbers(original):
        return False
    low = variant.lower()
    if any(name not in low for name in names(original)):
        return False
    original_words = set(re.findall(r"[\w'’\-]+", original.lower()))
    return all(n in original_words or n in _ALLOWED_NEW for n in names(variant))


# ── programmatic T: swap the compared entities ─────────────────────────────────

_OR_RE = re.compile(r"^(?P<head>.*?[,:?]\s+)(?P<a>[^,:?]+?)\s+or\s+(?P<b>[^,:?]+?)(?P<tail>\s*\?\s*)$")
_SAME_RE = re.compile(
    r"^(?P<aux>Are|Were|Is|Was|Do|Does|Did|Have|Had)\s+(?P<both>both\s+)?(?P<a>.+?)\s+and\s+"
    r"(?P<b>.+?)\s+(?P<rest>(?=.*\bsame\b)(?:have|has|had|share|shared|located|born|come|both|"
    r"(?:of|in|from)\s+the\s+same|the\s+same)\b.*\?)\s*$")


def _parallel_operands(a: str, b: str) -> bool:
    """Swap only two operands of the same kind: two names, or two phrases with the
    same head ("director of film X" / "director of film Y"). Anything else — a
    distributive "the directors of films X and Y", an appositive "mountains, X and
    Y," — would change the meaning when reordered, so it is left alone."""
    if any(s in a + b for s in (",", " or ", " and ")):
        return False
    if len(a.split()) > 12 or len(b.split()) > 12:
        return False
    return (a[:1].isupper() and b[:1].isupper()) or a.split()[0] == b.split()[0]


def _apply_swap(question: str, candidate: str = "", round_index: int = 0) -> Optional[Variant]:
    q = question.strip()
    m = _OR_RE.match(q)
    if m and _parallel_operands(m.group("a"), m.group("b")):
        swapped = f"{m.group('head')}{m.group('b')} or {m.group('a')}{m.group('tail')}"
        return qa_variant(REL_QA_SWAP, swapped, "programmatic")
    m = _SAME_RE.match(q)
    if m and _parallel_operands(m.group("a"), m.group("b")):
        swapped = (f"{m.group('aux')} {m.group('both') or ''}{m.group('b')} and "
                   f"{m.group('a')} {m.group('rest')}")
        return qa_variant(REL_QA_SWAP, " ".join(swapped.split()), "programmatic")
    return None


# ── catalogue ──────────────────────────────────────────────────────────────────

_KEEP = ("Keep every name, title, date and number exactly as written, introduce no new "
         "name, and do not answer anything.")

REL_QA_SWAP = Relation(
    name="qa_swap_operands",
    transformation="List the two entities being compared in the opposite order",
    direction="symmetric",
    relation_text="the compared entities were listed in the opposite order, which cannot "
                  "change the answer",
    apply=_apply_swap,
)
REL_QA_DECOMPOSE = Relation(
    name="qa_decompose",
    transformation=("Rewrite the question as the explicit chain of simpler sub-questions it "
                    "implies, numbered 1., 2., ..., where a later sub-question refers to an "
                    "earlier answer as #1, #2, and the last sub-question asks exactly what the "
                    "original question asks. " + _KEEP),
    direction="increasing",
    relation_text="the question was spelled out as its chain of sub-questions, so the final "
                  "answer must be identical",
    preserve_numbers=False,   # validated by preserves_surface instead
)
REL_QA_PARAPHRASE = Relation(
    name="qa_paraphrase",
    transformation="Paraphrase the question so that it asks exactly the same thing. " + _KEEP,
    direction="symmetric",
    relation_text="the question was reworded without changing what it asks, so the answer "
                  "must be identical",
    preserve_numbers=False,
)
REL_QA_CLOZE = Relation(
    name="qa_cloze",
    transformation=("Restate the question as a fill-in-the-blank statement in which ___ stands "
                    "for the answer; for a yes/no question, write a statement prefixed with "
                    "'True or false:' instead. " + _KEEP),
    direction="symmetric",
    relation_text="the question was restated as a statement to complete, so the answer must "
                  "be identical (true/false read as yes/no)",
    preserve_numbers=False,
)

QA_CATALOGUE: List[Relation] = [REL_QA_SWAP, REL_QA_DECOMPOSE, REL_QA_PARAPHRASE, REL_QA_CLOZE]


def get_qa_catalogue(names_: Optional[List[str]] = None) -> List[Relation]:
    """C for QA, optionally restricted to the named relations (``--fot_relations``)."""
    if not names_:
        return list(QA_CATALOGUE)
    wanted = {n.strip().lower() for n in names_}
    return [r for r in QA_CATALOGUE if r.name in wanted]
