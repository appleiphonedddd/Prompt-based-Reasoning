"""
Answer extraction and scoring for knowledge-intensive QA (Table 4.6).

Two metric families, fixed per dataset:

- ``em_f1``    — HotpotQA, MuSiQue, 2WikiMultihopQA, IIRC. Exact match and token
                 F1 after the official HotpotQA / SQuAD normalisation (lowercase,
                 strip punctuation and articles, collapse whitespace), maximised
                 over the gold aliases.
- ``accuracy`` — StrategyQA. The yes/no verdict is read off the prediction and
                 compared with the boolean label.

Both papers extract the answer from the tail of a CoT that ends in "So the answer
is X." (DRAGIN App. B, IRCoT §3.2); :func:`extract_answer` does that, and also
accepts FoT's ``ANSWER:`` tag so that a repaired candidate is scored the same way
as a solved one.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# ── Answer extraction ──────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
_ANSWER_IS_RE = re.compile(r"answer\s+is\s*:?\s*(.+)", re.IGNORECASE)


def has_answer(text: str) -> bool:
    """True once a CoT has reached its "So the answer is ..." sentence."""
    return bool(_ANSWER_IS_RE.search(text) or _TAG_RE.search(text))


def extract_answer(text: str) -> str:
    """Pull the short final answer out of a CoT or an ``ANSWER:``-tagged response.

    Precedence: the last ``ANSWER:`` tag, then the last "answer is" phrase, then
    the last non-empty line. The match runs to the end of its line and loses a
    trailing period, so "So the answer is 19 June 2013." gives "19 June 2013".
    """
    text = (text or "").strip()
    for pattern in (_TAG_RE, _ANSWER_IS_RE):
        matches = list(pattern.finditer(text))
        if matches:
            answer = matches[-1].group(1).strip()
            if answer:
                return _clean(answer)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return _clean(lines[-1]) if lines else ""


def _clean(answer: str) -> str:
    answer = answer.splitlines()[0].strip()
    answer = answer.strip("*`\"' ")
    return re.sub(r"[.\s]+$", "", answer).strip()


# ── Normalisation (official HotpotQA / SQuAD evaluation script) ────────────────

def normalize_answer(s: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def yes_no(answer: str) -> Optional[str]:
    """Read a yes/no verdict off an answer, else None.

    The leading word decides ("Yes, they are" → "yes"); failing that, a single
    unambiguous verdict word anywhere ("likely yes" → "yes"). An answer that
    mentions both, or neither, has no verdict.
    """
    words = normalize_answer(answer).split()
    if not words:
        return None
    verdicts = {"yes": "yes", "true": "yes", "no": "no", "false": "no"}
    if words[0] in verdicts:
        return verdicts[words[0]]
    found = {verdicts[w] for w in words if w in verdicts}
    return found.pop() if len(found) == 1 else None


def answer_key(answer: str) -> str:
    """Canonical key used to compare two model answers (FoT's rho for QA).

    Yes/no verdicts collapse to "yes"/"no" (so "True" and "Yes, they are" agree);
    everything else is compared after the official normalisation — never by a
    numeric reading, which would make "July 21, 1926" equal "July 21, 1930".
    """
    verdict = yes_no(answer)
    return verdict if verdict is not None else normalize_answer(answer)


# ── Scoring ────────────────────────────────────────────────────────────────────

def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1, with the official special case for yes/no/noanswer."""
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    special = ("yes", "no", "noanswer")
    if (pred in special or ref in special) and pred != ref:
        return 0.0
    pred_toks, ref_toks = pred.split(), ref.split()
    common = Counter(pred_toks) & Counter(ref_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_toks)
    recall = same / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def score(prediction: str, golds: Sequence[str], metric: str) -> Dict[str, float]:
    """Score one prediction against its gold aliases.

    Args:
        prediction: the extracted short answer.
        golds: gold aliases; for ``accuracy`` the first entry is "yes" or "no".
        metric: "em_f1" or "accuracy".
    """
    if metric == "accuracy":
        return {"acc": float(yes_no(prediction) == golds[0])}
    return {
        "em": max(exact_match(prediction, g) for g in golds),
        "f1": max(f1_score(prediction, g) for g in golds),
    }


@dataclass
class ScoreBoard:
    """Running means of the per-question scores (EM/F1 or Acc)."""

    metric: str
    totals: Dict[str, float] = field(default_factory=dict)
    count: int = 0

    def record(self, scores: Dict[str, float]) -> None:
        for name, value in scores.items():
            self.totals[name] = self.totals.get(name, 0.0) + value
        self.count += 1

    def means(self) -> Dict[str, float]:
        names = ["acc"] if self.metric == "accuracy" else ["em", "f1"]
        return {n: (self.totals.get(n, 0.0) / self.count if self.count else 0.0)
                for n in names}

    def summary(self) -> str:
        return "  ".join(f"{n.upper()}={v:.4f}" for n, v in self.means().items())


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
