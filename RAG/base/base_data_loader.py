"""
Base definitions for the knowledge-intensive QA datasets of Table 4.6.

This module defines:
- QAExample: one normalised question with its gold aliases
- Passage: one retrievable unit of a corpus
- BaseDataLoader: the interface every dataset under ``RAG/datasets/`` implements

A loader owns everything that is specific to its dataset and nothing that is
specific to a retrieval strategy: how to load and normalise the labelled split,
the few-shot CoT exemplars (DRAGIN App. F / IRCoT App. G), the per-dataset
generation defaults (DRAGIN Table 9), and the passages of the IRCoT-style
"dataset" corpus (IRCoT App. A). Every retrieval strategy builds its prompt from
the same exemplars and the same template, so a difference in the table reflects
the strategy and not the prompt.
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

# The answer-format directive handed to FoT's Solve/Repair, whose prompts are
# instruction-style rather than few-shot.
SHORT_ANSWER = ("Answer the question. The final answer must be as short as possible: "
                "an entity name, a date, a number, or yes/no -- never a sentence.")
YES_NO_ANSWER = "Answer the yes/no question. The final answer must be exactly 'yes' or 'no'."


@dataclass
class QAExample:
    """One question of an evaluation split.

    Attributes:
        qid: the dataset's own question id.
        question: the question text, exactly as the model sees it.
        answers: gold aliases. For yes/no datasets a single "yes" or "no".
        metadata: dataset-specific extras (question type, hop count, ...).
    """

    qid: str
    question: str
    answers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Passage:
    """One retrievable unit: a ~100-word Wikipedia chunk or a dataset paragraph."""

    title: str
    text: str

    def render(self) -> str:
        """The form a passage takes in the prompt (DRAGIN App. F: title, then text)."""
        return f"{self.title} {self.text}".strip() if self.title else self.text


def split_words(text: str, size: int = 100) -> List[str]:
    """Chunk a text into ``size``-word passages (DPR's 100-word segmentation)."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    return _TAG_RE.sub("", text)


class BaseDataLoader(ABC):
    """Abstract base class for a knowledge-intensive QA dataset.

    Class attributes a subclass sets:
        name: registry key, e.g. "hotpotqa".
        metric: "em_f1" or "accuracy".
        default_split: the labelled split evaluated by default.
        generate_max_length: DRAGIN Table 9 "generate length" (new tokens per
            generation, and the cap on a DRAGIN answer).
        dragin_threshold: DRAGIN Table 9 theta (Llama2-13b-chat column).
        dragin_top_n: DRAGIN Table 9 "top n tokens" kept by QFS.
        exemplars: (question, CoT ending in "So the answer is X.") pairs.
        instruction: the line placed right before the test question (App. F);
            empty where the paper's template has none.
    """

    name: str = ""
    metric: str = "em_f1"
    default_split: str = "validation"
    generate_max_length: int = 100
    dragin_threshold: float = 1.2
    dragin_top_n: int = 35
    exemplars: List[Tuple[str, str]] = []
    instruction: str = ""

    def __init__(self, split: Optional[str] = None) -> None:
        self.split = split or self.default_split

    # ── evaluation split ───────────────────────────────────────────────────────
    @abstractmethod
    def load_examples(self) -> List[QAExample]:
        """Load and normalise every labelled question of ``self.split``."""

    def load(self, num_samples: Optional[int] = None, shuffle: bool = False,
             seed: int = 0) -> List[QAExample]:
        """The evaluated subset: optionally shuffled, then the first ``num_samples``.

        Without ``shuffle`` this is the front of the split, as DRAGIN samples it.
        A question that is also a few-shot exemplar is dropped, since the prompt
        would hand the model its answer.
        """
        demo_questions = {q.strip().lower() for q, _ in self.exemplars}
        examples = [e for e in self.load_examples()
                    if e.question.strip().lower() not in demo_questions]
        if shuffle:
            random.Random(seed).shuffle(examples)
        if num_samples is not None and num_samples >= 0:
            examples = examples[:num_samples]
        return examples

    # ── retrieval corpus ───────────────────────────────────────────────────────
    @abstractmethod
    def corpus_passages(self) -> Iterator[Passage]:
        """Passages of the IRCoT-style corpus: every context paragraph shipped
        with the dataset, pooled over its splits (IRCoT App. A). Duplicates are
        removed by the index builder."""

    # ── prompting ──────────────────────────────────────────────────────────────
    @property
    def answer_instruction(self) -> str:
        """Answer-format directive for FoT's instruction-style prompts."""
        return YES_NO_ANSWER if self.metric == "accuracy" else SHORT_ANSWER
