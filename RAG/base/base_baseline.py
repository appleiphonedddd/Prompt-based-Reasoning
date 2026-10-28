"""
Base definitions for the retrieval strategies of Table 4.6.

This module defines:
- Usage: thread-safe per-question counters (LLM calls, tokens, retrievals)
- SolveResult: the outcome of one strategy run on one question
- BaseRAG: the interface every strategy under ``RAG/baseline/`` implements

A strategy decides *when* to retrieve and *with what query*; everything else is
shared and lives here, so that rows of the table differ in the retrieval strategy
alone (the property the thesis states for Table 4.6):

- one prompt template (DRAGIN App. F): the dataset's exemplars, then — when there
  are passages — ``Context: [1] ... [k]`` and "Answer in the same format as
  before.", then the dataset's instruction line and ``Question: ... Answer:``;
- one BM25 retriever and one corpus;
- one decoding policy (greedy, the dataset's generation length);
- one answer protocol: a CoT ends in "So the answer is X."; if it never gets
  there, "So the answer is" is appended and the model completes it (DRAGIN
  App. B).

A strategy instance is built per question (it is cheap), so its counters belong
to that question even when several questions run concurrently.
"""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from RAG.base.base_data_loader import BaseDataLoader, Passage
from RAG.base.generator import Generator, Prompt
from RAG.base.retriever import BM25Retriever
from RAG.metrics import extract_answer, has_answer

# Stop before the model starts writing the next few-shot example.
STOP = ("\nQuestion:", "\nContext:", "\n\n\n")


@dataclass
class Usage:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retrievals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_call(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.llm_calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def add_retrieval(self) -> None:
        with self._lock:
            self.retrievals += 1

    def as_dict(self) -> Dict[str, int]:
        return {"llm_calls": self.llm_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "retrievals": self.retrievals}


@dataclass
class SolveResult:
    """One strategy run on one question.

    Attributes:
        text: the full chain of thought, ending in "So the answer is X."
        answer: the short answer extracted from ``text``.
        passages: the passages in the prompt that produced ``text`` (the
            evidence FoT's repair is shown).
        trace: human-readable log of the strategy's decisions.
    """

    text: str
    answer: str
    passages: List[Passage] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)


# ── text utilities ─────────────────────────────────────────────────────────────

_ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "no", "vs", "etc", "inc", "ltd", "co",
           "mt", "gen", "lt", "col", "capt", "prof", "rev", "ft", "fr", "sgt", "u.s", "e.g", "i.e"}


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Character spans of the sentences of ``text``.

    A sentence ends at ``.``, ``!`` or ``?`` followed by whitespace, unless the
    period closes an initial ("F.W. Murnau", "Edward L. Cahn"), an abbreviation
    ("Dr. Hibbert", "No. 1") or a decimal number.
    """
    spans, start = [], 0
    for m in re.finditer(r"[.!?](?=\s+|$)", text):
        end = m.end()
        if m.group() == ".":
            word = re.search(r"(?:^|[\s(])([\w.]+)\.$", text[start:end])
            token = word.group(1).lower() if word else ""
            if token in _ABBREV or (len(token.split(".")[-1]) == 1 and token[-1:].isalpha()):
                continue
        if re.fullmatch(r"\W*(\d+|[A-Za-z])[.)]", text[start:end].strip()):
            continue  # a list marker ("1.", "b)") opens a sentence, it does not end one
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    # Shrink each span to its stripped content.
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        out.append((s, e))
    return out


def first_sentence(text: str) -> str:
    spans = sentence_spans(text)
    return text[spans[0][0]:spans[0][1]].strip() if spans else ""


def join_text(*parts: str) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def answer_end(text: str) -> Optional[int]:
    """End offset of the first "answer is ..." line, or None if there is none."""
    m = re.search(r"answer\s+is\s*:?[^\n]*", text, re.IGNORECASE)
    return m.end() if m else None


def trim_after_answer(text: str) -> str:
    """Drop anything the model wrote after its "So the answer is ..." line."""
    end = answer_end(text)
    return (text[:end] if end is not None else text).strip()


def dedupe(passages: Sequence[Passage], limit: Optional[int] = None) -> List[Passage]:
    seen, out = set(), []
    for p in passages:
        key = (p.title, p.text)
        if key not in seen:
            seen.add(key)
            out.append(p)
        if limit is not None and len(out) >= limit:
            break
    return out


class BaseRAG(ABC):
    """Abstract base class for a retrieval strategy.

    Attributes:
        generator: the frozen LLM.
        retriever: the shared BM25 retriever (None for No Retrieval).
        loader: the dataset, which supplies exemplars and generation defaults.
        topk: passages per retrieval call (3 in DRAGIN, Table 8).
        max_new_tokens: tokens per generation (DRAGIN Table 9 "generate length").
        usage: this question's counters.
    """

    name: str = ""

    def __init__(self, generator: Generator, retriever: Optional[BM25Retriever],
                 loader: BaseDataLoader, topk: int = 3,
                 max_new_tokens: Optional[int] = None, usage: Optional[Usage] = None) -> None:
        self.generator = generator
        self.retriever = retriever
        self.loader = loader
        self.topk = topk
        self.max_new_tokens = max_new_tokens or loader.generate_max_length
        self.usage = usage or Usage()

    # ── shared operations ──────────────────────────────────────────────────────
    def complete(self, prompt: Prompt, max_new_tokens: Optional[int] = None,
                 stop: Sequence[str] = STOP) -> str:
        out = self.generator.complete(prompt, max_new_tokens or self.max_new_tokens, stop)
        self.usage.add_call(out.input_tokens, out.output_tokens)
        return out.text

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Passage]:
        if self.retriever is None:
            return []
        self.usage.add_retrieval()
        return self.retriever.retrieve(query, k or self.topk)

    def build_prompt(self, question: str, passages: Sequence[Passage] = (),
                     partial: str = "") -> Prompt:
        """The shared few-shot template (DRAGIN App. F, Prompts 1-8)."""
        query = ""
        if passages:
            query += "Context:\n" + "".join(f"[{i}] {p.render()}\n"
                                            for i, p in enumerate(passages, start=1))
            query += "Answer in the same format as before.\n"
        if self.loader.instruction:
            query += self.loader.instruction + "\n"
        query += f"Question: {question}"
        return Prompt(demos=list(self.loader.exemplars), query=query, partial=partial)

    def finish(self, question: str, passages: Sequence[Passage], text: str) -> str:
        """Make sure the CoT ends in an answer (DRAGIN App. B)."""
        text = trim_after_answer(text)
        if has_answer(text):
            return text
        # A CoT cut off right at "So the answer is" is completed, not re-prefixed.
        text = re.sub(r"\s*(so\s+)?the\s+answer\s+is\s*:?\s*$", "", text, flags=re.IGNORECASE)
        prefix = join_text(text, "So the answer is")
        tail = self.complete(self.build_prompt(question, passages, prefix),
                             max_new_tokens=32, stop=("\n",))
        return join_text(prefix, first_sentence(tail))

    def result(self, text: str, passages: Sequence[Passage], trace: List[str]) -> SolveResult:
        return SolveResult(text=text, answer=extract_answer(text),
                           passages=list(passages), trace=trace)

    @abstractmethod
    def solve(self, question: str) -> SolveResult:
        """Answer ``question`` with this strategy's retrieval schedule."""
