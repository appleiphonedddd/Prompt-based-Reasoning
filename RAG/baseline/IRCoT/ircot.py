"""
IRCoT: Interleaving Retrieval with Chain-of-Thought (Trivedi et al., ACL 2023).

Retrieval and reasoning guide each other one sentence at a time (§3.1, Fig. 2):

    P ← Retrieve(q, K)                         # base set, question as the query
    T ← ""
    repeat until "answer is" ∈ T or max_steps:
        t ← first sentence of Reason(q, P, T)  # reason step: extend the CoT
        T ← T + t
        P ← P ∪ Retrieve(t, K)                 # retrieve step: last CoT sentence
                                               #   is the query; |P| capped
    answer ← Reader(q, P)                      # QA reader (§3.2)

Paper settings: at most 8 reasoning steps and 15 collected paragraphs (footnotes
4-5); K is tuned per dataset over {2, 4, 6, 8} — here ``--ircot_topk``, defaulting
to the shared ``--retrieve_topk``. Only the *first* sentence of each reason step
is kept even if the model writes more.

The reader is a separate CoT prompt over all collected paragraphs, which the paper
found better than reading the answer off the retrieval CoT (App. F, Table 6);
``reader="none"`` gives that ablation. The reason step and the reader use the
template shared by every strategy, so this row differs from Single-round only in
its retrieval schedule.
"""

from __future__ import annotations

from typing import List, Optional

from RAG.base.base_baseline import BaseRAG, SolveResult, dedupe, first_sentence, join_text
from RAG.base.base_data_loader import Passage
from RAG.metrics import has_answer


class IRCoT(BaseRAG):
    name = "ircot"

    def __init__(self, *args, step_topk: Optional[int] = None, max_steps: int = 8,
                 max_paragraphs: int = 15, reader: str = "cot", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step_topk = step_topk or self.topk
        self.max_steps = max_steps
        self.max_paragraphs = max_paragraphs
        self.reader = reader

    def _collect(self, collected: List[Passage], query: str) -> List[Passage]:
        if len(collected) >= self.max_paragraphs:
            return collected
        return dedupe(collected + self.retrieve(query, self.step_topk), self.max_paragraphs)

    def solve(self, question: str) -> SolveResult:
        collected = self._collect([], question)
        trace = [f"[Retrieve] query=question → {len(collected)} paragraphs"]
        cot = ""
        for step in range(1, self.max_steps + 1):
            sentence = first_sentence(self.complete(self.build_prompt(question, collected, cot)))
            if not sentence:
                break
            cot = join_text(cot, sentence)
            trace.append(f"[Reason {step}] {sentence}")
            if has_answer(sentence):
                break
            collected = self._collect(collected, sentence)
            trace.append(f"[Retrieve {step}] query=last sentence → {len(collected)} paragraphs")

        if self.reader == "cot":
            text = self.complete(self.build_prompt(question, collected))
            trace.append("[Reader] CoT over the collected paragraphs")
        else:
            text = cot
        text = self.finish(question, collected, text)
        return self.result(text, collected, trace)
