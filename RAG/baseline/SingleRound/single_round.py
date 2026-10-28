"""
Single-round retrieval (DRAGIN's SR-RAG, IRCoT's OneR; Table 4.6 ref. [55]).

Retrieve once, before reasoning, with the original question as the query; the
passages go into the prompt and the model writes its whole chain of thought on
top of them. What to retrieve is fixed by the question alone, so evidence for a
later hop — whose entity only surfaces mid-reasoning — is out of reach.
"""

from __future__ import annotations

from RAG.base.base_baseline import BaseRAG, SolveResult


class SingleRound(BaseRAG):
    name = "single"

    def solve(self, question: str) -> SolveResult:
        passages = self.retrieve(question)
        text = self.complete(self.build_prompt(question, passages))
        text = self.finish(question, passages, text)
        return self.result(text, passages,
                           [f"[Retrieve] query=question → {len(passages)} passages"])
