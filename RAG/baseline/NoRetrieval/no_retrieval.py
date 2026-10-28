"""
No Retrieval — the reference row of Table 4.6 (DRAGIN's "wo-RAG", IRCoT's "NoR").

The model writes its chain of thought from parametric knowledge alone, with the
same exemplars and template every retrieval strategy uses.
"""

from __future__ import annotations

from RAG.base.base_baseline import BaseRAG, SolveResult


class NoRetrieval(BaseRAG):
    name = "none"

    def solve(self, question: str) -> SolveResult:
        text = self.complete(self.build_prompt(question))
        text = self.finish(question, [], text)
        return self.result(text, [], ["[Generate] no retrieval"])
