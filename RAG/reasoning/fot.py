"""
FoT on top of a retrieval strategy: the "FoT" rows of Table 4.6.

FoT is the repository's driver (:class:`baseline.FoT.FoT`, Algorithm 4) with the
retrieval strategy as its base reasoner:

  * Solve  — the strategy itself (No Retrieval / Single-round / IRCoT / DRAGIN),
             so the initial candidate is exactly the "CoT" row's answer.
  * Falsify — metamorphic regime with the QA catalogue (:mod:`RAG.reasoning.
             relations`). Every variant is solved by the *same strategy*, from
             scratch: it retrieves for its own wording, and the candidate never
             appears in its prompt (Remark 1).
  * Repair — pi_rep, shown the orbit of disagreeing answers *and* the passages
             the orbit retrieved (the candidate's first, then each variant's),
             so a repair can settle the disagreement on evidence. Under No
             Retrieval there are none and repair stays parametric.

Nothing else in the driver changes: tau, the orbit-majority rule, the archive
and its min_≺ selection all carry over. Two hooks are bound to QA: answers are
read off "So the answer is X." / ``ANSWER:`` by :func:`RAG.metrics.extract_answer`,
and compared with :func:`RAG.metrics.answer_key`.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from baseline.basebaseline import BaselineResponse
from baseline.FoT import FoT
from baseline.FoT.relations import Relation, Variant
from RAG.base.base_baseline import BaseRAG, dedupe
from RAG.base.base_data_loader import Passage
from RAG.metrics import answer_key, extract_answer
from RAG.reasoning.cot import Prediction
from RAG.reasoning.relations import get_qa_catalogue, preserves_surface, with_qa_rho


class RAGFoT(FoT):
    """Falsification-of-Thought whose Solve is a retrieval strategy."""

    def __init__(self, llm, method: BaseRAG, relations: Optional[List[str]] = None,
                 repair_passages: int = 10, **fot_kwargs: Any) -> None:
        super().__init__(llm, task=f"rag:{method.loader.name}", execute_code=False,
                         generate_relations=False, **fot_kwargs)
        self.method = method
        self.repair_passages = repair_passages
        self._catalogue = get_qa_catalogue(relations)
        self._evidence: Dict[str, List[Passage]] = {}
        self._evidence_lock = threading.Lock()

    # ── QA hooks ───────────────────────────────────────────────────────────────
    def _answer(self, text: str) -> str:
        return extract_answer(text)

    def _key(self, answer: str) -> str:
        return answer_key(answer)

    # ── Solve: the retrieval strategy ──────────────────────────────────────────
    def solve(self, question: str) -> str:
        result = self.method.solve(question)
        with self._evidence_lock:
            self._evidence[question] = result.passages
        return result.text

    def build_variant(self, relation: Relation, question: str) -> Optional[Variant]:
        variant = super().build_variant(relation, question)
        if variant is None or not preserves_surface(question, variant.question):
            return None
        return with_qa_rho(variant)

    # ── Repair: pi_rep with the orbit's evidence ───────────────────────────────
    def _evidence_block(self, question: str) -> str:
        with self._evidence_lock:
            pools = [self._evidence.get(question, [])] + [
                p for q, p in self._evidence.items() if q != question]
        passages = dedupe([p for pool in pools for p in pool], self.repair_passages)
        if not passages:
            return ""
        return ("Retrieved passages (evidence gathered while answering the formulations "
                "below):\n" + "".join(f"[{i}] {p.render()}\n"
                                      for i, p in enumerate(passages, start=1)))

    def repair(self, question, candidate, witness, history):
        saved = self._solve_ctx
        self._solve_ctx = "\n\n".join(p for p in (saved, self._evidence_block(question)) if p)
        try:
            return super().repair(question, candidate, witness, history)
        finally:
            self._solve_ctx = saved

    def run(self, question: str, **kwargs: Any) -> BaselineResponse:
        self._evidence = {}
        return super().run(question, **kwargs)


class FoTReasoning:
    """FoT over a retrieval strategy, returning the same record as CoT."""

    name = "fot"

    def __init__(self, method: BaseRAG, **fot_kwargs: Any) -> None:
        self.method = method
        self.fot = RAGFoT(method.generator, method, **fot_kwargs)

    def run(self, question: str) -> Prediction:
        response = self.fot.run(question, instruction=self.method.loader.answer_instruction)
        usage = self.method.usage.as_dict()
        # FoT's own calls (pi_mr, pi_rep) are counted by the driver, the strategy's
        # (every Solve) by the strategy.
        usage["llm_calls"] += response.num_llm_calls
        usage["input_tokens"] += response.total_input_tokens
        usage["output_tokens"] += response.total_output_tokens
        with self.fot._evidence_lock:
            passages = self.fot._evidence.get(question, [])
        meta = {k: response.metadata[k] for k in (
            "initial_answer", "answer_changed", "accepted_fixpoint", "returned",
            "repairs_used", "witnesses", "catalogue")}
        return Prediction(answer=response.final_answer, text=response.reasoning_trace,
                          trace=response.intermediate_steps,
                          passage_titles=[p.title for p in passages], usage=usage,
                          metadata=meta)
