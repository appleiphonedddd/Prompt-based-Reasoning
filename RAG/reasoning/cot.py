"""
CoT reasoning: the "CoT" rows of Table 4.6.

The retrieval strategy's own chain of thought is the answer — what each paper
reports. ``Prediction`` is the common record both reasoning modes return to
``eval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from RAG.base.base_baseline import BaseRAG


@dataclass
class Prediction:
    answer: str
    text: str
    trace: List[str]
    passage_titles: List[str]
    usage: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoTReasoning:
    name = "cot"

    def __init__(self, method: BaseRAG) -> None:
        self.method = method

    def run(self, question: str) -> Prediction:
        result = self.method.solve(question)
        return Prediction(answer=result.answer, text=result.text, trace=result.trace,
                          passage_titles=[p.title for p in result.passages],
                          usage=self.method.usage.as_dict())
