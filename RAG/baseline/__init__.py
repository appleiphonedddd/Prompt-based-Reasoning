"""
Retrieval strategies for Table 4.6.

``RETRIEVAL_REGISTRY`` maps ``--retrieval`` to (class, kwargs-extractor), the same
pattern as ``BASELINE_REGISTRY`` in ``main.py``.
"""

from RAG.baseline.DRAGIN.dragin import DRAGIN
from RAG.baseline.IRCoT.ircot import IRCoT
from RAG.baseline.NoRetrieval.no_retrieval import NoRetrieval
from RAG.baseline.SingleRound.single_round import SingleRound

RETRIEVAL_REGISTRY = {
    "none":   (NoRetrieval, lambda _: {}),
    "single": (SingleRound, lambda _: {}),
    "ircot":  (IRCoT, lambda a: dict(
        step_topk=a.ircot_topk,
        max_steps=a.ircot_max_steps,
        max_paragraphs=a.ircot_max_paragraphs,
        reader=a.ircot_reader,
    )),
    "dragin": (DRAGIN, lambda a: dict(
        threshold=a.dragin_threshold,
        top_n=a.dragin_top_n,
        max_rounds=a.dragin_max_rounds,
    )),
}

__all__ = ["RETRIEVAL_REGISTRY", "NoRetrieval", "SingleRound", "IRCoT", "DRAGIN"]
