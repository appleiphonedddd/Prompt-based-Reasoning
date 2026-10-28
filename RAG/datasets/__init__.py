"""
Knowledge-intensive QA datasets for the RAG experiments (Table 4.6).

``DATASET_REGISTRY`` maps the ``--dataset`` key to its loader class. Modules are
resolved with importlib because ``2WikiMQA`` starts with a digit and so cannot be
written in an ``import`` statement.
"""

import importlib
from typing import Dict, Type

from RAG.base.base_data_loader import BaseDataLoader

_LOADERS = {
    "hotpotqa":   ("RAG.datasets.HotpotQA.hotpotqa",     "HotpotQA"),
    "musique":    ("RAG.datasets.MuSiQue.musique",       "MuSiQue"),
    "2wikimqa":   ("RAG.datasets.2WikiMQA.twowikimqa",   "TwoWikiMQA"),
    "iirc":       ("RAG.datasets.IIRC.iirc",             "IIRC"),
    "strategyqa": ("RAG.datasets.StrategyQA.strategyqa", "StrategyQA"),
}


def get_loader(name: str) -> Type[BaseDataLoader]:
    module, cls = _LOADERS[name.lower()]
    return getattr(importlib.import_module(module), cls)


DATASET_REGISTRY: Dict[str, str] = {k: v[1] for k, v in _LOADERS.items()}
