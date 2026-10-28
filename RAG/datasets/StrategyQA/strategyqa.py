"""
StrategyQA (Geva et al., 2021) — yes/no questions with implicit reasoning steps.

Source: https://huggingface.co/datasets/voidful/StrategyQA (raw JSON files). The
``test`` file carries no labels (490 rows of ``qid`` + ``question`` only), so the
default is ``train`` (2290 labelled questions) — the file DRAGIN evaluates on:
the test question of its App. F Prompt 5 ("Are more people today related to
Genghis Khan than Julius Caesar?") is that file's first row. ``train_filtered``
is accepted too.

Metric: accuracy of the yes/no verdict.

Corpus for ``--corpus dataset``: ``strategyqa_train_paragraphs.json``, the
Wikipedia paragraphs annotated as evidence.

Prompt: DRAGIN App. F, Prompt 5 (Wei et al., 2022).
"""

from __future__ import annotations

import json
from typing import Iterator, List

from huggingface_hub import hf_hub_download

from RAG.base.base_data_loader import BaseDataLoader, Passage, QAExample, split_words

_REPO = "voidful/StrategyQA"
_FILES = {
    "train": "strategyqa_train.json",
    "train_filtered": "strategyqa_train_filtered.json",
    "test": "strategyqa_test.json",
}


def _load_json(filename: str):
    with open(hf_hub_download(_REPO, filename, repo_type="dataset"), encoding="utf-8") as f:
        return json.load(f)


class StrategyQA(BaseDataLoader):
    name = "strategyqa"
    metric = "accuracy"
    default_split = "train"
    generate_max_length = 100
    dragin_threshold = 1.0
    dragin_top_n = 25
    instruction = "Following the examples above, answer the question by reasoning step-by-step."
    exemplars = [
        ("Do hamsters provide food for any animals?",
         "Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some "
         "animals. So the answer is yes."),
        ("Could Brooke Shields succeed at University of Pennsylvania?",
         "Brooke Shields went to Princeton University. Princeton University is about as academically "
         "rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the "
         "University of Pennsylvania. So the answer is yes."),
        ("Hydrogen's atomic number squared exceeds number of Spice Girls?",
         "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's "
         "atomic number squared is less than 5. So the answer is no."),
        ("Is it common to see frost during some college commencements?",
         "College commencement ceremonies can happen in December, May, and June. December is in the "
         "winter, so there can be frost. Thus, there could be frost at some commencements. So the "
         "answer is yes."),
        ("Could a llama birth twice during War in Vietnam (1945-46)?",
         "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more "
         "than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the "
         "answer is no."),
        ("Would a pear sink in water?",
         "The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than "
         "water float. Thus, a pear would float. So the answer is no."),
    ]

    def load_examples(self) -> List[QAExample]:
        if self.split not in _FILES:
            raise ValueError(f"StrategyQA split must be one of {list(_FILES)}, got '{self.split}'")
        rows = _load_json(_FILES[self.split])
        examples = [QAExample(qid=r["qid"], question=r["question"],
                              answers=["yes" if r["answer"] else "no"],
                              metadata={"term": r.get("term")})
                    for r in rows if "answer" in r]
        if not examples:
            raise ValueError(f"StrategyQA split '{self.split}' has no labels; use --split train")
        return examples

    def corpus_passages(self) -> Iterator[Passage]:
        for paragraph in _load_json("strategyqa_train_paragraphs.json").values():
            for chunk in split_words(paragraph["content"]):
                yield Passage(title=paragraph["title"], text=chunk)
