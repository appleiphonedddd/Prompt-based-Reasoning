from __future__ import annotations

from typing import Iterator, List

from datasets import load_dataset as hf_load_dataset

from RAG.base.base_data_loader import BaseDataLoader, Passage, QAExample
from RAG.datasets.HotpotQA.hotpotqa import context_passages

_REPO = "framolfese/2WikiMultihopQA"


class TwoWikiMQA(BaseDataLoader):
    name = "2wikimqa"
    metric = "em_f1"
    default_split = "validation"
    generate_max_length = 64
    dragin_threshold = 0.6
    dragin_top_n = 25
    instruction = ""
    exemplars = [
        ("When did the director of film Hypocrite (Film) die?",
         "The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the "
         "answer is 19 June 2013."),
        ("Are both Kurram Garhi and Trojkrsti located in the same country?",
         "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of "
         "Republic of Macedonia. Thus, they are not in the same country. So the answer is no."),
        ("Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have "
         "the same nationality?",
         "Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by "
         "Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, "
         "they do not have the same nationality. So the answer is no."),
        ("Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
         "Boraqchin is married to Ögedei Khan. Ögedei Khan's father is Genghis Khan. Thus, Boraqchin's "
         "father-in-law is Genghis Khan. So the answer is Genghis Khan."),
        ("Who was born first out of Martin Hodge and Ivania Martinich?",
         "Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, "
         "Martin Hodge was born first. So the answer is Martin Hodge."),
        ("When did the director of film Laughter In Hell die?",
         "The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, "
         "1963. So the answer is August 25, 1963."),
    ]

    def load_examples(self) -> List[QAExample]:
        rows = hf_load_dataset(_REPO, split=self.split)
        examples = [QAExample(qid=r["id"], question=r["question"], answers=[r["answer"]],
                              metadata={"type": r["type"]})
                    for r in rows if r["answer"]]
        if not examples:
            raise ValueError(f"2WikiMultihopQA split '{self.split}' has no gold answers "
                             "(the test answers are withheld); use --split validation")
        return examples

    def corpus_passages(self) -> Iterator[Passage]:
        for split in ("train", "validation", "test"):
            for row in hf_load_dataset(_REPO, split=split):
                yield from context_passages(row["context"])
