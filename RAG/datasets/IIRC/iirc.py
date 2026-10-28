"""
IIRC (Ferguson et al., 2020) — incomplete-information reading comprehension.

Source: https://huggingface.co/datasets/voidful/IIRC (``validation``). Each row is
one main passage with its questions nested under ``questions``; they are
flattened here. Questions whose answer type is ``none`` (unanswerable) are
dropped, as in DRAGIN and IRCoT. Gold aliases by answer type:

- ``span``   — the answer spans joined in order (most answers have one span);
- ``value``  — the value, and the value with its unit ("4" and "4 years");
- ``binary`` — "yes" / "no".

Like DRAGIN (App. F, Prompt 8) the question is asked open-domain: the main passage
is *not* shown, and everything must be retrieved.

Corpus for ``--corpus dataset``: the linked Wikipedia articles of the official
release (``context_articles.json``, downloaded on first use) plus the main
passages, stripped of HTML and cut into 100-word passages (IRCoT App. A).

Prompt: DRAGIN App. F, Prompt 7 (8 exemplars).
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator, List

from datasets import load_dataset as hf_load_dataset

from RAG.base.base_data_loader import (BaseDataLoader, Passage, QAExample,
                                       split_words, strip_html)

_REPO = "voidful/IIRC"
_ARTICLES_URL = "https://iirc-dataset.s3.us-west-2.amazonaws.com/context_articles.tar.gz"
_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus" / "iirc"


def gold_answers(answer: dict) -> List[str]:
    """Aliases for one IIRC answer record, or [] for an unanswerable question."""
    kind = answer["type"]
    if kind == "span":
        spans = [s["text"].strip() for s in answer["answer_spans"] or [] if s["text"].strip()]
        return [" ".join(spans)] if spans else []
    if kind == "value":
        value = (answer["answer_value"] or "").strip()
        unit = (answer["answer_unit"] or "").strip()
        return [value, f"{value} {unit}"] if value and unit else ([value] if value else [])
    if kind == "binary":
        value = (answer["answer_value"] or "").strip().lower()
        return [value] if value else []
    return []


def text_passages(title: str, text: str) -> Iterator[Passage]:
    for paragraph in strip_html(text).split("\n"):
        for chunk in split_words(paragraph):
            yield Passage(title=title, text=chunk)


class IIRC(BaseDataLoader):
    name = "iirc"
    metric = "em_f1"
    default_split = "validation"
    generate_max_length = 128
    dragin_threshold = 1.25
    dragin_top_n = 25
    instruction = ""
    exemplars = [
        ("What is the age difference between the kicker and the quarterback for the Chargers?",
         "The kicker for the Chargers is Nate Kaeding. The quarterback (QB) for the Chargers is Philip "
         "Rivers. Nate Kaeding was born in the year 1982. Philip Rivers was born in the year 1981. Thus, "
         "the age difference between them is of 1 year. So the answer is 1."),
        ("How many years was the ship that took the battalion from New South Wales to Ceylon in service?",
         "The ship that took the battalion from New South Wales to Ceylon is General Hewitt. General "
         "Hewitt was launched in Calcutta in 1811. General Hewitt was sold for a hulk or to be broken up "
         "in 1864. So she served for a total of 1864 - 1811 = 53 years. So the answer is 53."),
        ("What year was the theatre that held the 2016 NFL Draft built?",
         "The theatre that held the 2016 NFL Draft is Auditorium Theatre. The Auditorium Theatre was "
         "built in 1889. So the answer is 1889."),
        ("How long had Milan been established by the year that Nava returned there as a reserve in the "
         "first team's defense?",
         "Nava returned to Milan as a reserve in the first team's defense in the year 1990. Milan had "
         "been established in the year 1899. Thus, Milan had been established for 1990 - 1899 = 91 years "
         "when Milan returned to Milan as a reserve in the first team's defense. So the answer is 91."),
        ("When was the town Scott was born in founded?",
         "Scott was born in the town of Cooksville, Illinois. Cooksville was founded in the year 1882. "
         "So the answer is 1882."),
        ("In what country did Wright leave the French privateers?",
         "Wright left the French privateers in Bluefield's river. Bluefields is the capital of the South "
         "Caribbean Autonomous Region (RAAS) in the country of Nicaragua. So the answer is Nicaragua."),
        ("Who plays the A-Team character that Dr. Hibbert fashioned his hair after?",
         "Dr. Hibbert fashioned his hair after Mr. T from The A-Team. Mr T.'s birthname is Lawrence "
         "Tureaud. So the answer is Lawrence Tureaud."),
        ("How many people attended the conference held near Berlin in January 1942?",
         "The conference held near Berlin in January 1942 is Wannsee Conference. Wannsee Conference was "
         "attended by 15 people. So the answer is 15."),
    ]

    def load_examples(self) -> List[QAExample]:
        examples: List[QAExample] = []
        for row in hf_load_dataset(_REPO, split=self.split):
            for q in row["questions"]:
                answers = gold_answers(q["answer"])
                if answers:
                    examples.append(QAExample(
                        qid=q["qid"], question=q["question"], answers=answers,
                        metadata={"answer_type": q["answer"]["type"],
                                  "main_passage_title": row["title"]}))
        return examples

    @staticmethod
    def _articles_path() -> Path:
        path = _CORPUS_DIR / "context_articles.json"
        if not path.exists():
            _CORPUS_DIR.mkdir(parents=True, exist_ok=True)
            archive = _CORPUS_DIR / "context_articles.tar.gz"
            print(f"Downloading IIRC context articles → {archive}")
            urllib.request.urlretrieve(_ARTICLES_URL, archive)
            with tarfile.open(archive) as tar:
                tar.extract("context_articles.json", _CORPUS_DIR, filter="data")
            archive.unlink()
        return path

    def corpus_passages(self) -> Iterator[Passage]:
        with open(self._articles_path(), encoding="utf-8") as f:
            articles = json.load(f)
        for title, text in articles.items():
            yield from text_passages(title, text)
        for split in ("train", "validation"):
            for row in hf_load_dataset(_REPO, split=split):
                yield from text_passages(row["title"], row["text"])
