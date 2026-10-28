"""
MuSiQue (Trivedi et al., 2022) — 2-4 hop questions composed from single-hop ones.

Source: https://huggingface.co/datasets/dgslibisey/MuSiQue (``validation``, 2417
questions; the answerable subset, as IRCoT uses). Gold aliases are the answer
plus ``answer_aliases``.

Corpus for ``--corpus dataset``: the 20 paragraphs attached to every question,
pooled over train and validation (IRCoT App. A).

DRAGIN does not evaluate MuSiQue, so there is no App. F prompt and no Table 9
column. The exemplars are the first eight CoT annotations of IRCoT Listing 3,
rewritten into the shared "So the answer is X." form, and the generation
defaults follow HotpotQA's (the closest multi-hop setting in Table 9), with a
longer budget for the extra hops.
"""

from __future__ import annotations

from typing import Iterator, List

from datasets import load_dataset as hf_load_dataset

from RAG.base.base_data_loader import BaseDataLoader, Passage, QAExample

_REPO = "dgslibisey/MuSiQue"


class MuSiQue(BaseDataLoader):
    name = "musique"
    metric = "em_f1"
    default_split = "validation"
    generate_max_length = 128
    dragin_threshold = 1.2
    dragin_top_n = 35
    instruction = "Answer the following question by reasoning step-by-step, following the example above."
    exemplars = [
        ("When did the first large winter carnival take place in the city where CIMI-FM is licensed to "
         "broadcast?",
         "CIMI-FM is licensed to broadcast in Quebec City. The first large winter carnival in Quebec City "
         "took place in 1894. So the answer is 1894."),
        ("When was Neville A. Stanton's employer founded?",
         "The employer of Neville A. Stanton is University of Southampton. The University of Southampton "
         "was founded in 1862. So the answer is 1862."),
        ("What county is Hebron located in, in the same province the Heritage Places Protection Act "
         "applies to?",
         "Heritage Places Protection Act applies to the jurisdiction of Prince Edward Island. Hebron, "
         "Prince Edward Island is located in the Prince County. So the answer is Prince County."),
        ("What weekly publication in the Connecticut city with the most Zagat rated restaurants is issued "
         "by university of America-Lite: How Imperial Academia Dismantled Our Culture's author?",
         "The author of America-Lite: How Imperial Academia Dismantled Our Culture is David Gelernter. "
         "David Gelernter was educated at the Yale University. The city in Connecticut that has the "
         "highest number of Zagat-rated restaurants is New Haven. The weekly publication in New Haven "
         "that is issued by Yale University is Yale Herald. So the answer is Yale Herald."),
        ("What is the headquarters for the organization who sets the standards for ISO 21500?",
         "The standards for ISO 21500 were set by International Organization for Standardization. The "
         "International Organization for Standardization has headquarters in Geneva. So the answer is "
         "Geneva."),
        ("What did the publisher of Banjo-Tooie rely primarily on for its support?",
         "The publisher of Banjo-Tooie is Nintendo. Nintendo relied primarily for its support on "
         "first-party games. So the answer is first-party games."),
        ("In which county was the birthplace of the Smoke in tha City performer?",
         "The performer of Smoke in tha City is MC Eiht. MC Eiht's birthplace is Compton. Compton is "
         "located in the county of Los Angeles County. So the answer is Los Angeles County."),
        ("What region of the state where Guy Shepherdson was born, contains SMA Negeri 68?",
         "Guy Shepherdson was born in Jakarta. SMA Negeri 68 Jakarta is located in Central Jakarta. So "
         "the answer is Central Jakarta."),
    ]

    def load_examples(self) -> List[QAExample]:
        rows = hf_load_dataset(_REPO, split=self.split)
        return [QAExample(qid=r["id"], question=r["question"],
                          answers=[r["answer"]] + [a for a in r["answer_aliases"] if a],
                          metadata={"hops": int(r["id"].split("hop")[0]) if "hop" in r["id"] else None})
                for r in rows if r["answerable"] and r["answer"]]

    def corpus_passages(self) -> Iterator[Passage]:
        for split in ("train", "validation"):
            for row in hf_load_dataset(_REPO, split=split):
                for p in row["paragraphs"]:
                    yield Passage(title=p["title"], text=" ".join(p["paragraph_text"].split()))
