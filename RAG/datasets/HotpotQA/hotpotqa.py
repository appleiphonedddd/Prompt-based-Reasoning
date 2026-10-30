from __future__ import annotations

from typing import Iterator, List

from datasets import load_dataset as hf_load_dataset

from RAG.base.base_data_loader import BaseDataLoader, Passage, QAExample

_REPO = "hotpotqa/hotpot_qa"


def context_passages(context) -> Iterator[Passage]:
    """{'title': [...], 'sentences': [[...], ...]} → one passage per paragraph."""
    for title, sentences in zip(context["title"], context["sentences"]):
        text = " ".join(" ".join(s.strip() for s in sentences).split())
        if text:
            yield Passage(title=title, text=text)


class HotpotQA(BaseDataLoader):
    name = "hotpotqa"
    metric = "em_f1"
    default_split = "validation"
    generate_max_length = 100
    dragin_threshold = 1.2
    dragin_top_n = 35
    instruction = "Answer the following question by reasoning step-by-step, following the example above."
    exemplars = [
        ("Jeremy Theobald and Christopher Nolan share what profession?",
         "Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and "
         "screenwriter. Therefore, they both share the profession of being a producer. So the answer is "
         "producer."),
        ("What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?",
         "Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the "
         "films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by "
         "F.W. Murnau. So the answer is The Phantom Hour."),
        ("How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?",
         "The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988. The "
         "number of episodes Reply 1988 has is 20. So the answer is 20."),
        ("Were Lonny and Allure both founded in the 1990s?",
         "Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, "
         "only Allure was founded in 1990s. So the answer is no."),
        ("Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?",
         "The actor who played astronaut Alan Shepard in \"The Right Stuff\" is Scott Glenn. The movie "
         "Vertical Limit also starred Scott Glenn. So the answer is Scott Glenn."),
        ("What was the 2014 population of the city where Lake Wales Medical Center is located?",
         "Lake Wales Medical Center is located in the city of Polk County, Florida. The population of "
         "Polk County in 2014 was 15,140. So the answer is 15,140."),
        ("Who was born first? Jan de Bont or Raoul Walsh?",
         "Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul "
         "Walsh was born the first. So the answer is Raoul Walsh."),
        ("In what country was Lost Gravity manufactured?",
         "The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German "
         "company. So the answer is Germany."),
    ]

    def load_examples(self) -> List[QAExample]:
        rows = hf_load_dataset(_REPO, "fullwiki", split=self.split)
        examples = [QAExample(qid=r["id"], question=r["question"], answers=[r["answer"]],
                              metadata={"type": r["type"], "level": r["level"]})
                    for r in rows if r["answer"]]
        if not examples:
            raise ValueError(f"HotpotQA split '{self.split}' has no gold answers "
                             "(the fullwiki test answers are withheld); use --split validation")
        return examples

    def corpus_passages(self) -> Iterator[Passage]:
        for split in ("train", "validation"):
            for row in hf_load_dataset(_REPO, "distractor", split=split):
                yield from context_passages(row["context"])
