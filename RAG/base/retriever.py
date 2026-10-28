"""
The BM25 retriever shared by every retrieval strategy (Table 4.6).

All strategies share one retriever and one corpus, so a difference in the table
reflects *when* and *with what query* a strategy retrieves, never what it can
retrieve. BM25 follows both papers (DRAGIN §4.5 chooses it over dense retrievers
after Ram et al. 2023; IRCoT §4.1); it is implemented with ``bm25s`` instead of an
Elasticsearch server, with Elasticsearch's defaults (Lucene BM25, k1 = 1.2,
b = 0.75, lowercasing, no stemming). English stopwords are dropped at index time:
their IDF is ~0 so the ranking barely moves, and it keeps a query from scanning
every posting list of a 21M-passage index.

Two corpora (``--corpus``):

- ``wiki``    — the DPR December-2018 Wikipedia dump, 21M passages of 100 words
                (``psgs_w100.tsv``), the corpus DRAGIN uses for all four of its
                datasets. One index serves every dataset.
- ``dataset`` — IRCoT's construction (App. A): the context paragraphs shipped with
                the dataset itself, pooled over its splits. One index per dataset;
                small and quick to build.

Indexes live under ``RAG/data/index/<corpus>`` and are built on first use. The
passages are kept outside the BM25 index in a JSONL file with a byte-offset table,
so a lookup reads only the k lines it needs, even for the 13 GB Wikipedia dump.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np

from RAG.base.base_data_loader import BaseDataLoader, Passage

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WIKI_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
WIKI_PATH = DATA_DIR / "corpus" / "wiki" / "psgs_w100.tsv.gz"

_CHUNK = 200_000  # passages tokenised per batch while indexing


def wiki_passages(path: Path = WIKI_PATH) -> Iterator[Passage]:
    """Stream the DPR dump (columns: id, text, title), downloading it if absent."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading the DPR Wikipedia dump (4.7 GB) → {path}")
        urllib.request.urlretrieve(WIKI_URL, path)
    csv.field_size_limit(sys.maxsize)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            if len(row) == 3:
                yield Passage(title=row[2], text=row[1])


class PassageStore:
    """Passages on disk as JSONL, addressed by row number through a byte-offset table."""

    def __init__(self, directory: Path) -> None:
        self.path = directory / "passages.jsonl"
        self.offsets = np.load(directory / "offsets.npy", mmap_mode="r")
        self._file = open(self.path, "rb")
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, i: int) -> Passage:
        with self._lock:
            self._file.seek(int(self.offsets[i]))
            row = json.loads(self._file.readline())
        return Passage(title=row["title"], text=row["text"])

    @staticmethod
    def write(passages: Iterable[Passage], directory: Path) -> Iterator[str]:
        """Write unique passages and yield the text each one is indexed on."""
        seen = set()
        offsets: List[int] = []
        with open(directory / "passages.jsonl", "wb") as f:
            for p in passages:
                digest = hashlib.blake2b(f"{p.title}\x00{p.text}".encode(), digest_size=8).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                offsets.append(f.tell())
                f.write(json.dumps({"title": p.title, "text": p.text}, ensure_ascii=False).encode() + b"\n")
                yield p.render()
        np.save(directory / "offsets.npy", np.asarray(offsets, dtype=np.int64))


class BM25Retriever:
    """BM25 over one corpus. ``retrieve`` is thread-safe (FoT solves orbits in parallel)."""

    def __init__(self, directory: Path) -> None:
        import bm25s
        from bm25s.tokenization import Tokenizer

        self.directory = directory
        self.index = bm25s.BM25.load(str(directory), mmap=True)
        # The index-time tokenizer's splitter and stopwords, applied directly: its
        # own query path rebuilds a reverse vocabulary (millions of words) per call.
        tokenizer = Tokenizer(stopwords="en", stemmer=None)
        self._split = tokenizer.splitter
        self._stopwords = set(tokenizer.stopwords)
        self.store = PassageStore(directory)

    def retrieve(self, query: str, k: int) -> List[Passage]:
        """Top-k passages by BM25 score (only passages sharing a term with the query).

        Scores are summed straight from the index's posting lists — the term
        columns of its precomputed CSC score matrix — touching only the passages
        that contain a query term. ``bm25s.BM25.retrieve`` returns the same ranking
        and scores but, without numba, materialises a score for every one of the
        21M Wikipedia passages: ~2 s a query against milliseconds here.
        """
        vocab = self.index.vocab_dict
        terms = {vocab[w] for w in self._split(query.lower())
                 if w not in self._stopwords and w in vocab}
        if not terms:
            return []
        scores = self.index.scores
        data, indices, indptr = scores["data"], scores["indices"], scores["indptr"]
        cols = [slice(indptr[t], indptr[t + 1]) for t in terms]
        docs = np.concatenate([indices[c] for c in cols])
        weights = np.concatenate([data[c] for c in cols])
        if docs.size == 0:
            return []
        unique, inverse = np.unique(docs, return_inverse=True)
        totals = np.bincount(inverse, weights=weights)
        k = min(k, totals.size)
        top = np.argpartition(-totals, k - 1)[:k]
        top = top[np.argsort(-totals[top], kind="stable")]
        return [self.store[int(unique[i])] for i in top if totals[i] > 0]

    # ── construction ───────────────────────────────────────────────────────────
    @staticmethod
    def build(passages: Iterable[Passage], directory: Path,
              k1: float = 1.2, b: float = 0.75) -> None:
        import bm25s
        from bm25s.tokenization import Tokenizer

        directory.mkdir(parents=True, exist_ok=True)
        tokenizer = Tokenizer(stopwords="en", stemmer=None)
        ids: List[List[int]] = []
        batch: List[str] = []
        t0 = time.time()
        for text in PassageStore.write(passages, directory):
            batch.append(text)
            if len(batch) == _CHUNK:
                ids.extend(tokenizer.tokenize(batch, update_vocab=True, return_as="ids",
                                              show_progress=False))
                batch = []
                print(f"  tokenised {len(ids):,} passages ({time.time() - t0:.0f}s)", flush=True)
        if batch:
            ids.extend(tokenizer.tokenize(batch, update_vocab=True, return_as="ids",
                                          show_progress=False))
        print(f"  indexing {len(ids):,} passages ...", flush=True)
        index = bm25s.BM25(k1=k1, b=b, method="lucene")
        index.index((ids, tokenizer.get_vocab_dict()), show_progress=False)
        index.save(str(directory), show_progress=False)
        tokenizer.save_vocab(str(directory))
        (directory / "DONE").write_text(f"{len(ids)} passages, built in {time.time() - t0:.0f}s\n")
        print(f"  index ready → {directory} ({time.time() - t0:.0f}s)", flush=True)


def index_dir(corpus: str, loader: Optional[BaseDataLoader]) -> Path:
    return DATA_DIR / "index" / ("wiki" if corpus == "wiki" else loader.name)


def load_retriever(corpus: str, loader: BaseDataLoader) -> BM25Retriever:
    """Open the index for ``corpus``, building it first if it does not exist yet."""
    directory = index_dir(corpus, loader)
    if not (directory / "DONE").exists():
        print(f"Building the BM25 index for corpus '{directory.name}' (one-time) ...")
        source = wiki_passages() if corpus == "wiki" else loader.corpus_passages()
        BM25Retriever.build(source, directory)
    return BM25Retriever(directory)
