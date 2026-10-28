"""
DRAGIN: Dynamic Retrieval Augmented Generation based on the Information Needs of
LLMs (Su et al., ACL 2024).

Retrieval is triggered by the model's own *real-time information need* and the
query is read off its own self-attention:

  * RIND — when to retrieve (§3.1). For every generated token,
        S_RIND(t_i) = H_i · a_max(i) · s_i                          (Eq. 5)
    H_i is the entropy of the next-token distribution (Eq. 1); a_max(i) the
    largest last-layer attention any later token pays to t_i (Eqs. 2-3); s_i is
    0 for stopwords (Eq. 4). Retrieval fires as soon as some token exceeds θ.
  * QFS — what to retrieve (§3.2). The attention that the triggering token's
    generation step pays to every *preceding* token of the whole context
    (question + output so far) ranks those tokens; the top n non-stopwords, in
    their original order, form the query.
  * Continue (§3.3). The output is truncated before the sentence holding the
    trigger, the top-k passages replace any earlier ones in the prompt, and
    generation resumes from the cut — monitored by RIND again.

Implementation notes, following the released code where the paper is silent:

  * The entropy and attention of a word are summed over its sub-word tokens, and
    within a sentence the attention is renormalised to mean 1 (``a / Σa · |sent|``),
    the scale the θ values of Table 9 were tuned on.
  * The attention pass for RIND covers the generated tokens only.
  * The first sentence generated after a retrieval is never re-triggered: it was
    written with the new evidence in view, and re-triggering it could loop on the
    same query. (The released code keeps exactly that sentence, too.)
  * Generation stops at "So the answer is", or once the output exceeds the
    dataset's generation length (Table 9), as in the released code.

DRAGIN needs token entropies and attention weights, so it runs only on a local
Hugging Face model (``--backend hf``) — the limitation DRAGIN §7 states for
closed APIs.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from RAG.base.base_baseline import (STOP, BaseRAG, SolveResult, answer_end, join_text,
                                    sentence_spans)
from RAG.base.generator import TokenSignal, cut_at_stop
from RAG.metrics import has_answer

_WORD_RE = re.compile(r"\S+")


def is_content_word(word: str) -> bool:
    """s_i of Eq. (4): 1 unless the word is a stopword or has no letters/digits."""
    core = re.sub(r"^\W+|\W+$", "", word).lower()
    return bool(core) and any(c.isalnum() for c in core) and core not in ENGLISH_STOP_WORDS


def query_word(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word)


class DRAGIN(BaseRAG):
    name = "dragin"

    def __init__(self, *args, threshold: Optional[float] = None, top_n: Optional[int] = None,
                 max_rounds: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.generator.supports_internals:
            raise ValueError("DRAGIN reads token entropies and self-attention, which only a "
                             "local Hugging Face model exposes: run it with --backend hf")
        self.threshold = threshold if threshold is not None else self.loader.dragin_threshold
        self.top_n = top_n or self.loader.dragin_top_n
        self.max_rounds = max_rounds

    # ── RIND ───────────────────────────────────────────────────────────────────
    def rind(self, text: str, signals: Sequence[TokenSignal], skip_first: bool
             ) -> Optional[Tuple[int, List[Tuple[int, int]], float]]:
        """First sentence with a token over θ: (sentence index, its trigger word
        spans, the largest score), or None when no information need is detected."""
        words = [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]
        scored = []
        for s, e, w in words:
            toks = [t for t in signals if t.start < e and t.end > s]
            scored.append((s, e, sum(t.entropy for t in toks), sum(t.max_attention for t in toks),
                           1.0 if is_content_word(w) else 0.0))
        for idx, (ss, se) in enumerate(sentence_spans(text)):
            if skip_first and idx == 0:
                continue
            sent = [w for w in scored if w[0] >= ss and w[1] <= se]
            total = sum(w[3] for w in sent)
            if not sent or total <= 0:
                continue
            values = [(w[0], w[1], w[2] * (w[3] / total * len(sent)) * w[4]) for w in sent]
            hits = [(s, e) for s, e, v in values if v > self.threshold]
            if hits:
                return idx, hits, max(v for _, _, v in values)
        return None

    # ── QFS ────────────────────────────────────────────────────────────────────
    def qfs(self, question: str, prior: str, sentence: str,
            hits: List[Tuple[int, int]]) -> str:
        """Query from the attention the trigger tokens pay to the preceding context."""
        head = join_text(question, prior)
        context = join_text(head, sentence)
        offset = len(context) - len(sentence)
        hits = [(s + offset, e + offset) for s, e in hits]
        first_hit = min(s for s, _ in hits)

        offsets, att = self.generator.context_attention(context)
        hit_tokens = [i for i, (s, e) in enumerate(offsets)
                      if e > s and any(s < he and e > hs for hs, he in hits)]
        before = [i for i, (s, e) in enumerate(offsets) if e > s and e <= first_hit]
        if not before or not hit_tokens:
            return " ".join(query_word(w) for w in sentence.split() if is_content_word(w))

        # Token i is generated from position i-1, so row i-1 is the attention that
        # produced it; restrict it to the context before the first trigger word.
        weight = [0.0] * len(offsets)
        for i in hit_tokens:
            row = att[max(i - 1, 0), before]
            row = row / row.sum() if float(row.sum()) > 0 else row
            for j, v in zip(before, row.tolist()):
                weight[j] += v / len(hit_tokens)

        candidates = []
        for m in _WORD_RE.finditer(context[:first_hit]):
            if not is_content_word(m.group()):
                continue
            w = sum(weight[i] for i, (s, e) in enumerate(offsets)
                    if e > s and s < m.end() and e > m.start())
            candidates.append((w, m.start(), query_word(m.group())))
        top = sorted(candidates, key=lambda c: -c[0])[:self.top_n]
        return " ".join(word for _, _, word in sorted(top, key=lambda c: c[1]))

    # ── generation loop (§3.3) ─────────────────────────────────────────────────
    def solve(self, question: str) -> SolveResult:
        text, passages, trace = "", [], []
        protect_first = False
        for _ in range(self.max_rounds):
            prompt = self.build_prompt(question, passages, text)
            completion, signals = self.generator.complete_with_signals(
                prompt, self.max_new_tokens, STOP)
            self.usage.add_call(completion.input_tokens, completion.output_tokens)
            new = cut_at_stop(completion.text, STOP)
            end = answer_end(new)
            if end is not None:
                new = new[:end]
            if not new.strip():
                break

            trigger = self.rind(new, signals, skip_first=protect_first)
            if trigger is None:
                text = join_text(text, new)
                protect_first = False
                trace.append(f"[Generate] {new.strip()}")
            else:
                idx, hits, value = trigger
                spans = sentence_spans(new)
                prefix = new[:spans[idx][0]]
                sentence = new[spans[idx][0]:spans[idx][1]]
                hits = [(s - spans[idx][0], e - spans[idx][0]) for s, e in hits]
                query = self.qfs(question, join_text(text, prefix), sentence, hits)
                passages = self.retrieve(query)            # replaces the previous passages
                text = join_text(text, prefix)
                protect_first = True
                trace.append(f"[RIND] S={value:.2f} > θ={self.threshold} in "
                             f"{sentence.strip()!r} → truncate")
                trace.append(f"[QFS] query={query!r} → {len(passages)} passages")
            if has_answer(text) or self.generator.count_tokens(text) > self.max_new_tokens:
                break

        text = self.finish(question, passages, text)
        return self.result(text, passages, trace)
