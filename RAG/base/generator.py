"""
Text generators for the RAG experiments.

Every retrieval strategy prompts a *frozen* model few-shot and needs plain text
continuation: IRCoT extends a partial chain of thought one sentence at a time,
and DRAGIN truncates its own output and resumes from the cut. A chat endpoint
cannot continue a half-written assistant turn reliably, so each backend exposes:

- ``complete``  — raw continuation of a prompt, greedy (both papers decode
                  greedily for reproducibility), cut at stop strings;
- ``generate``  — the ``BaseLLM`` contract (one chat turn), used by FoT's
                  instruction-style operators (pi_mr, pi_rep), so a ``Generator``
                  can be handed to :class:`baseline.FoT.FoT` as its ``llm``.

Backends:

- ``OllamaGenerator`` — any model served by Ollama, through its native API:
  ``/api/chat`` with the partial CoT as a trailing assistant message, which
  Ollama continues (``chat`` format), or ``/api/generate`` with ``raw: true``
  (``raw`` format). ``num_ctx`` is set explicitly: Ollama's default window
  silently truncates a prompt carrying 15 retrieved paragraphs.
- ``HFGenerator`` — a local Hugging Face checkpoint. The only backend with
  ``supports_internals``: DRAGIN's RIND needs every generated token's entropy and
  the last layer's self-attention, and QFS needs that attention over the whole
  context, neither of which an inference server exposes (DRAGIN §7).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import requests

from models.base import BaseLLM, LLMResponse
from utils.config import get_config


CHAT_FORMAT = ("Answer every question in exactly the format of the example answers: a few "
               "short factual sentences in plain prose, then the final sentence "
               "'So the answer is <answer>.' No lists, headings, bold text or step labels.")


@dataclass
class Prompt:
    """A few-shot prompt, renderable as flat text or as chat turns.

    The content is DRAGIN's template either way. ``raw`` renders it as one text
    (exemplars, then ``Context``, the instruction and ``Question: q\\nAnswer:``)
    for a base model to continue. ``chat`` renders each exemplar as a user /
    assistant exchange and the test question as the last user turn, with the
    partial chain of thought prefilled as the start of the assistant's reply, and
    one system line restating the exemplars' format. Instruct models keep the
    terse format this way; on flat text (or without the line) they drift into
    numbered, step-labelled essays that blow the generation budget and hand IRCoT
    a list marker for a "sentence".

    Attributes:
        demos: (question, CoT) exemplars.
        query: the last user turn: context block, instruction and question.
        partial: the chain of thought so far, which the model must continue.
    """

    demos: List[Tuple[str, str]]
    query: str
    partial: str = ""

    def flat(self) -> str:
        text = "".join(f"Question: {q}\nAnswer: {a}\n" for q, a in self.demos)
        text += f"{self.query}\nAnswer:"
        return text + (" " + self.partial.strip() if self.partial.strip() else "")

    def messages(self) -> List[dict]:
        msgs = [{"role": "system", "content": CHAT_FORMAT}]
        for q, a in self.demos:
            msgs += [{"role": "user", "content": f"Question: {q}"},
                     {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": self.query})
        if self.partial.strip():
            msgs.append({"role": "assistant", "content": self.partial.strip()})
        return msgs


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TokenSignal:
    """One generated token: its character span in the completion and RIND's inputs."""

    start: int
    end: int
    entropy: float      # H_i, Eq. (1)
    max_attention: float  # a_max(i), Eq. (3)


def cut_at_stop(text: str, stop: Optional[Sequence[str]]) -> str:
    cut = len(text)
    for s in stop or ():
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


class Generator(BaseLLM):
    """A frozen model that can both continue raw text and answer a chat turn."""

    supports_internals = False
    prompt_format = "chat"

    def complete(self, prompt: Prompt, max_new_tokens: int,
                 stop: Optional[Sequence[str]] = None) -> Completion:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Token count of ``text``; a word-based estimate where no tokenizer is local."""
        return int(len(text.split()) * 1.3)


# ── Ollama ─────────────────────────────────────────────────────────────────────

class OllamaGenerator(Generator):
    """A model served by Ollama, addressed through the native (non-OpenAI) API."""

    def __init__(self, model: str, prompt_format: str = "chat", num_ctx: int = 8192,
                 seed: int = 0, chat_max_tokens: int = 2048, timeout: float = 600.0) -> None:
        super().__init__(api_key="local", model=model)
        self.prompt_format = prompt_format
        base = get_config()["llm"]["local"]["base_url"].rstrip("/")
        self.base_url = base[:-3] if base.endswith("/v1") else base
        self.num_ctx = num_ctx
        self.seed = seed
        self.chat_max_tokens = chat_max_tokens
        self.timeout = timeout

    def _post(self, route: str, payload: dict) -> dict:
        r = requests.post(f"{self.base_url}/api/{route}", json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama /api/{route} error {r.status_code}: {r.text[:300]}")
        return r.json()

    def complete(self, prompt: Prompt, max_new_tokens: int,
                 stop: Optional[Sequence[str]] = None) -> Completion:
        options = {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx,
                   "num_predict": max_new_tokens, "stop": list(stop or [])}
        if self.prompt_format == "raw":
            data = self._post("generate", {"model": self.model, "prompt": prompt.flat(),
                                           "raw": True, "stream": False, "options": options})
            text = data.get("response", "")
        else:
            # A trailing assistant message is continued, not answered anew (prefill).
            data = self._post("chat", {"model": self.model, "messages": prompt.messages(),
                                       "stream": False, "options": options})
            text = data["message"]["content"]
        return Completion(text=cut_at_stop(text, stop),
                          input_tokens=data.get("prompt_eval_count", 0),
                          output_tokens=data.get("eval_count", 0))

    def generate(self, prompt: str, temperature: float = 0,
                 logprobs: bool = False) -> LLMResponse:
        data = self._post("chat", {
            "model": self.model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "seed": self.seed,
                        "num_ctx": self.num_ctx, "num_predict": self.chat_max_tokens},
        })
        return LLMResponse(content=data["message"]["content"], model_name=self.model,
                           input_tokens=data.get("prompt_eval_count", 0),
                           output_tokens=data.get("eval_count", 0), raw_response=data)


# ── Hugging Face ───────────────────────────────────────────────────────────────

class HFGenerator(Generator):
    """A local Hugging Face causal LM, with access to its logits and attention.

    Calls are serialised with a lock: FoT solves an orbit in threads, and one GPU
    model gains nothing from concurrent ``generate`` calls.
    """

    supports_internals = True

    def __init__(self, model: str, prompt_format: str = "chat", dtype: str = "bfloat16",
                 chat_max_tokens: int = 1024) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(api_key="local", model=model)
        self.prompt_format = prompt_format
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(
            model, dtype=getattr(torch, dtype), device_map="auto", attn_implementation="sdpa")
        self.lm.eval()
        self.chat_max_tokens = chat_max_tokens
        self.pad_id = (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                       else self.tokenizer.eos_token_id)
        self._lock = threading.Lock()

    @property
    def device(self):
        return self.lm.get_input_embeddings().weight.device

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _render(self, prompt: Prompt) -> str:
        if self.prompt_format == "raw":
            return prompt.flat()
        messages = prompt.messages()
        prefill = messages[-1]["role"] == "assistant"
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not prefill,
            continue_final_message=prefill)

    def _generate(self, text: str, max_new_tokens: int, stop: Optional[Sequence[str]],
                  temperature: float = 0.0, signals: bool = False, chat: bool = True):
        # A chat-rendered prompt already carries its special tokens.
        enc = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=not chat).to(self.device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.pad_id,
                      return_dict_in_generate=True, output_logits=signals)
        if stop:
            kwargs.update(stop_strings=list(stop), tokenizer=self.tokenizer)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with self.torch.no_grad():
            out = self.lm.generate(**enc, **kwargs)
        n_prompt = enc["input_ids"].shape[1]
        return out, out.sequences[0, n_prompt:], n_prompt

    def complete(self, prompt: Prompt, max_new_tokens: int,
                 stop: Optional[Sequence[str]] = None) -> Completion:
        with self._lock:
            _, gen, n_prompt = self._generate(self._render(prompt), max_new_tokens, stop,
                                              chat=self.prompt_format == "chat")
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return Completion(cut_at_stop(text, stop), n_prompt, len(gen))

    def generate(self, prompt: str, temperature: float = 0,
                 logprobs: bool = False) -> LLMResponse:
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        with self._lock:
            _, gen, n_prompt = self._generate(chat, self.chat_max_tokens, None, temperature)
        return LLMResponse(content=self.tokenizer.decode(gen, skip_special_tokens=True),
                           model_name=self.model, input_tokens=n_prompt,
                           output_tokens=len(gen))

    # ── internals for DRAGIN ───────────────────────────────────────────────────
    def _last_layer_attention(self, input_ids):
        """Self-attention of the last layer, [heads, T, T] (DRAGIN footnote 2).

        SDPA kernels never materialise the attention matrix, so the model is
        switched to eager attention for this one forward pass.
        """
        self.lm.set_attn_implementation("eager")
        try:
            with self.torch.no_grad():
                out = self.lm(input_ids.unsqueeze(0).to(self.device), output_attentions=True)
        finally:
            self.lm.set_attn_implementation("sdpa")
        return out.attentions[-1][0].float()

    def complete_with_signals(self, prompt: Prompt, max_new_tokens: int,
                              stop: Optional[Sequence[str]] = None
                              ) -> Tuple[Completion, List[TokenSignal]]:
        """Greedy continuation plus, per generated token, H_i and a_max(i).

        H_i is the entropy of the next-token distribution at step i (Eq. 1).
        a_max(i) is the largest attention any *later* generated token pays to
        token i in the last layer (Eqs. 2-3), averaged over heads; as in DRAGIN's
        released code the attention pass covers the generated tokens only.
        """
        torch = self.torch
        with self._lock:
            out, gen, n_prompt = self._generate(self._render(prompt), max_new_tokens, stop,
                                                signals=True, chat=self.prompt_format == "chat")
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
            kept = cut_at_stop(text, stop)
            spans = self._char_spans(gen)
            n = sum(1 for s, _ in spans if s < len(kept)) or len(spans)
            gen = gen[:n]
            logits = torch.stack(out.logits[:n], dim=0)[:, 0, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            entropy = -(logp.exp() * logp).sum(-1)
            att = self._last_layer_attention(gen)                      # [H, n, n]
        # a_max(i) = max_{j > i} A[j, i]: mask the diagonal and above, max over j.
        later = torch.tril(torch.ones(n, n, device=att.device), diagonal=-1)
        amax = (att * later).max(dim=1).values.mean(0)                # [n]
        signals = [TokenSignal(s, e, float(h), float(a))
                   for (s, e), h, a in zip(spans[:n], entropy.tolist(), amax.tolist())]
        return Completion(kept, n_prompt, n), signals

    def _char_spans(self, ids) -> List[Tuple[int, int]]:
        """Character span of each token in the decoded text (incremental decoding)."""
        spans, prev = [], 0
        for k in range(len(ids)):
            cur = len(self.tokenizer.decode(ids[:k + 1], skip_special_tokens=True))
            spans.append((prev, max(prev, cur)))
            prev = max(prev, cur)
        return spans

    def context_attention(self, text: str) -> Tuple[List[Tuple[int, int]], "object"]:
        """Token character offsets of ``text`` and its last-layer attention [T, T],
        averaged over heads — QFS's view of the whole context (§3.2)."""
        enc = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        with self._lock:
            att = self._last_layer_attention(enc["input_ids"][0]).mean(0)
        return [tuple(o) for o in enc["offset_mapping"][0].tolist()], att


def build_generator(backend: str, model: str, prompt_format: str = "chat",
                    num_ctx: int = 8192, hf_dtype: str = "bfloat16") -> Generator:
    if backend == "ollama":
        return OllamaGenerator(model, prompt_format=prompt_format, num_ctx=num_ctx)
    if backend == "hf":
        return HFGenerator(model, prompt_format=prompt_format, dtype=hf_dtype)
    raise ValueError(f"Unknown backend '{backend}' (expected 'ollama' or 'hf')")
