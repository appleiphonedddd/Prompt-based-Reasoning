"""
Evaluation entry point for the knowledge-intensive QA experiments (Table 4.6).

One run fills one cell of the table: a dataset × a retrieval strategy × a
reasoning mode, on one LLM.

    python RAG/eval.py --dataset hotpotqa --retrieval ircot --reasoning fot \
        --model qwen2.5:32b

Retrieval: none | single | ircot | dragin      Reasoning: cot | fot
Datasets:  hotpotqa | musique | 2wikimqa | iirc | strategyqa
Metrics:   EM / F1, except StrategyQA (accuracy)

Every run writes, under ``RAG/results/``:

- ``{dataset}_{retrieval}_{reasoning}_{model}_{ts}.jsonl`` — one record per
  question (prediction, gold, scores, CoT, trace, cost, FoT witnesses), written as
  it goes; pass it back with ``--resume`` to continue an interrupted run;
- ``... .json`` — the summary (metrics, cost per question, all hyperparameters).

DRAGIN needs token entropies and attention, so it runs only with ``--backend hf``
(a local Hugging Face checkpoint). For a fair table, run every row of a model
with the same backend.
"""

from __future__ import annotations

import os
import sys

# Run from anywhere, as `python RAG/eval.py` or `python -m RAG.eval`. The script's
# own directory must not be on sys.path: RAG/datasets would shadow the Hugging
# Face `datasets` package. The repository root must be, and must be the working
# directory, since utils.config reads ./config.yaml.
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_RAG_DIR)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _RAG_DIR]
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

from RAG.base.base_baseline import Usage
from RAG.base.generator import build_generator
from RAG.base.retriever import load_retriever
from RAG.baseline import RETRIEVAL_REGISTRY
from RAG.datasets import DATASET_REGISTRY, get_loader
from RAG.metrics import ScoreBoard, mean, score
from RAG.reasoning.cot import CoTReasoning, Prediction
from RAG.reasoning.fot import FoTReasoning

logging.getLogger().setLevel(logging.ERROR)

RESULTS_DIR = os.path.join(_RAG_DIR, "results")


# ── Evaluator ─────────────────────────────────────────────────────────────────
class Evaluator:
    """Runs one (dataset, retrieval, reasoning, model) cell of Table 4.6."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.loader = get_loader(args.dataset)(split=args.split)
        self.validate()

    def validate(self) -> None:
        a = self.args
        if a.retrieval == "dragin" and a.backend != "hf":
            raise ValueError("DRAGIN reads token entropies and self-attention, which only a "
                             "local Hugging Face model exposes: use --backend hf "
                             "--model <hf-checkpoint>")

    def build_reasoner(self, generator, retriever, usage: Usage):
        a = self.args
        cls, extract_kwargs = RETRIEVAL_REGISTRY[a.retrieval]
        method = cls(generator, retriever, self.loader, topk=a.retrieve_topk,
                     max_new_tokens=a.generate_max_length, usage=usage, **extract_kwargs(a))
        if a.reasoning == "cot":
            return CoTReasoning(method)
        return FoTReasoning(
            method,
            budget=a.fot_budget,
            probes=a.fot_probes,
            tau=a.fot_tau,
            falsify_temperature=a.fot_falsify_temp,
            repair_temperature=a.fot_repair_temp,
            carry_witness_history=a.fot_witness_history,
            orbit_majority=a.fot_orbit_majority,
            relations=a.fot_relations,
            repair_passages=a.fot_repair_passages,
        )

    def _paths(self) -> str:
        if self.args.resume:
            return self.args.resume
        os.makedirs(RESULTS_DIR, exist_ok=True)
        a = self.args
        model_slug = a.model.replace(":", "-").replace("/", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(RESULTS_DIR, f"{a.dataset}_{a.retrieval}_{a.reasoning}_"
                                         f"{model_slug}_{timestamp}.jsonl")

    def run(self) -> None:
        a = self.args
        if a.build_index_only:
            load_retriever(a.corpus, self.loader)
            return

        examples = self.loader.load(a.num_samples, a.shuffle, a.seed)
        print(f"\n--- Starting Evaluation ---")
        print(f"Model:      {a.model} ({a.backend})")
        print(f"Dataset:    {a.dataset} [{self.loader.split}]  metric={self.loader.metric}")
        print(f"Retrieval:  {a.retrieval}  (corpus={a.corpus}, top-k={a.retrieve_topk})")
        print(f"Reasoning:  {a.reasoning}")
        print(f"Questions:  {len(examples)}")

        retriever = None if a.retrieval == "none" else load_retriever(a.corpus, self.loader)
        generator = build_generator(a.backend, a.model, prompt_format=a.prompt_format,
                                    num_ctx=a.num_ctx, hf_dtype=a.hf_dtype)

        pred_path = self._paths()
        records: List[Dict[str, Any]] = []
        if a.resume and os.path.exists(pred_path):
            with open(pred_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            print(f"Resuming:   {len(records)} questions already in {pred_path}")
        done = {r["qid"] for r in records}
        todo = [e for e in examples if e.qid not in done]

        board = ScoreBoard(self.loader.metric)
        for r in records:
            board.record(r["scores"])
        lock = threading.Lock()
        n = len(examples)

        def answer(example) -> Dict[str, Any]:
            usage = Usage()
            t0 = time.perf_counter()
            try:
                pred: Prediction = self.build_reasoner(generator, retriever, usage).run(
                    example.question)
                error = None
            except Exception as exc:  # scored incorrect; the run continues
                pred = Prediction("", "", [], [], usage.as_dict())
                error = repr(exc)
            return {
                "qid": example.qid, "question": example.question, "gold": example.answers,
                "prediction": pred.answer, "scores": score(pred.answer, example.answers,
                                                           self.loader.metric),
                "time_s": round(time.perf_counter() - t0, 3), **pred.usage,
                "error": error, "output": pred.text, "trace": pred.trace,
                "passages": pred.passage_titles, "fot": pred.metadata or None,
                "metadata": example.metadata,
            }

        with open(pred_path, "a", encoding="utf-8") as sink, \
                ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            futures = [pool.submit(answer, e) for e in todo]
            for future in as_completed(futures):
                rec = future.result()
                with lock:
                    records.append(rec)
                    board.record(rec["scores"])
                    sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sink.flush()
                    ok = max(rec["scores"].values()) == 1.0
                    status = f"✗ ERROR {rec['error']}" if rec["error"] else ("✓" if ok else "✗")
                    print(f"  [{len(records)}/{n}] {status} ({rec['time_s']:.1f}s) "
                          f"pred={rec['prediction']!r} gold={rec['gold'][0]!r}   "
                          f"[{board.summary()}]", flush=True)

        self.report(records, board, pred_path)

    def report(self, records: List[Dict[str, Any]], board: ScoreBoard, pred_path: str) -> None:
        a = self.args
        metrics = {k: round(v, 4) for k, v in board.means().items()}
        efficiency = {
            "avg_time_per_question_s": round(mean([r["time_s"] for r in records]), 3),
            "avg_llm_calls": round(mean([r["llm_calls"] for r in records]), 3),
            "avg_retrievals": round(mean([r["retrievals"] for r in records]), 3),
            "avg_input_tokens": round(mean([r["input_tokens"] for r in records]), 1),
            "avg_output_tokens": round(mean([r["output_tokens"] for r in records]), 1),
            "errors": sum(1 for r in records if r["error"]),
        }
        summary: Dict[str, Any] = {
            "metadata": {
                "model": a.model, "backend": a.backend, "dataset": a.dataset,
                "split": self.loader.split, "retrieval": a.retrieval, "reasoning": a.reasoning,
                "corpus": a.corpus, "num_questions": len(records),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            "metrics": metrics,
            "efficiency": efficiency,
            "hyperparameters": {k: v for k, v in vars(a).items() if k != "resume"},
            "predictions": os.path.relpath(pred_path, _ROOT),
        }
        fot = [r["fot"] for r in records if r.get("fot")]
        if fot:
            summary["fot"] = {
                "answer_changed_rate": round(mean([float(m["answer_changed"]) for m in fot]), 4),
                "accepted_fixpoint_rate": round(mean([float(m["accepted_fixpoint"]) for m in fot]), 4),
                "avg_repairs": round(mean([m["repairs_used"] for m in fot]), 3),
            }

        print("\n" + "=" * 70)
        print(f"{a.dataset} | {a.retrieval} + {a.reasoning} | {a.model}")
        print("  " + "  ".join(f"{k.upper()}: {v:.4f}" for k, v in metrics.items()))
        print(f"  Avg time/question: {efficiency['avg_time_per_question_s']:.2f}s   "
              f"LLM calls: {efficiency['avg_llm_calls']:.2f}   "
              f"retrievals: {efficiency['avg_retrievals']:.2f}   "
              f"errors: {efficiency['errors']}")
        if fot:
            print(f"  FoT: answer changed {summary['fot']['answer_changed_rate']:.2%}, "
                  f"accepted fixpoint {summary['fot']['accepted_fixpoint_rate']:.2%}")
        summary_path = pred_path[:-len(".jsonl")] + ".json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Results saved → {os.path.relpath(summary_path, _ROOT)}")


# ── CLI argument groups ───────────────────────────────────────────────────────

def general_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="qwen2.5:32b",
                        help="Ollama model name (--backend ollama) or Hugging Face "
                             "checkpoint (--backend hf)")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "hf"],
                        help="ollama: served model, raw completion API. hf: local "
                             "transformers model (required by DRAGIN)")
    parser.add_argument("--dataset", default="hotpotqa", choices=list(DATASET_REGISTRY))
    parser.add_argument("--retrieval", default="single", choices=list(RETRIEVAL_REGISTRY),
                        help="none | single | ircot | dragin")
    parser.add_argument("--reasoning", default="cot", choices=["cot", "fot"])
    parser.add_argument("--split", default=None,
                        help="Labelled split (default: validation; train for StrategyQA, "
                             "whose test file has no labels)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Questions to evaluate, from the front of the split (DRAGIN "
                             "uses 1000); -1 for the whole split")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the split (with --seed) before taking --num_samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus", default="wiki", choices=["wiki", "dataset"],
                        help="wiki: DPR Wikipedia, 21M passages (DRAGIN). dataset: the "
                             "dataset's own context paragraphs (IRCoT App. A)")
    parser.add_argument("--retrieve_topk", type=int, default=3,
                        help="Passages per retrieval call (DRAGIN Table 8: 3)")
    parser.add_argument("--prompt_format", default="chat", choices=["chat", "raw"],
                        help="chat: exemplars as user/assistant turns, partial CoT prefilled "
                             "(instruct models). raw: DRAGIN's flat text template (base models)")
    parser.add_argument("--generate_max_length", type=int, default=None,
                        help="New tokens per generation (default: DRAGIN Table 9 per dataset)")
    parser.add_argument("--num_ctx", type=int, default=8192,
                        help="Ollama context window (its default silently truncates long prompts)")
    parser.add_argument("--hf_dtype", default="bfloat16")
    parser.add_argument("--workers", type=int, default=1,
                        help="Questions answered concurrently (Ollama: set OLLAMA_NUM_PARALLEL too)")
    parser.add_argument("--resume", default=None,
                        help="Predictions .jsonl of an interrupted run to continue")
    parser.add_argument("--build_index_only", action="store_true",
                        help="Build the BM25 index for --corpus (and --dataset) and exit")


def ircot_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("IRCoT")
    g.add_argument("--ircot_topk", type=int, default=None,
                   help="Paragraphs per retrieve step K (paper tunes {2,4,6,8}; "
                        "default: --retrieve_topk)")
    g.add_argument("--ircot_max_steps", type=int, default=8,
                   help="Maximum reasoning steps (paper: 8)")
    g.add_argument("--ircot_max_paragraphs", type=int, default=15,
                   help="Cap on collected paragraphs (paper: 15)")
    g.add_argument("--ircot_reader", default="cot", choices=["cot", "none"],
                   help="cot: separate CoT reader over the collected paragraphs (paper); "
                        "none: read the answer off the retrieval CoT (App. F ablation)")


def dragin_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("DRAGIN")
    g.add_argument("--dragin_threshold", type=float, default=None,
                   help="RIND threshold θ (default: DRAGIN Table 9, Llama2-13b-chat column)")
    g.add_argument("--dragin_top_n", type=int, default=None,
                   help="Tokens QFS keeps for the query (default: Table 9)")
    g.add_argument("--dragin_max_rounds", type=int, default=10,
                   help="Safety cap on generate/retrieve rounds per question")


def fot_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("FoT")
    g.add_argument("--fot_budget", type=int, default=3,
                   help="Budget K: maximum Falsify→Repair iterations")
    g.add_argument("--fot_probes", type=int, default=3,
                   help="Probes n: relations drawn from the QA catalogue per attempt (orbit "
                        "size). Each variant is answered by the full retrieval strategy")
    g.add_argument("--fot_tau", type=int, default=2,
                   help="Refutation threshold τ: violated relations needed before a repair")
    g.add_argument("--fot_relations", nargs="+", default=None,
                   help="Restrict the catalogue to these relations (qa_swap_operands "
                        "qa_decompose qa_paraphrase qa_cloze)")
    g.add_argument("--no_fot_orbit_majority", action="store_false", dest="fot_orbit_majority",
                   help="Ablation: drop the orbit-majority acceptance rule")
    g.add_argument("--fot_falsify_temp", type=float, default=0.7,
                   help="Temperature for building variants (pi_mr)")
    g.add_argument("--fot_repair_temp", type=float, default=0.0,
                   help="Temperature for witness-guided Repair")
    g.add_argument("--no_fot_witness_history", action="store_false", dest="fot_witness_history",
                   help="Do not carry past witnesses into Repair")
    g.add_argument("--fot_repair_passages", type=int, default=10,
                   help="Passages from the orbit's retrievals shown to Repair")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Knowledge-intensive QA: retrieval strategy × reasoning (Table 4.6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    general_args(parser)
    ircot_args(parser)
    dragin_args(parser)
    fot_args(parser)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    for key, val in vars(args).items():
        print(f"{key} = {val}")

    Evaluator(args).run()
