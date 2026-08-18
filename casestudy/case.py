"""
FoT case study: replay the Solve → Falsify → Repair loop, question by question.

``main.py`` reports only accuracy and timing; this script runs the *same* FoT
baseline (same registry, same CLI flags) but instruments it so every step of the
driver loop is recorded and printed:

  * the initial Solve and the answer it produced,
  * for each round k: the falsification regime, the probe and the trusted
    checker's verdict (executable regime) or the whole orbit — variant, the
    answer it received independently, its pull-back and whether rho held
    (metamorphic regime),
  * the damage score s(a) = (violations, orbit support) that decides the archive,
  * the witness that fired, the repair it drove and the new candidate,
  * which candidate was finally returned (fixpoint vs best measured archive
    entry), and whether it matches ground truth,
  * every LLM call (stage, temperature, tokens, latency, and with
    ``--show_prompts`` the full prompt and completion).

Everything is also written to a JSON file for offline analysis: the structured
records (rounds, orbit, witnesses, every LLM call with its full prompt and
completion) *and* a verbatim copy of the terminal transcript — per case in
``cases[i]["transcript"]`` and for the whole session in ``transcript``.

Usage (all of main.py's flags are accepted, --baseline is forced to fot)::

    # first 3 Game-of-24 puzzles (executable regime: trusted checker c_q)
    python casestudy/case.py --benchmark gameof24 --limit 3

    # specific MGSM questions, metamorphic regime, with full prompts
    python casestudy/case.py --benchmark mgsm --language en --indices 0 5 12 \
        --show_prompts

    # BigBenchHard subtask, tighter budget
    python casestudy/case.py --benchmark bigbenchhard \
        --bigbenchhard_task geometric_shapes --limit 2 --fot_budget 2

Author: Egor Morozov
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# The script lives in casestudy/, the package it instruments in the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as main_mod  # noqa: E402
from baseline.FoT.fot import FoT, Damage, Witness, _extract_answer, _parse_probe  # noqa: E402
from baseline.FoT.relations import Variant  # noqa: E402


# ── Instrumented baseline ──────────────────────────────────────────────────────

class TracedFoT(FoT):
    """FoT with the driver loop wired for observation.

    Behaviour is untouched: every override delegates to ``super()`` and only
    records what passed through it. The recorded state is reset at the start of
    every :meth:`run`, so ``calls`` and ``rounds`` always describe the last
    question solved.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tls = threading.local()          # current stage, per thread
        self.calls: List[Dict[str, Any]] = []  # every LLM call, in order
        self.rounds: List[Dict[str, Any]] = []  # one entry per Falsify round
        self._checks: List[Dict[str, Any]] = []  # trusted checker verdicts
        self._drawn: List[Variant] = []        # Sample(C, n) of the current round
        if self._checker is not None:
            self._checker = self._trace_checker(self._checker)

    # ── recording helpers ──────────────────────────────────────────────────────
    @contextmanager
    def _stage(self, name: str):
        previous = getattr(self._tls, "stage", None)
        self._tls.stage = name
        try:
            yield
        finally:
            self._tls.stage = previous

    def _trace_checker(self, inner):
        """Wrap c_q so its verdict is recorded without changing it."""

        def traced(question: str, candidate: str, probe: str = "",
                   *, timeout: float = 10.0):
            result = inner(question, candidate, probe, timeout=timeout)
            self._checks.append({
                "probe": probe,
                "candidate": candidate,
                "verdict": result.verdict,
                "detail": result.detail,
            })
            return result

        return traced

    def _round(self) -> Dict[str, Any]:
        return self.rounds[-1]

    # ── every model call goes through _gen ─────────────────────────────────────
    def _gen(self, prompt: str, context: str, temperature: float):
        stage = getattr(self._tls, "stage", None) or "solve"
        started = time.perf_counter()
        response = super()._gen(prompt, context, temperature)
        self.calls.append({
            "stage": stage,
            "round": len(self.rounds),          # 0 = before the first Falsify
            "temperature": temperature,
            "seconds": round(time.perf_counter() - started, 3),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "prompt": f"{context}\n\n{prompt}" if context else prompt,
            "response": response.content,
        })
        return response

    # ── stage labelling ────────────────────────────────────────────────────────
    def _solve_variant(self, question: str) -> str:
        cached = question in self._answer_cache
        with self._stage("solve_variant"):
            answer = super()._solve_variant(question)
        if cached and self.rounds:
            self._round().setdefault("reused_answers", []).append(answer)
        return answer

    def build_variant(self, relation, question: str):
        with self._stage("build_variant"):
            return super().build_variant(relation, question)

    def _generate_catalogue(self, question: str) -> None:
        with self._stage("generate_catalogue"):
            super()._generate_catalogue(question)

    # ── the driver loop, round by round ────────────────────────────────────────
    def falsify(self, question: str, candidate: str,
                round_index: int = 0) -> Tuple[Optional[Witness], Damage]:
        self.rounds.append({
            "k": round_index + 1,
            "regime": "executable" if self._has_checker else "metamorphic",
            "candidate": _extract_answer(candidate),
            "orbit": [],
            "probe": None,
            "checks": [],
            "witness": None,
            "repaired_answer": None,
        })
        witness, damage = super().falsify(question, candidate, round_index)
        record = self._round()
        record["violations"] = damage.violations
        record["support"] = damage.support
        record["survived"] = witness is None
        if witness is not None:
            record["witness"] = witness.record()
            record["witness"]["orbit_text"] = witness.orbit_text
            record["witness"]["relations_text"] = witness.relations_text
        return witness, damage

    def _falsify_executable(self, question: str, answer: str):
        seen = len(self._checks)
        with self._stage("probe"):
            witness, damage = super()._falsify_executable(question, answer)
        probes = [c for c in self.calls if c["stage"] == "probe"]
        if self.rounds:
            self._round()["probe"] = (_parse_probe(probes[-1]["response"])
                                      if probes else None)
            self._round()["checks"] = self._checks[seen:]
        return witness, damage

    def _sample(self, question: str, answer: str, round_index: int) -> List[Variant]:
        drawn = super()._sample(question, answer, round_index)
        self._drawn = drawn
        return drawn

    def _falsify_metamorphic(self, question: str, answer: str, round_index: int):
        self._drawn = []
        witness, damage = super()._falsify_metamorphic(question, answer, round_index)
        # Rebuild the orbit exactly as the driver computed it: the variants that
        # were drawn, the answers they received (cached by the driver), their
        # pull-backs into q's frame, and rho's verdict.
        orbit: List[Dict[str, Any]] = []
        for variant in self._drawn:
            a_prime = self._answer_cache.get(variant.question, "")
            orbit.append({
                "relation": variant.relation,
                "relation_text": variant.relation_text,
                "source": variant.source,
                "slot": variant.slot,
                "variant": variant.question,
                "answer": a_prime,
                "pulled_back": _safe(lambda: variant.pullback(a_prime)
                                     if variant.pullback else None),
                "expected": _safe(lambda: variant.expected_value(answer)),
                "violated": _safe(lambda: not variant.holds(answer, a_prime)),
            })
        if self.rounds:
            self._round()["orbit"] = orbit
        return witness, damage

    def repair(self, question: str, candidate: str, witness: Witness,
               history: List[str]) -> str:
        with self._stage("repair"):
            repaired = super().repair(question, candidate, witness, history)
        if self.rounds:
            self._round()["repaired_answer"] = _extract_answer(repaired)
        return repaired

    # ── entry point ────────────────────────────────────────────────────────────
    def run(self, question: str, **kwargs: Any):
        self.calls = []
        self.rounds = []
        self._checks = []
        self._drawn = []
        return super().run(question, **kwargs)


def _safe(fn):
    """Evaluate a relation callback defensively — tracing must never break a run."""
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"<error: {exc!r}>"


# ── Rendering ──────────────────────────────────────────────────────────────────

RULE = "=" * 88
THIN = "-" * 88


class Tee:
    """Mirror everything written to stdout into a buffer.

    The console rendering *is* the case study, so the JSON keeps a verbatim copy
    of it — per case and for the whole session — alongside the structured
    records. Nothing is re-rendered: what the file holds is exactly what was
    printed, dataset loader chatter included.
    """

    def __init__(self, stream) -> None:
        self.stream = stream
        self._buffer: List[str] = []
        self._length = 0

    def write(self, text: str) -> int:
        self.stream.write(text)
        self._buffer.append(text)
        self._length += len(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()

    def isatty(self) -> bool:
        return getattr(self.stream, "isatty", lambda: False)()

    def mark(self) -> int:
        """Current position, for slicing out one case's transcript."""
        return self._length

    def text(self, start: int = 0) -> str:
        return "".join(self._buffer)[start:]


def clip(text: Any, limit: Optional[int]) -> str:
    s = "" if text is None else str(text)
    s = s.strip()
    if limit is None or len(s) <= limit:
        return s
    return s[:limit].rstrip() + f" …[+{len(s) - limit} chars]"


def indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines()) or prefix


def print_case(case: Dict[str, Any], limit: Optional[int], show_prompts: bool) -> None:
    """Print one question's full FoT solving process."""
    print("\n" + RULE)
    header = (f"CASE {case['position']}/{case['total']}  ·  {case['benchmark']}"
              f"{'/' + case['subtask'] if case.get('subtask') else ''}"
              f"  ·  problem #{case['index']}  ·  regime={case['regime']}")
    print(header)
    print(RULE)

    print("QUESTION")
    print(indent(clip(case["question"], limit)))
    print("GROUND TRUTH")
    print(indent(clip(case["ground_truth"], limit)))
    if case.get("error"):
        print(f"\n✗ ERROR: {case['error']}")
        return
    if case["catalogue"]:
        print(f"CATALOGUE C   {', '.join(case['catalogue'])}"
              f"   (n={case['probes']}, tau={case['tau']}, K={case['budget']})")
    else:
        print(f"CHECKER c_q   {case['benchmark']}"
              f"{':' + case['subtask'] if case.get('subtask') else ''}"
              f"   (K={case['budget']})")

    print(THIN)
    print(f"[Solve]  → {case['initial_answer']!r}")

    for rnd in case["rounds"]:
        print(THIN)
        print(f"[Falsify k={rnd['k']}]  ({rnd['regime']})  candidate={rnd['candidate']!r}")

        if rnd["regime"] == "executable":
            print(f"   probe (pi_probe):  {clip(rnd['probe'], limit) or '(none parsed)'}")
            for check in rnd["checks"]:
                mark = {"fail": "✗ FAIL", "pass": "✓ PASS"}.get(check["verdict"], "? UNDECIDED")
                print(f"   checker c_q:  {mark}  —  {clip(check['detail'], limit)}")
        else:
            if not rnd["orbit"]:
                print("   orbit: (no applicable relation — FoT degrades to Solve)")
            for i, entry in enumerate(rnd["orbit"], start=1):
                flag = "✗ VIOLATED" if entry["violated"] is True else "✓ holds"
                print(f"   [{i}] {entry['relation']}  ({entry['source']})  {flag}")
                print(f"       rho: {clip(entry['relation_text'], limit)}")
                print(f"       variant q':")
                print(indent(clip(entry["variant"], limit), "         "))
                print(f"       answer a' = {entry['answer']!r}"
                      f"   expected {entry['expected']!r}"
                      f"   pulled back g⁻¹(a') = {entry['pulled_back']!r}")

        print(f"   damage s(a) = (violations={rnd['violations']}, "
              f"orbit support={rnd['support']})")
        if rnd["survived"]:
            print("   → w = ⊥  ·  candidate survived this round (fixpoint)")
        else:
            witness = rnd["witness"]
            print(f"   → witness ({witness['regime']}, sound={witness['sound']}): "
                  f"{clip(witness['summary'], limit)}")
            print(f"[Repair k={rnd['k']}]  → {rnd['repaired_answer']!r}")

    print(THIN)
    mark = "✓ CORRECT" if case["is_correct"] else "✗ WRONG"
    print(f"RESULT  {mark}   final={case['final_answer']!r}")
    print(f"        initial={case['initial_answer']!r}   answer_changed={case['answer_changed']}"
          f"   returned={case['returned']}   accepted_fixpoint={case['accepted_fixpoint']}"
          f"   repairs={case['repairs_used']}")
    print(f"        llm_calls={case['num_llm_calls']}  tokens in/out="
          f"{case['input_tokens']}/{case['output_tokens']}  time={case['seconds']:.1f}s")

    if show_prompts:
        print(THIN)
        print("LLM CALLS")
        for i, call in enumerate(case["llm_calls"], start=1):
            print(f"\n   ── call {i}  [{call['stage']}, round {call['round']}, "
                  f"T={call['temperature']}, {call['seconds']}s, "
                  f"{call['input_tokens']}→{call['output_tokens']} tok]")
            print("   PROMPT:")
            print(indent(clip(call["prompt"], limit), "     | "))
            print("   RESPONSE:")
            print(indent(clip(call["response"], limit), "     > "))


def print_summary(cases: List[Dict[str, Any]]) -> None:
    scored = [c for c in cases if not c.get("error")]
    print("\n" + RULE)
    print("SUMMARY")
    print(RULE)
    print(f"{'#':>5}  {'ok':^4}  {'regime':^12}  {'returned':^9}  {'k':>2}  "
          f"{'flip':^5}  {'calls':>5}  {'time':>7}   answer")
    for case in cases:
        if case.get("error"):
            print(f"{case['index']:>5}  {'ERR':^4}  {clip(case['error'], 60)}")
            continue
        print(f"{case['index']:>5}  {'✓' if case['is_correct'] else '✗':^4}  "
              f"{case['regime']:^12}  {case['returned']:^9}  "
              f"{case['repairs_used']:>2}  "
              f"{'yes' if case['answer_changed'] else 'no':^5}  "
              f"{case['num_llm_calls']:>5}  {case['seconds']:>6.1f}s   "
              f"{clip(case['final_answer'], 40)!r}")

    if not scored:
        return
    correct = sum(1 for c in scored if c["is_correct"])
    flipped = [c for c in scored if c["answer_changed"]]
    helped = sum(1 for c in flipped if c["is_correct"])
    print(THIN)
    print(f"Accuracy            {correct}/{len(scored)} = "
          f"{100.0 * correct / len(scored):.1f}%")
    print(f"Answers changed     {len(flipped)}/{len(scored)}  "
          f"(correct after change: {helped}/{len(flipped) if flipped else 0})")
    print(f"Fixpoints accepted  {sum(1 for c in scored if c['accepted_fixpoint'])}/{len(scored)}")
    print(f"Witnesses fired     {sum(c['repairs_used'] for c in scored)} total")
    print(f"Avg LLM calls       {sum(c['num_llm_calls'] for c in scored) / len(scored):.1f}")
    print(f"Avg time/question   {sum(c['seconds'] for c in scored) / len(scored):.1f}s")


# ── Case assembly ──────────────────────────────────────────────────────────────

def build_case(position: int, total: int, problem, response, result,
               baseline: TracedFoT, elapsed: float, args) -> Dict[str, Any]:
    meta = response.metadata
    return {
        "position": position,
        "total": total,
        "index": problem.index,
        "benchmark": meta.get("task") or args.benchmark,
        "subtask": meta.get("subtask"),
        "question": problem.question,
        "ground_truth": problem.ground_truth,
        "problem_metadata": problem.metadata,
        "regime": meta.get("regime"),
        "catalogue": meta.get("catalogue", []),
        "budget": meta.get("budget"),
        "probes": meta.get("probes"),
        "tau": meta.get("tau"),
        "initial_answer": meta.get("initial_answer"),
        "final_answer": response.final_answer,
        "answer_changed": meta.get("answer_changed"),
        "accepted_fixpoint": meta.get("accepted_fixpoint"),
        "returned": meta.get("returned"),
        "repairs_used": meta.get("repairs_used"),
        "is_correct": bool(result.is_correct),
        "score": result.score,
        "eval_details": result.details,
        "rounds": baseline.rounds,
        "llm_calls": baseline.calls,
        "intermediate_steps": response.intermediate_steps,
        "final_reasoning_trace": response.reasoning_trace,
        "witness_history": meta.get("witness_history", []),
        "num_llm_calls": response.num_llm_calls,
        "input_tokens": response.total_input_tokens,
        "output_tokens": response.total_output_tokens,
        "seconds": elapsed,
    }


def resolve_indices(args, size: int) -> List[int]:
    if args.indices:
        bad = [i for i in args.indices if not 0 <= i < size]
        if bad:
            raise SystemExit(f"index/indices out of range (dataset has {size}): {bad}")
        return list(args.indices)
    start = max(0, args.start)
    return list(range(start, min(size, start + args.limit)))


# ── CLI ────────────────────────────────────────────────────────────────────────

def case_args(parser) -> None:
    group = parser.add_argument_group("Case study")
    group.add_argument("--indices", nargs="+", type=int, default=None,
                       help="Specific problem indices to trace (overrides --start/--limit)")
    group.add_argument("--start", type=int, default=0,
                       help="First problem index to trace")
    group.add_argument("--limit", type=int, default=3,
                       help="Number of problems to trace from --start")
    group.add_argument("--show_prompts", action="store_true",
                       help="Print the full prompt and completion of every LLM call")
    group.add_argument("--max_chars", type=int, default=600,
                       help="Truncate printed texts to this many characters (0 = no limit). "
                            "The JSON output is never truncated.")
    group.add_argument("--out", default=None,
                       help="Path of the JSON trace file "
                            "(default: casestudy/results/<benchmark>_fot_<model>_<ts>.json)")
    group.add_argument("--no_save", action="store_true",
                       help="Do not write the JSON trace file")


def default_out_path(args) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    model_slug = args.model.replace(":", "-").replace("/", "-")
    name = args.benchmark.lower()
    if name == "bigbenchhard":
        name = f"{name}_{args.bigbenchhard_task}"
    elif name == "mgsm":
        name = f"{name}_{args.language}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(here, "results", f"case_{name}_fot_{model_slug}_{stamp}.json")


def main() -> None:
    parser = main_mod.build_parser()
    case_args(parser)
    args = parser.parse_args()

    # This script studies FoT only; swap in the instrumented subclass so the
    # registry, the kwargs extractor and the task/subtask threading are shared
    # with main.py rather than duplicated here.
    args.baseline = "fot"
    main_mod.BASELINE_REGISTRY["fot"] = (TracedFoT, main_mod.BASELINE_REGISTRY["fot"][1])

    if args.benchmark.lower() == "mgsm" and str(args.language).lower() == "all":
        args.language = "en"
        print("[case] --language 'all' is meaningless for a case study → using 'en'")

    # Everything printed from here on is mirrored into the trace file, so the
    # JSON carries the same transcript the terminal showed.
    tee = Tee(sys.stdout)
    sys.stdout = tee
    try:
        run_cases(args, tee)
    finally:
        sys.stdout = tee.stream


def run_cases(args, tee: Tee) -> None:
    evaluator = main_mod.Evaluator(args)
    dataset = evaluator.build_dataset()
    baseline: TracedFoT = evaluator.build_baseline(evaluator.build_client(), dataset)
    indices = resolve_indices(args, len(dataset))
    limit = args.max_chars if args.max_chars > 0 else None

    print(RULE)
    print("FoT CASE STUDY")
    print(RULE)
    print(f"Model:      {args.model}")
    print(f"Benchmark:  {args.benchmark}"
          + (f" / {args.bigbenchhard_task}" if args.benchmark.lower() == "bigbenchhard" else "")
          + (f" / {args.language}" if args.benchmark.lower() == "mgsm" else ""))
    print(f"Dataset:    {len(dataset)} problems, tracing {len(indices)}: {indices}")
    print(f"Baseline:   {baseline!r}")

    system_prompt = dataset.get_system_prompt()
    instruction = dataset.get_instruction()

    cases: List[Dict[str, Any]] = []
    for position, index in enumerate(indices, start=1):
        problem = dataset.get_problem(index)
        started = time.perf_counter()
        mark = tee.mark()
        try:
            response = baseline.run(problem.question,
                                    system_prompt=system_prompt,
                                    instruction=instruction)
        except Exception as exc:
            cases.append({
                "position": position, "total": len(indices), "index": index,
                "benchmark": args.benchmark, "subtask": getattr(args, "bigbenchhard_task", None),
                "question": problem.question, "ground_truth": problem.ground_truth,
                "regime": "n/a", "catalogue": [], "error": repr(exc),
                "seconds": time.perf_counter() - started,
                "llm_calls": baseline.calls, "rounds": baseline.rounds,
            })
            print_case(cases[-1], limit, args.show_prompts)
            cases[-1]["transcript"] = tee.text(mark)
            continue
        elapsed = time.perf_counter() - started
        result = dataset.evaluate_answer(response.final_answer, problem.ground_truth)
        case = build_case(position, len(indices), problem, response, result,
                          baseline, elapsed, args)
        cases.append(case)
        print_case(case, limit, args.show_prompts)
        case["transcript"] = tee.text(mark)

    summary_mark = tee.mark()
    print_summary(cases)
    summary_text = tee.text(summary_mark)

    if not args.no_save:
        path = args.out or default_out_path(args)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "metadata": {
                    "model": args.model,
                    "benchmark": args.benchmark,
                    "subtask": (args.bigbenchhard_task
                                if args.benchmark.lower() == "bigbenchhard" else None),
                    "language": args.language if args.benchmark.lower() == "mgsm" else None,
                    "baseline": repr(baseline),
                    "indices": indices,
                    "fot": {
                        "budget": args.fot_budget, "probes": args.fot_probes,
                        "tau": args.fot_tau, "relations": args.fot_relations,
                        "execute_code": args.fot_execute_code,
                        "orbit_majority": args.fot_orbit_majority,
                        "witness_history": args.fot_witness_history,
                        "generate_relations": args.fot_generate_relations,
                        "solve_temperature": args.fot_solve_temp,
                        "falsify_temperature": args.fot_falsify_temp,
                        "repair_temperature": args.fot_repair_temp,
                    },
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "show_prompts": args.show_prompts,
                    "max_chars": args.max_chars,
                },
                "cases": cases,
                # Verbatim copies of what the terminal showed: the whole session,
                # and the summary on its own. Each case carries its own slice in
                # cases[i]["transcript"].
                "transcript": tee.text(),
                "summary_transcript": summary_text,
            }, handle, indent=2, ensure_ascii=False, default=str)
        print(f"\nTrace saved → {path}")


if __name__ == "__main__":
    main()
