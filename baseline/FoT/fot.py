"""
Falsification-of-Thought (FoT): reasoning by metamorphic self-refutation.

FoT makes *self-refutation* the engine of reasoning. Instead of generating and
selecting, it generates a candidate and then repeatedly tries to *break* it,
using each successful break to drive a targeted repair. The guiding principle is
to **refute by construction, not by verdict**: the model is refuted by its own
outputs, never by its own self-assessment.

The driver loop (Algorithm 4 of the paper):

    a ← Solve(q); H ← ∅; S ← {(a, 0)}      # candidate; witness history; archive
    for k in 1..K:                          # budget K
        w ← Falsify(q, a)                   # executable or metamorphic witness
        if w = ⊥:
            return a                        # survived falsification: accept fixpoint
        H ← H ∪ {w}
        a ← Repair(q, a, w, H)              # witness-guided repair
        S ← S ∪ {(a, rounds survived)}
    return argmax_{(a,s) ∈ S} s             # budget exhausted: best-surviving candidate

FoT composes three operators, each a prompt to a single frozen model M:

  * Solve  : q → a            produce a candidate (any base reasoner; CoT here).
  * Falsify: (q, a) → w | ⊥   construct a witness that a is wrong, or ⊥.
  * Repair : (q, a, w, H) → a' revise a to resolve w.

The falsifier (Algorithm 2) operates in one of two regimes, selected by the
task-level predicate ``HasChecker(q)`` — FIXED PER BENCHMARK, not decided at run
time by the model (see :mod:`baseline.FoT.checkers`):

  * **Executable falsification** — when the benchmark exposes a cheap, decisive
    checker c_q (arithmetic evaluation for Game of 24 and BBH multi-step
    arithmetic, program execution for CRUXEval, ``sat`` for Python Puzzles). The
    model only *proposes a probe* (pi_probe) — it is forbidden to give a verdict;
    the trusted external c_q decides. A failure is therefore a SOUND witness.

  * **Metamorphic falsification** — when no checker exists. The query is
    transformed by a fixed catalogue C of semantics-preserving relations
    (:mod:`baseline.FoT.relations`), each variant is solved *independently*, and
    the witness is a concrete disagreement inside the resulting orbit of answers.
    By Proposition 1 a violated valid relation certifies that at least one answer
    in the orbit is wrong, with no appeal to the model's self-assessment: the
    residual unsoundness moves from an unauditable per-instance hallucination to
    the design-time validity of C.

Three design decisions carry the weight of the metamorphic branch:

  * *Independence* (Remark 1) — the candidate never appears in the prompt that
    solves a variant, so the follow-up answer is evidence rather than an echo.
    This is structural: pi_solve simply has no candidate slot.
  * *Directionality* (Remark 2) — every relation in C must be non-decreasing in
    reliability, so a disagreement is more likely to indict a than a'.
  * *Corroboration* (Remark 3) — repair fires only when at least τ relations are
    violated **and** a is outside the majority of the orbit. The pilot's inverted
    asymmetry (one violation triggers repair, unanimity required to accept) is
    recovered as an ablation with ``tau=1`` and ``use_majority=False``.

This replaces the pilot's assertion-based semantic regime (pi_nec + pi_chk and the
``ReCheckable`` guard), in which the model asserted that the candidate violated a
self-generated necessary condition — sound-looking but, being a verdict on its own
work, the source of the MGSM regression reported in the paper. There is no
judgement template in the metamorphic branch: rho(a, a') is evaluated by the
driver, not by the model.

Cost: O(K·(1+n)) model calls (n = probes per round), the same order as the pilot
and far below the recursive expansion of ToT/GoT.

Reference: "Falsification-of-Thought: Reasoning by Metamorphic Self-Refutation".
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from baseline.basebaseline import BaseBaseline, BaselineResponse
from baseline.FoT.checkers import CheckResult, Checker, get_checker
from baseline.FoT.relations import (
    Relation,
    Variant,
    equality_variant,
    get_catalogue,
    normalize_answer,
    parse_generated_catalogue,
)
from models.base import BaseLLM


# ── Tagged-output parsers (Extract) ─────────────────────────────────────────────

def _extract_answer(text: str) -> str:
    """Pull the final answer from the paper's ``ANSWER:`` tag.

    Falls back to a legacy ``### Answer`` section and finally the last non-empty
    line, so a parseable answer is recovered even when the model omits the tag.
    """
    matches = list(re.finditer(r"ANSWER:\s*(.+?)(?:\n\s*\n|\Z)", text,
                               re.DOTALL | re.IGNORECASE))
    if matches and matches[-1].group(1).strip():
        return matches[-1].group(1).strip()
    m = re.search(r"###\s*Answer\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def _parse_probe(text: str) -> str:
    """Parse the ``PROBE:`` line (pi_probe output)."""
    m = re.search(r"PROBE:\s*(.+)", text, re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _parse_variant(text: str) -> str:
    """Parse the ``VARIANT:`` block (pi_mr output); ``RELATION:`` ends it.

    The relation line is ignored on purpose: rho comes from the catalogue, which
    was audited at design time, never from what the model says about it here.
    """
    m = re.search(r"VARIANT:\s*(.*?)(?=^\s*RELATION:|\Z)", text,
                  re.DOTALL | re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


@dataclass
class Witness:
    """A constructive artifact demonstrating that a candidate is wrong.

    Attributes:
        text: human-readable, self-contained description of the failure, used for
            the witness history H and for the run metadata.
        sound: True iff produced by the executable regime, where the trusted
            external checker c_q reported a real failure — in which case it
            certifies the candidate is incorrect. A metamorphic witness is sound
            only *relative to the relation* (Proposition 1): it certifies that
            some answer in the orbit is wrong, not necessarily the candidate's.
        regime: "executable" or "metamorphic".
        orbit_block: the <orbit> slot of pi_rep — the equivalent formulations and
            the answers they received (executable: the probe and the checker's
            detail d).
        relations_block: the <relations> slot of pi_rep.
        violations: names of the relations that were violated.
        satisfied: how many probes of this round the candidate survived (the
            archive's corroboration score).
    """

    text: str
    sound: bool
    regime: str
    orbit_block: str = ""
    relations_block: str = ""
    violations: List[str] = field(default_factory=list)
    satisfied: int = 0


class FoT(BaseBaseline):
    """Falsification-of-Thought prompting baseline."""

    def __init__(
        self,
        llm: BaseLLM,
        task: Optional[str] = None,
        subtask: Optional[str] = None,
        budget: int = 3,
        probes: int = 3,
        tau: int = 2,
        use_majority: bool = True,
        solve_temperature: float = 0.0,
        falsify_temperature: float = 0.7,
        repair_temperature: float = 0.0,
        execute_code: bool = True,
        code_timeout: float = 10.0,
        carry_witness_history: bool = True,
        relations: Optional[List[str]] = None,
        generate_relations: bool = False,
    ) -> None:
        """Initialize the FoT baseline.

        Args:
            llm: An instance of a BaseLLM subclass.
            task: Benchmark name (e.g. "gameof24", "mgsm"). Selects the trusted
                checker c_q and the relation catalogue C, fixing ``HasChecker(q)``
                for the whole run.
            subtask: Sub-benchmark, e.g. the BigBenchHard task name. Checker and
                catalogue lookup are most-specific-first.
            budget: K — maximum number of Falsify→Repair iterations.
            probes: n — relations drawn from C per falsification round (the
                executable regime always issues exactly one probe).
            tau: τ — how many relations must be violated before a repair fires
                (Remark 3). Ignored in the executable regime, where one sound
                failure is decisive.
            use_majority: Apply the orbit-majority acceptance rule: refuse to
                repair a candidate that the bulk of the orbit agrees with.
                ``tau=1, use_majority=False`` recovers the pilot's behaviour.
            solve_temperature: Sampling temperature for Solve (initial and
                variant solves).
            falsify_temperature: Sampling temperature for the falsifier's own
                calls (pi_probe, pi_mr).
            repair_temperature: Sampling temperature for witness-guided Repair.
            execute_code: If True, use the executable regime on benchmarks that
                have a trusted checker c_q. If False, force the metamorphic
                regime everywhere. The checker runs the benchmark's own reference
                code (CRUXEval functions, Python-Puzzles ``sat``) in a
                timeout-bounded subprocess; disable on an untrusted host.
            code_timeout: Per-execution timeout in seconds for the checker.
            carry_witness_history: If True, carry the history H of past witnesses
                into Repair so fixes do not reintroduce previously resolved
                failures (discouraging oscillation).
            relations: Restrict C to these relation names (see
                :mod:`baseline.FoT.relations`); None uses the whole catalogue.
            generate_relations: Ablation — build C with pi_mr-gen from the first
                question of the run instead of using the hand-written catalogue.
        """
        super().__init__(llm, baseline_name="FoT")
        self.task = task
        self.subtask = subtask
        self.budget = budget
        self.probes = probes
        self.tau = tau
        self.use_majority = use_majority
        self.solve_temperature = solve_temperature
        self.falsify_temperature = falsify_temperature
        self.repair_temperature = repair_temperature
        self.execute_code = execute_code
        self.code_timeout = code_timeout
        self.carry_witness_history = carry_witness_history
        self.generate_relations = generate_relations

        # HasChecker(q): fixed per benchmark, not decided at run time by the model.
        self._checker: Optional[Checker] = (
            get_checker(task, subtask) if execute_code else None)
        self._has_checker = self._checker is not None

        # C: also fixed per benchmark. Empty in the executable regime, where the
        # checker decides and no relation is needed.
        self._catalogue: List[Relation] = (
            [] if self._has_checker else get_catalogue(task, subtask, relations))
        self._catalogue_generated = False

        # Context split (persona scoping): the answer-producing operators (Solve,
        # Repair) receive the full task framing + answer-format directives so their
        # output stays parseable; the falsifier's own calls receive only a one-line
        # task description, so the format persona never conflicts with "propose a
        # probe" / "rewrite this problem".
        self._solve_ctx = ""   # system_prompt + instruction
        self._task_ctx = ""    # one-line task description only

    # ── helper ─────────────────────────────────────────────────────────────────
    def _gen(self, prompt: str, context: str, temperature: float):
        full = f"{context}\n\n{prompt}" if context else prompt
        return self.call_llm(full, temperature=temperature)

    # ── Solve : q → a  (Algorithm 1, pi_solve) ──────────────────────────────────
    def solve(self, question: str) -> str:
        """Produce a candidate with a CoT-style base reasoner.

        The same template serves the initial solve and every follow-up solve
        inside the falsifier. It has no slot for a candidate answer, which is what
        enforces Remark 1 structurally rather than by convention.
        """
        r = self._gen(
            "Solve the following problem. Reason step by step and do not skip "
            "steps.\n\n"
            f"Problem:\n{question}\n\n"
            "When you are finished, write the final answer on its own line, "
            "exactly as:\nANSWER: <your answer>",
            context=self._solve_ctx,
            temperature=self.solve_temperature)
        return r.content.strip()

    # ── Falsify : (q, a) → w | ⊥  (Algorithm 2) ─────────────────────────────────
    def falsify(self, question: str, candidate: str,
                round_index: int = 0) -> Optional[Witness]:
        """Attempt to construct a witness that ``candidate`` is wrong.

        Dispatches on the fixed per-benchmark ``HasChecker(q)`` predicate. The
        regime does not change across rounds or instances.
        """
        answer = _extract_answer(candidate)
        if self._has_checker:
            return self._falsify_executable(question, answer)
        return self._falsify_metamorphic(question, answer, round_index)

    # ── Executable regime (pi_probe + trusted c_q) ──────────────────────────────
    def _falsify_executable(self, question: str, answer: str) -> Optional[Witness]:
        """The model proposes ONE probe (never a verdict); the checker decides.

        A failure is a sound witness; a pass (or an undecidable check) is a sound
        survival — c_q never reports a false failure.
        """
        r = self._gen(
            "A candidate answer to the problem below has been produced. Your job "
            "is NOT to judge whether it is correct. Instead, propose ONE concrete "
            "probe that is most likely to expose an error in it -- for example a "
            "specific test input, an edge case, or a single sub-computation to "
            "recheck. An external verifier will run it.\n\n"
            f"Problem:\n{question}\n\n"
            f"Candidate answer:\n{answer}\n\n"
            "Output exactly one line, with a probe an external checker can "
            "execute:\nPROBE: <a single concrete input or check>",
            context=self._task_ctx,
            temperature=self.falsify_temperature)

        probe = _parse_probe(r.content)
        result: CheckResult = self._checker(  # type: ignore[misc]
            question, answer, probe, timeout=self.code_timeout)
        if result.verdict != "fail":
            return None  # "pass" or "undecided": the candidate survives.
        return Witness(
            text=result.detail,
            sound=True,
            regime="executable",
            orbit_block=(f"Probe run by an external checker: {probe or '(none)'}\n"
                         f"Result: {result.detail}"),
            relations_block="the answer must pass this check",
            violations=["checker"],
            satisfied=0,
        )

    # ── Metamorphic regime (catalogue C + independent variant solves) ───────────
    def _select_relations(self, question: str, candidate: str,
                          offset: int) -> List[Tuple[Relation, Optional[Variant]]]:
        """Sample(C, n): draw up to n *applicable* relations, deterministically.

        Programmatic transformations are applied here (they cost no model call);
        a relation that does not apply to this query is skipped, which is not a
        violation. Rounds start at different offsets so successive rounds attack
        the candidate from different directions.
        """
        selected: List[Tuple[Relation, Optional[Variant]]] = []
        size = len(self._catalogue)
        if size == 0:
            return selected
        for i in range(size):
            rel = self._catalogue[(offset + i) % size]
            if rel.programmatic:
                variant = rel.apply(question, candidate)  # type: ignore[misc]
                if variant is None:
                    continue
                selected.append((rel, variant))
            else:
                if rel.applicable is not None and not rel.applicable(question):
                    continue
                selected.append((rel, None))
            if len(selected) >= self.probes:
                break
        return selected

    def build_variant(self, relation: Relation, question: str) -> Optional[Variant]:
        """pi_mr: have the model apply a transformation it cannot get done in code.

        Used only for paraphrase-style relations. The prompt forbids solving
        either problem and never mentions the candidate, and the parsed
        ``RELATION:`` line is discarded — rho comes from the audited catalogue.
        """
        r = self._gen(
            "Rewrite the problem below into an EQUIVALENT variant by applying "
            "exactly this transformation, and nothing else:\n\n"
            f"  {relation.transformation}\n\n"
            "Constraints:\n"
            "  (a) Apply only the stated transformation. Do not simplify, clarify, "
            "correct, or add information, and do not remove anything the "
            "transformation does not remove.\n"
            "  (b) The variant must be fully self-contained and answerable on its own.\n"
            "  (c) Do NOT solve either the original problem or the variant.\n\n"
            f"Problem:\n{question}\n\n"
            "Output exactly these two lines:\n"
            "VARIANT: <the transformed problem, written out in full>\n"
            "RELATION: <how the variant's answer must relate to the original's answer>",
            context=self._task_ctx,
            temperature=self.falsify_temperature)
        variant_text = _parse_variant(r.content)
        if not variant_text:
            return None
        return equality_variant(relation, variant_text, source="model")

    def _probe(self, relation: Relation, variant: Optional[Variant],
               question: str) -> Optional[Tuple[Variant, str]]:
        """Build one variant if needed, then solve it independently."""
        if variant is None:
            variant = self.build_variant(relation, question)
            if variant is None:
                return None
        answer = _extract_answer(self.solve(variant.question))
        return (variant, answer)

    def _falsify_metamorphic(self, question: str, answer: str,
                             round_index: int) -> Optional[Witness]:
        """Refute the candidate with its own answers to transformed queries.

        Draws n relations, solves each variant independently, and returns a
        witness only if at least τ relations are violated *and* the candidate is
        outside the majority of the orbit (Remark 3).
        """
        selected = self._select_relations(question, answer, round_index * self.probes)
        if not selected:
            return None  # no applicable relation: FoT degenerates to Solve

        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = [executor.submit(self._probe, rel, var, question)
                       for rel, var in selected]
            probed = [f.result() for f in futures]

        orbit: List[str] = [answer]                       # O ← {a}, in q's frame
        violations: List[Tuple[Variant, str]] = []        # V
        lines: List[str] = []
        checked = 0
        for item in probed:
            if item is None:
                continue
            variant, a_prime = item
            checked += 1
            if variant.pullback is not None:
                pulled = variant.pullback(a_prime)
                if pulled is not None:
                    orbit.append(pulled)                  # O ← O ∪ {g^-1(a')}
            violated = not variant.holds(answer, a_prime)
            if violated:
                violations.append((variant, a_prime))
            lines.append(
                f"[{variant.relation}] {'DISAGREES' if violated else 'agrees'}\n"
                f"Formulation:\n{variant.question}\n"
                f"Answer given to it: {a_prime}")

        if len(violations) < max(1, self.tau):
            return None                                   # survives this round
        if self.use_majority and self._in_majority(answer, orbit):
            return None                                   # the orbit backs the candidate

        names = [v.relation for v, _ in violations]
        summary = "; ".join(
            f"under '{v.relation_text}', the variant answered {a_prime!r} "
            f"while the candidate answers {answer!r}"
            for v, a_prime in violations)
        return Witness(
            text=f"metamorphic disagreement ({', '.join(names)}): {summary}",
            sound=False,
            regime="metamorphic",
            orbit_block="\n\n".join(lines),
            relations_block="\n".join(
                f"- [{v.relation}] {v.relation_text}" for v, _ in violations),
            violations=names,
            satisfied=checked - len(violations),
        )

    @staticmethod
    def _in_majority(answer: str, orbit: List[str]) -> bool:
        """Majority(O): is the candidate among the most frequent answers in the orbit?"""
        counts: Dict[str, int] = {}
        for a in orbit:
            counts[normalize_answer(a)] = counts.get(normalize_answer(a), 0) + 1
        top = max(counts.values())
        return counts.get(normalize_answer(answer), 0) >= top

    # ── Repair : (q, a, w, H) → a'  (Algorithm 3, pi_rep) ───────────────────────
    def repair(self, question: str, candidate: str, witness: Witness,
               history: List[str]) -> str:
        """Produce a revised candidate that specifically resolves ``witness``.

        In the metamorphic regime the witness carries the whole orbit, so repair
        is constraint satisfaction over several equivalent formulations rather
        than "your answer is wrong, try again": the prompt presents a
        contradiction, never a judgement, which keeps the model working on the
        problem instead of on its own credibility. The history H is carried
        forward so repairs do not reintroduce previously resolved failures.
        """
        hist = "(none)"
        if self.carry_witness_history and history:
            hist = "\n".join(f"- {w}" for w in history)

        if witness.regime == "executable":
            lead = ("An external checker ran the probe below against the previous "
                    "answer and it failed. Produce a corrected answer that "
                    "specifically resolves that failure. Re-derive the problem "
                    "independently and do not assume the previous answer is "
                    "correct. Make sure your new answer does not reintroduce any "
                    "of the past failures listed.")
            orbit_label = "Probe and checker result:"
            relations_label = "What must hold:"
        else:
            lead = ("Below are several formulations of the same problem, together "
                    "with the answer that was previously given to each one. These "
                    "answers are mutually inconsistent, so at least one of them is "
                    "wrong.\n\nProduce a single answer that is consistent with "
                    "EVERY formulation. Re-derive each formulation independently "
                    "and do not assume that any previous answer is correct. Make "
                    "sure your new answer does not reintroduce any of the past "
                    "failures listed.")
            orbit_label = "Equivalent formulations and the answers they received:"
            relations_label = "Relations that must hold between those answers:"

        r = self._gen(
            f"{lead}\n\n"
            f"Original problem:\n{question}\n\n"
            f"Previous answer to the original problem:\n{_extract_answer(candidate)}\n\n"
            f"{orbit_label}\n{witness.orbit_block}\n\n"
            f"{relations_label}\n{witness.relations_block}\n\n"
            f"Past failures to avoid:\n{hist}\n\n"
            "Reason step by step. When finished, write the final answer on its own "
            "line, exactly as:\nANSWER: <your answer>",
            context=self._solve_ctx,
            temperature=self.repair_temperature)
        return r.content.strip()

    # ── pi_mr-gen: model-proposed catalogue (ablation only) ─────────────────────
    def _generate_catalogue(self, question: str) -> None:
        """Build C with the model instead of by hand (catalogue ablation).

        Generated once per run from one example problem, since C is a property of
        the task rather than of the query. Only proposals whose stated relation is
        answer-preserving are kept — rho must be evaluable by the driver.
        """
        r = self._gen(
            "Propose transformations that change the surface form of a problem of "
            "this kind while leaving its answer either unchanged, or changed in a "
            "way you can state exactly. Do not propose anything whose effect on "
            "the answer you cannot state. Do not solve any problem.\n\n"
            "Each proposal must be:\n"
            "  (a) applicable to any problem of this kind, not just the example below;\n"
            "  (b) mechanically applicable, with no judgement needed to carry it out;\n"
            "  (c) paired with an exact statement of how the answer must change.\n\n"
            f"Example problem of this kind:\n{question}\n\n"
            "Output a numbered list, one transformation per line, in exactly this "
            "form:\n"
            "MR1: TRANSFORM: <what to change> | RELATION: <how the answer must change>\n"
            "MR2: TRANSFORM: <what to change> | RELATION: <how the answer must change>\n"
            "...",
            context=self._task_ctx,
            temperature=self.falsify_temperature)
        generated = parse_generated_catalogue(r.content)
        if generated:
            self._catalogue = generated
        self._catalogue_generated = True

    # ── Driver loop (Algorithm 4) ───────────────────────────────────────────────
    def run(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> BaselineResponse:
        self.reset_counters()
        # Persona scoping (see __init__): Solve/Repair get the full framing; the
        # falsifier's own calls get only the one-line task description.
        self._solve_ctx = "\n\n".join(p for p in (system_prompt, instruction) if p)
        self._task_ctx = (instruction or system_prompt or "").strip()

        if (self.generate_relations and not self._has_checker
                and not self._catalogue_generated):
            self._generate_catalogue(question)

        candidate = self.solve(question)                  # a ← Solve(q)
        trace: List[str] = [f"[Solve] {_extract_answer(candidate)!r}"]
        witness_history: List[str] = []                   # H ← ∅
        # S: (candidate, corroboration score). "Rounds survived" is graded by how
        # many probes of the round the candidate withstood, so a candidate refuted
        # on 1 of 3 relations outranks one refuted on 3 of 3; ties resolve to the
        # earliest entry, which makes the initial Solve answer the floor. A
        # candidate refuted by a SOUND witness is dropped from S outright: it is
        # provably wrong and must never be returned.
        archive: List[Tuple[str, int]] = [(candidate, 0)]

        accepted = False
        regimes: List[str] = []
        iterations = 0
        for k in range(self.budget):
            witness = self.falsify(question, candidate, round_index=k)
            if witness is None:
                # Survived falsification → accept the fixpoint.
                accepted = True
                trace.append(f"[Falsify k={k + 1}] survived → accept")
                break
            iterations += 1
            regimes.append(witness.regime)
            trace.append(
                f"[Falsify k={k + 1}] witness ({witness.regime}, "
                f"sound={witness.sound}, violations={witness.violations}): "
                f"{witness.text}")
            if witness.sound:
                archive.pop()                             # provably wrong
            else:
                archive[-1] = (archive[-1][0], witness.satisfied)
            witness_history.append(witness.text)          # H ← H ∪ {w}
            candidate = self.repair(question, candidate, witness, witness_history[:-1])
            archive.append((candidate, 0))                # S ← S ∪ {(a, 0)}
            trace.append(f"[Repair k={k + 1}] → {_extract_answer(candidate)!r}")

        returned = "fixpoint" if accepted else "last"
        if not accepted and archive:
            # Budget exhausted: return the best-surviving candidate, not the last
            # step of an unguided walk.
            best = max(range(len(archive)), key=lambda i: archive[i][1])
            if archive[best][0] != candidate:
                candidate = archive[best][0]
                returned = "archive"
                trace.append(f"[Archive] best-surviving → {_extract_answer(candidate)!r}")

        final_answer = _extract_answer(candidate)
        return self.create_response(
            final_answer=final_answer,
            reasoning_trace=candidate,
            intermediate_steps=trace,
            metadata={
                "task": self.task,
                "subtask": self.subtask,
                "regime": "executable" if self._has_checker else "metamorphic",
                "has_checker": self._has_checker,
                "catalogue": [r.name for r in self._catalogue],
                "accepted_fixpoint": accepted,
                "returned": returned,
                "budget": self.budget,
                "probes": self.probes,
                "tau": self.tau,
                "use_majority": self.use_majority,
                "repairs_used": iterations,
                "budget_exhausted": not accepted,
                "execute_code": self.execute_code,
                "falsification_regimes": regimes,
                "witness_history": witness_history,
            },
        )

    def __repr__(self) -> str:
        return (f"FoT(baseline_name='{self.baseline_name}', "
                f"llm={self.llm.__class__.__name__}, task={self.task!r}, "
                f"subtask={self.subtask!r}, budget={self.budget}, "
                f"probes={self.probes}, tau={self.tau}, "
                f"has_checker={self._has_checker}, "
                f"catalogue={[r.name for r in self._catalogue]})")
