"""
Falsification-of-Thought (FoT): reasoning by metamorphic self-refutation.

FoT makes *self-refutation* the engine of reasoning. Instead of generating and
selecting, it generates a candidate and then tries to *break* it, using a
successful break to drive a targeted repair. The guiding principle is:

    Refute by construction, not by verdict.

The model is refuted by its own outputs, never by its own self-assessment. No
prompt in FoT asks "is this correct?": the model only ever produces *objects* —
an answer, a probe, a variant of the problem, an answer to that variant — and
every accept/refute decision is issued by an executable checker c_q or by the
driver's own comparison of two model-produced answers.

The driver loop (Algorithm 4 of the paper):

    a ← Solve(q); H ← ∅; S ← ∅            # candidate; witness history; archive
    for k in 1..K:                        # budget K
        (w, s) ← Falsify(q, a)            # Alg. 2; s is the damage score of a
        S ← S ∪ {(a, s, k)}               # archive a with the evidence gathered
        if w = ⊥: return a                # survived falsification: fixpoint
        H ← H ∪ {w}
        a ← Repair(q, a, w, H)            # witness-guided repair
    return min_≺ S                        # budget exhausted: best measured (Def. 2)

FoT composes three operators, each a prompt to a single frozen model M:

  * Solve  : q → a            produce a candidate (any base reasoner; CoT here).
  * Falsify: (q, a) → w | ⊥   construct a witness that a is wrong, or ⊥.
  * Repair : (q, a, w, H) → a' revise a to resolve w.

The falsifier (Algorithm 2) operates in one of two regimes, selected by the
task-level predicate ``HasChecker(q)`` — FIXED PER BENCHMARK, not decided at run
time by the model (see :mod:`baseline.FoT.checkers`):

  * **Executable regime** — when the benchmark exposes a cheap, decisive checker
    c_q (arithmetic evaluation for Game of 24 and BBH multi-step arithmetic,
    program execution for CRUXEval, ``sat`` for Python Puzzles). The model only
    *proposes a probe* (pi_probe) — it is forbidden to give a verdict; the trusted
    external c_q decides. A failure is therefore a SOUND witness.

  * **Metamorphic regime** — when no checker exists. The query is transformed by
    a fixed, human-audited catalogue C of semantics-preserving relations
    (:mod:`baseline.FoT.relations`), each variant is solved *independently*, and
    the witness is a concrete disagreement inside the resulting orbit
    O = {a} ∪ {g⁻¹(a'_i)}. By Proposition 1 a violated valid relation certifies
    that at least one answer in the orbit is wrong, with no appeal to the model's
    opinion — which moves the residual unsoundness from an unauditable,
    per-instance hallucination to an auditable, design-time constant.

Three design decisions carry the weight of the metamorphic branch:

  * *Independence* (Remark 1) — the candidate never appears in the prompt that
    solves a variant, so the follow-up answer is evidence rather than an echo.
    This is structural: pi_solve simply has no candidate slot.
  * *Directionality* (Remark 2) — every relation in C must be non-decreasing in
    reliability, so a disagreement is more likely to indict a than a'. Distractor
    insertion is excluded on this ground.
  * *Corroboration* (Remark 3) — repair fires only when at least ``tau``
    relations are violated *and* a is outside the majority of the orbit. Setting
    ``tau = 1`` and dropping the majority test recovers the pilot's behaviour,
    which is available as an ablation.

This replaces the pilot's assertion-based regime (pi_nec + pi_chk and the
``ReCheckable`` guard), in which the model asserted that the candidate violated a
self-generated necessary condition — the source of the MGSM regression reported
in the paper. There is no judgement template anywhere in FoT: rho is evaluated by
the driver.

Cost: O(K·(1 + n)) model calls — one Solve per variant, plus at most one call per
variant to construct it. Far below the recursive expansion of ToT/GoT. The
executable regime costs O(K·2): one probe settles the verdict.

Reference: "Falsification-of-Thought: Reasoning by Metamorphic Self-Refutation".
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from baseline.basebaseline import BaseBaseline, BaselineResponse
from baseline.FoT.checkers import CheckResult, Checker, get_checker
from baseline.FoT.relations import (
    Relation,
    Variant,
    answer_key,
    equality_variant,
    get_catalogue,
    numeric_literals,
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


def _majority_keys(orbit: List[str]) -> List[str]:
    """Majority(O): the canonical keys of the most frequent orbit members."""
    counts: Dict[str, int] = {}
    for value in orbit:
        key = answer_key(value)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return []
    top = max(counts.values())
    return [k for k, c in counts.items() if c == top]


# ── Witness and damage score ────────────────────────────────────────────────────

@dataclass
class OrbitEntry:
    """One variant of the query, the answer it received, and whether rho held."""

    variant: Variant
    answer: str
    pulled: Optional[str]
    violated: bool


@dataclass
class Damage:
    """s(a) = (v, c): relations violated, and orbit members that agree with a.

    Definition 2. In the executable regime ``violations`` ∈ {0, 1} records the
    checker's verdict and ``support`` its complement, so one archive order covers
    both regimes.
    """

    violations: int
    support: int

    def order_key(self, round_index: int) -> Tuple[int, int, int]:
        """≺ : fewest violations, then widest orbit support, then earliest."""
        return (self.violations, -self.support, round_index)


@dataclass
class Witness:
    """A constructive artifact demonstrating that a candidate is wrong.

    In the metamorphic regime w = ⟨V, O⟩: the violated relations together with the
    whole orbit — several equivalent formulations of one problem and their
    mutually inconsistent answers. In the executable regime it is the failing
    probe and the checker's detail d. Both fill the same two slots of pi_rep.

    Attributes:
        regime: "executable" or "metamorphic".
        sound: True iff the trusted external checker c_q reported the failure, in
            which case the witness certifies that the candidate is incorrect. A
            metamorphic witness is sound only *relative to the relation*
            (Proposition 1): it certifies that some answer in the orbit is wrong.
        orbit_text: pi_rep's ``<orbit>`` slot.
        relations_text: pi_rep's ``<relations>`` slot.
        violated: names of the relations that were violated.
        orbit: the pulled-back answers, in q's own coordinate frame.
        summary: one-line rendering, used for the trace and the history H.
    """

    regime: str
    sound: bool
    orbit_text: str
    relations_text: str
    summary: str
    violated: List[str] = field(default_factory=list)
    orbit: List[str] = field(default_factory=list)

    def record(self) -> Dict[str, Any]:
        """Machine-readable summary for the run metadata (witness precision)."""
        return {
            "regime": self.regime,
            "sound": self.sound,
            "violated": list(self.violated),
            "orbit": list(self.orbit),
            "summary": self.summary,
        }


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
        solve_temperature: float = 0.0,
        falsify_temperature: float = 0.7,
        repair_temperature: float = 0.0,
        execute_code: bool = True,
        code_timeout: float = 10.0,
        carry_witness_history: bool = True,
        orbit_majority: bool = True,
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
            probes: n — relations drawn from C per falsification attempt, i.e. the
                size of the orbit. Ignored in the executable regime, where one
                probe settles a deterministic checker's verdict.
            tau: refutation threshold τ — relations that must be violated before a
                repair fires (Remark 3). ``tau = 1`` with ``orbit_majority=False``
                recovers the pilot's single-violation trigger.
            solve_temperature: Sampling temperature for Solve — the initial one
                and every follow-up solve of a variant. Independence here comes
                from perturbing the *input*, not from resampling, so 0 is fine.
            falsify_temperature: Sampling temperature for the falsifier's own
                construction calls (pi_probe, pi_mr).
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
            orbit_majority: If True, refuse to repair a candidate that holds the
                majority of its own orbit (Remark 3). Turn off for the pilot
                ablation.
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
        self.solve_temperature = solve_temperature
        self.falsify_temperature = falsify_temperature
        self.repair_temperature = repair_temperature
        self.execute_code = execute_code
        self.code_timeout = code_timeout
        self.carry_witness_history = carry_witness_history
        self.orbit_majority = orbit_majority
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

        # Per-question caches: a variant depends on the query (and, for backward
        # substitution, on the candidate), so identical variants are built and
        # solved once even when several rounds draw them.
        self._variant_cache: Dict[Tuple[str, int], Optional[Variant]] = {}
        self._answer_cache: Dict[str, str] = {}

        # Context split (persona scoping): the answer-producing operators (Solve,
        # Repair) receive the full task framing + answer-format directives so their
        # output stays parseable; the falsifier's own calls receive only a one-line
        # task description, so the format persona never conflicts with "propose a
        # probe" / "rewrite this problem".
        self._solve_ctx = ""   # system_prompt + instruction
        self._task_ctx = ""    # one-line task description only

    # ── helpers ────────────────────────────────────────────────────────────────
    def _gen(self, prompt: str, context: str, temperature: float):
        full = f"{context}\n\n{prompt}" if context else prompt
        return self.call_llm(full, temperature=temperature)

    @staticmethod
    def _parallel(calls: List[Callable[[], str]]) -> List[str]:
        """Run independent completions concurrently (the orbit's Solve calls)."""
        if len(calls) <= 1:
            return [call() for call in calls]
        with ThreadPoolExecutor(max_workers=len(calls)) as executor:
            return [f.result() for f in [executor.submit(c) for c in calls]]

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

    def _solve_variant(self, question: str) -> str:
        """Solve a variant, reusing the answer if this variant was already solved."""
        if question not in self._answer_cache:
            self._answer_cache[question] = _extract_answer(self.solve(question))
        return self._answer_cache[question]

    # ── Falsify : (q, a) → (w | ⊥, s)  (Algorithm 2) ────────────────────────────
    def falsify(self, question: str, candidate: str,
                round_index: int = 0) -> Tuple[Optional[Witness], Damage]:
        """Attempt to construct a witness that ``candidate`` is wrong.

        Returns the witness (or None) together with the damage score s measured
        for the candidate during the attempt. Dispatches on the fixed
        per-benchmark ``HasChecker(q)`` predicate; the regime never changes across
        rounds.
        """
        answer = _extract_answer(candidate)
        if self._has_checker:
            return self._falsify_executable(question, answer)
        return self._falsify_metamorphic(question, answer, round_index)

    # ── Executable regime (pi_probe + trusted c_q) ──────────────────────────────
    def _falsify_executable(self, question: str,
                            answer: str) -> Tuple[Optional[Witness], Damage]:
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
            return None, Damage(0, 1)   # "pass" or "undecided": survives
        summary = f"an external check failed: {result.detail}"
        return Witness(
            regime="executable",
            sound=True,
            orbit_text=(f"An external verifier ran this probe against the answer "
                        f"{answer!r}: {probe or '(the checker chose its own check)'}\n"
                        f"It failed: {result.detail}"),
            relations_text="the answer must pass the check the verifier ran",
            summary=summary,
            violated=["checker"],
        ), Damage(1, 0)

    # ── Metamorphic regime (catalogue C + an orbit of independent solves) ───────
    def _sample(self, question: str, answer: str,
                round_index: int) -> List[Variant]:
        """Sample(C, n): the first ``n`` applicable relations, in catalogue order.

        A relation that does not fit this query returns None and is skipped, which
        is *not* a violation — so an unregistered benchmark degrades to
        FoT ≡ Solve rather than to a misapplied relation. Programmatic
        transformations are applied here and cost no model call; paraphrase-style
        ones go through pi_mr.
        """
        drawn: List[Variant] = []
        for relation in self._catalogue:
            if len(drawn) >= max(1, self.probes):
                break
            key = (relation.name, round_index if relation.name == "mask_quantity" else 0)
            if key not in self._variant_cache:
                self._variant_cache[key] = self._instantiate(
                    relation, question, answer, round_index)
            variant = self._variant_cache[key]
            if variant is not None:
                drawn.append(variant)
        return drawn

    def _instantiate(self, relation: Relation, question: str, answer: str,
                     round_index: int) -> Optional[Variant]:
        """Realise q' = T(q) for one relation, or None when T does not apply."""
        if relation.apply is not None:
            return relation.apply(question, answer, round_index)
        if relation.applicable is not None and not relation.applicable(question):
            return None
        return self.build_variant(relation, question)

    def build_variant(self, relation: Relation, question: str) -> Optional[Variant]:
        """pi_mr: have the model apply a transformation it cannot get done in code.

        Used only for paraphrase-style relations. The prompt forbids solving
        either problem and never mentions the candidate, and the parsed
        ``RELATION:`` line is discarded — rho comes from the audited catalogue.

        The result is then *mechanically validated*: a model-instantiated
        transform must carry over every numeric literal of the original, so a
        paraphrase that quietly drops or invents a quantity is rejected rather
        than checked against a relation it no longer satisfies.
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
        if (relation.preserve_numbers
                and numeric_literals(variant_text) != numeric_literals(question)):
            return None
        return equality_variant(relation, variant_text, source="model")

    def _falsify_metamorphic(self, question: str, answer: str,
                             round_index: int) -> Tuple[Optional[Witness], Damage]:
        """Build the orbit, collect the violations, and apply Remark 3's rule."""
        variants = self._sample(question, answer, round_index)
        if not variants:
            return None, Damage(0, 1)     # nothing applicable: survives

        # The variants are solved independently and concurrently; the candidate is
        # never shown to any of them (Remark 1).
        answers = self._parallel(
            [(lambda q=v.question: self._solve_variant(q)) for v in variants])

        entries: List[OrbitEntry] = []
        orbit: List[str] = [answer]       # O ← {a}
        for variant, a_prime in zip(variants, answers):
            pulled = variant.pullback(a_prime) if variant.pullback else None
            if pulled is not None:
                orbit.append(pulled)      # O ← O ∪ {g⁻¹(a')}
            entries.append(OrbitEntry(variant, a_prime, pulled,
                                      not variant.holds(answer, a_prime)))

        violated = [e for e in entries if e.violated]
        support = sum(1 for o in orbit if answer_key(o) == answer_key(answer))
        damage = Damage(len(violated), support)

        if len(violated) < max(1, self.tau):
            return None, damage           # not corroborated: survives this round
        if self.orbit_majority and len(orbit) > 1 and (
                answer_key(answer) in _majority_keys(orbit)):
            # The bulk of the orbit agrees with a: the defect is more likely to sit
            # at a variant's answer than at the candidate, so no repair fires.
            return None, damage
        return self._make_witness(answer, entries, violated, orbit), damage

    def _make_witness(self, answer: str, entries: List[OrbitEntry],
                      violated: List[OrbitEntry], orbit: List[str]) -> Witness:
        """Render w = ⟨V, O⟩ into pi_rep's ``<orbit>`` and ``<relations>`` slots."""
        orbit_lines: List[str] = []
        relation_lines: List[str] = []
        for i, entry in enumerate(entries, start=1):
            variant = entry.variant
            orbit_lines.append(
                f"[{i}] Formulation ({variant.relation}):\n{variant.question}\n"
                f"    Answer it received: {entry.answer!r}")
            expected = variant.expected_value(answer)
            requirement = (f" Given the previous answer {answer!r}, this "
                           f"formulation's answer must be {expected!r}."
                           if expected is not None else "")
            relation_lines.append(
                f"[{i}] {variant.relation}: {variant.relation_text}.{requirement}"
                f" Observed: {entry.answer!r}"
                f"{'  <-- INCONSISTENT' if entry.violated else ''}")

        names = ", ".join(sorted({e.variant.relation for e in violated}))
        details = "; ".join(
            f"{e.variant.relation} gave {e.answer!r} where "
            f"{e.variant.expected_value(answer)!r} was required"
            for e in violated)
        return Witness(
            regime="metamorphic",
            sound=False,
            orbit_text="\n\n".join(orbit_lines),
            relations_text="\n".join(relation_lines),
            summary=(f"the answer {answer!r} was inconsistent with "
                     f"{len(violated)} relation(s) [{names}]: {details}"),
            violated=[e.variant.relation for e in violated],
            orbit=list(orbit),
        )

    # ── Repair : (q, a, w, H) → a'  (Algorithm 3, pi_rep) ───────────────────────
    def repair(self, question: str, candidate: str, witness: Witness,
               history: List[str]) -> str:
        """Produce a revised candidate consistent with every formulation.

        The prompt presents a *contradiction* rather than a judgement: it never
        asserts that the previous answer was wrong, only that the set of answers
        cannot all be right, which keeps the repairer working on the problem
        rather than on its own credibility. Repair thereby becomes constraint
        satisfaction over the orbit — the setting in which models revise reliably,
        even though they locate failures poorly on their own. The history H is
        carried forward so repairs do not reintroduce previously resolved
        failures.
        """
        hist = "(none)"
        if self.carry_witness_history and history:
            hist = "\n".join(f"- {w}" for w in history)

        r = self._gen(
            "Below are several formulations of the same problem, together with the "
            "answer that was previously given to each one. These answers are "
            "mutually inconsistent, so at least one of them is wrong.\n\n"
            "Produce a single answer that is consistent with EVERY formulation. "
            "Re-derive each formulation independently and do not assume that any "
            "previous answer is correct. Make sure your new answer does not "
            "reintroduce any of the past failures listed.\n\n"
            f"Original problem:\n{question}\n\n"
            f"Previous answer to the original problem:\n{_extract_answer(candidate)}\n\n"
            f"Equivalent formulations and the answers they received:\n"
            f"{witness.orbit_text}\n\n"
            f"Relations that must hold between those answers:\n"
            f"{witness.relations_text}\n\n"
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
        self._variant_cache = {}
        self._answer_cache = {}

        if (self.generate_relations and not self._has_checker
                and not self._catalogue_generated):
            self._generate_catalogue(question)

        candidate = self.solve(question)                  # a ← Solve(q)
        initial_answer = _extract_answer(candidate)
        trace: List[str] = [f"[Solve] {initial_answer!r}"]
        history: List[str] = []                           # H ← ∅
        records: List[Dict[str, Any]] = []
        # S: the archive. An entry is written for a candidate at the moment it is
        # *measured*, so the candidate produced by the final Repair is never
        # archived — no evidence about it has been gathered (Def. 2).
        archive: List[Tuple[Tuple[int, int, int], str]] = []

        accepted = False
        for k in range(self.budget):                      # for k ← 1 to K
            witness, damage = self.falsify(question, candidate, k)
            archive.append((damage.order_key(k), candidate))
            trace.append(f"[Falsify k={k + 1}] violations={damage.violations}, "
                         f"orbit support={damage.support}")
            if witness is None:                           # w = ⊥
                accepted = True
                trace.append(f"[Falsify k={k + 1}] survived → accept fixpoint")
                break

            records.append(witness.record())
            trace.append(f"[Witness k={k + 1}] ({witness.regime}, "
                         f"sound={witness.sound}) {witness.summary}")
            history.append(witness.summary)               # H ← H ∪ {w}
            candidate = self.repair(question, candidate, witness, history[:-1])
            trace.append(f"[Repair k={k + 1}] → {_extract_answer(candidate)!r}")

        returned = "fixpoint"
        if not accepted and archive:
            # Budget exhausted: return min_≺ S — fewest violations, then widest
            # orbit support, then earliest. Ties break toward earlier entries, so
            # a_0 is displaced only by a strictly better-corroborated candidate.
            best = min(archive, key=lambda entry: entry[0])[1]
            returned = "archive"
            if best != candidate:
                candidate = best
                trace.append(f"[Archive] best measured → {_extract_answer(candidate)!r}")

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
                "probes": 1 if self._has_checker else self.probes,
                "tau": 1 if self._has_checker else self.tau,
                "repairs_used": len(records),
                "budget_exhausted": not accepted,
                "execute_code": self.execute_code,
                # Mechanism-level evaluation: the initial/final pair gives the flip
                # matrix, and the witness records give witness precision per regime
                # and per relation, once ground truth is joined.
                "initial_answer": initial_answer,
                "answer_changed": answer_key(initial_answer) != answer_key(final_answer),
                "witnesses": records,
                "witness_history": history,
            },
        )

    def __repr__(self) -> str:
        return (f"FoT(baseline_name='{self.baseline_name}', "
                f"llm={self.llm.__class__.__name__}, task={self.task!r}, "
                f"subtask={self.subtask!r}, budget={self.budget}, "
                f"probes={self.probes}, tau={self.tau}, "
                f"has_checker={self._has_checker}, "
                f"catalogue={[r.name for r in self._catalogue]})")
