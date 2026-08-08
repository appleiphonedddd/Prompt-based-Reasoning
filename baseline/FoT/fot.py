"""
Falsification-of-Thought (FoT): reasoning by self-refutation with relational
falsification.

FoT makes *self-refutation* the engine of reasoning. Instead of generating and
selecting, it generates a candidate and then repeatedly tries to *break* it,
using each successful break to drive a targeted repair. The design invariant that
runs through every operator is:

    The model constructs; a deterministic procedure decides.

No prompt in FoT asks "is this correct?". The model only ever produces
*objects* — an initial answer, a probe, a re-derived quantity, a solution to a
transformed problem — and every accept/refute decision is issued by an executable
checker c_q or by a deterministic comparator applied to model-produced values.

The driver loop (Algorithm 4 of the paper):

    a ← Solve(q); H ← ∅                     # initial candidate; witness history
    for k in 1..K:                          # budget K
        w ← ⊥
        for j in 1..m:                      # survival threshold m
            w ← Falsify(q, a)               # ONE probe / relation instance
            if w ≠ ⊥: break
        if w = ⊥:
            return a                        # survived m distinct checks: fixpoint
        H ← H ∪ {w}
        a ← Repair(q, a, w, H)              # witness-guided repair
    return a                                # budget exhausted: last candidate

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

  * **Relational regime** — when no checker exists. The query is attacked with a
    small, human-audited library R_q of relation schemas
    (:mod:`baseline.FoT.relations`) drawn from the paper's three families:
    backward substitution (rho_bwd — mask a stated quantity, re-derive it *from*
    the candidate through pi_mask), metamorphic transformation (rho_mr —
    transform the query, solve the follow-up independently with pi_solve, compare
    the pair against phi), and mechanical invariants (rho_inv — the comparator
    computes both sides itself and no model call is made at all). The witness is
    the mechanically exhibited contradiction ⟨rho, instance, (expected,
    observed)⟩.

Four design decisions carry the weight of the relational branch:

  * *Voted derivation* — the single model-produced value in a relational check is
    the majority over ``s`` independent completions; ties and an UNDETERMINED
    majority abstain. This converts "one bad sample can refute a correct answer"
    into "a false refutation requires a systematic majority error on a short,
    tightly constrained derivation".
  * *Independence* — for rho_mr the candidate never appears in the prompt that
    solves the follow-up, so its answer is evidence rather than an echo. This is
    structural: pi_solve simply has no candidate slot.
  * *Distinctness* — successive attempts cycle through distinct relations and
    distinct masked slots (Next, NextSlot), so surviving m attempts means
    surviving m *different* mechanical checks, not m re-rolls of one check.
  * *Attribution* — a metamorphic witness implicates the pair (a, a_f), so pi_rep
    explicitly licenses the model to defend its answer and blame the follow-up.

This replaces the pilot's assertion-based semantic regime (pi_nec + pi_chk and the
``ReCheckable`` guard), in which the model asserted that the candidate violated a
self-generated necessary condition — sound-looking but, being a verdict on its own
work, the source of the MGSM regression reported in the paper. There is no
judgement template anywhere in FoT: phi is evaluated by the driver.

Cost: O(K·(1 + m·s)) model calls, with small constants (K=3, m=3, s=5 in the
paper's experiments) — every call a short completion, and far below the recursive
expansion of ToT/GoT. The executable regime costs O(K·2): one probe settles the
verdict, because every checker is deterministic and probe-independent.

Reference: "Falsification-of-Thought: Reasoning by Self-Refutation with
Relational Falsification".
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from baseline.basebaseline import BaseBaseline, BaselineResponse
from baseline.FoT.checkers import CheckResult, Checker, get_checker
from baseline.FoT.relations import (
    Relation,
    Variant,
    enumerate_slots,
    equality_variant,
    get_catalogue,
    normalize_answer,
    numeric_literals,
    parse_generated_catalogue,
)
from models.base import BaseLLM


# The abstention token of pi_mask: the slot is not identifiable from the
# candidate, so the check must not be turned into a refutation.
UNDETERMINED = "UNDETERMINED"

# How many passes over R_q Next() may make before declaring it exhausted. One
# pass per attempt is not enough because rho_bwd yields a *new* instance (a new
# masked slot) every time it is drawn.
_MAX_LIBRARY_PASSES = 4


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


def _parse_derived(text: str) -> str:
    """Parse the ``DERIVED:`` line (pi_mask output).

    Anything unparseable is folded into UNDETERMINED rather than guessed at: a
    completion the driver cannot read must abstain, never refute.
    """
    matches = list(re.finditer(r"DERIVED:\s*(.+)", text, re.IGNORECASE))
    if not matches:
        return UNDETERMINED
    value = matches[-1].group(1).strip().rstrip(".")
    if not value or value.upper().startswith(UNDETERMINED):
        return UNDETERMINED
    return value


def _parse_variant(text: str) -> str:
    """Parse the ``VARIANT:`` block (pi_mr output); ``RELATION:`` ends it.

    The relation line is ignored on purpose: phi comes from the library, which
    was audited at design time, never from what the model says about it here.
    """
    m = re.search(r"VARIANT:\s*(.*?)(?=^\s*RELATION:|\Z)", text,
                  re.DOTALL | re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def _majority(values: List[str]) -> Optional[str]:
    """Majority(v_1..v_s): the plurality value, or None if the vote abstains.

    Abstains on a tie and whenever UNDETERMINED wins, which is what makes
    under-determination unable to masquerade as a refutation.
    """
    counts: Dict[str, int] = {}
    representative: Dict[str, str] = {}
    for value in values:
        key = UNDETERMINED if value == UNDETERMINED else normalize_answer(value)
        counts[key] = counts.get(key, 0) + 1
        representative.setdefault(key, value)
    if not counts:
        return None
    top = max(counts.values())
    winners = [k for k, c in counts.items() if c == top]
    if len(winners) != 1 or winners[0] == UNDETERMINED:
        return None
    return representative[winners[0]]


@dataclass
class Witness:
    """A constructive artifact demonstrating that a candidate is wrong.

    The witness is w = ⟨rho, instance, (expected, observed)⟩ — every component is
    either quoted from q, produced by the driver, or a voted model value, and the
    verdict inside it was issued by Compare, not by the model.

    Attributes:
        text: the self-contained rendering that fills pi_rep's <witness> slot and
            the witness history H.
        sound: True iff produced by the executable regime, where the trusted
            external checker c_q reported a real failure — in which case it
            certifies the candidate is incorrect. A relational witness sits lower
            on the soundness ladder (§3.6): it is sound conditional on the
            relation's validity and, for rho_mr, implicates the *pair*.
        regime: "executable" or "relational".
        family: "checker" | "bwd" | "mr" | "inv" — the schema family, reported so
            that witness precision can be broken down per family.
        relation: name of the relation (or "checker") that produced it.
        expected: what the check required.
        observed: what was actually produced.
        instance: the check instance — the masked or transformed problem, or the
            probe the checker ran. Empty for rho_inv, which has no instance.
        votes: how many independent completions the observed value is a majority
            over (0 when no model call was involved).
        survived: how many distinct checks this candidate withstood before this
            one refuted it.
    """

    text: str
    sound: bool
    regime: str
    family: str
    relation: str
    expected: str = ""
    observed: str = ""
    instance: str = ""
    votes: int = 0
    survived: int = 0

    def record(self) -> Dict[str, Any]:
        """Machine-readable summary for the run metadata (witness precision)."""
        return {
            "regime": self.regime,
            "family": self.family,
            "relation": self.relation,
            "expected": self.expected,
            "observed": self.observed,
            "votes": self.votes,
            "survived": self.survived,
            "sound": self.sound,
        }


class FoT(BaseBaseline):
    """Falsification-of-Thought prompting baseline."""

    def __init__(
        self,
        llm: BaseLLM,
        task: Optional[str] = None,
        subtask: Optional[str] = None,
        budget: int = 3,
        survival: int = 3,
        votes: int = 5,
        solve_temperature: float = 0.0,
        vote_temperature: float = 0.7,
        falsify_temperature: float = 0.7,
        repair_temperature: float = 0.0,
        execute_code: bool = True,
        code_timeout: float = 10.0,
        carry_witness_history: bool = True,
        relations: Optional[List[str]] = None,
        generate_relations: bool = False,
        archive: bool = False,
    ) -> None:
        """Initialize the FoT baseline.

        Args:
            llm: An instance of a BaseLLM subclass.
            task: Benchmark name (e.g. "gameof24", "mgsm"). Selects the trusted
                checker c_q and the relation library R_q, fixing ``HasChecker(q)``
                for the whole run.
            subtask: Sub-benchmark, e.g. the BigBenchHard task name. Checker and
                library lookup are most-specific-first.
            budget: K — maximum number of Falsify→Repair iterations.
            survival: m — falsification attempts a candidate must survive before
                it is accepted as a fixpoint. Each attempt draws a *distinct*
                relation instance. Forced to 1 in the executable regime, where a
                single probe already settles the verdict.
            votes: s — independent completions the voted derivation takes the
                majority over. Relational regime only.
            solve_temperature: Sampling temperature for the initial Solve.
            vote_temperature: Sampling temperature for the ``s`` completions of a
                relational check (pi_mask, and the follow-up pi_solve). Must be
                non-zero for the vote to carry information — at 0 a deterministic
                backend returns ``s`` copies of one completion.
            falsify_temperature: Sampling temperature for the falsifier's own
                construction calls (pi_probe, pi_mr).
            repair_temperature: Sampling temperature for witness-guided Repair.
            execute_code: If True, use the executable regime on benchmarks that
                have a trusted checker c_q. If False, force the relational
                regime everywhere. The checker runs the benchmark's own reference
                code (CRUXEval functions, Python-Puzzles ``sat``) in a
                timeout-bounded subprocess; disable on an untrusted host.
            code_timeout: Per-execution timeout in seconds for the checker.
            carry_witness_history: If True, carry the history H of past witnesses
                into Repair so fixes do not reintroduce previously resolved
                failures (discouraging oscillation).
            relations: Restrict R_q to these relation names (see
                :mod:`baseline.FoT.relations`); None uses the whole library.
            generate_relations: Ablation — build R_q with pi_mr-gen from the first
                question of the run instead of using the hand-written library.
            archive: Deviation from Algorithm 4, off by default. When True, budget
                exhaustion returns the *best-surviving* candidate seen (dropping
                any that a sound witness proved wrong) instead of the last step of
                the walk.
        """
        super().__init__(llm, baseline_name="FoT")
        self.task = task
        self.subtask = subtask
        self.budget = budget
        self.survival = survival
        self.votes = votes
        self.solve_temperature = solve_temperature
        self.vote_temperature = vote_temperature
        self.falsify_temperature = falsify_temperature
        self.repair_temperature = repair_temperature
        self.execute_code = execute_code
        self.code_timeout = code_timeout
        self.carry_witness_history = carry_witness_history
        self.generate_relations = generate_relations
        self.archive = archive

        # HasChecker(q): fixed per benchmark, not decided at run time by the model.
        self._checker: Optional[Checker] = (
            get_checker(task, subtask) if execute_code else None)
        self._has_checker = self._checker is not None

        # R_q: also fixed per benchmark. Empty in the executable regime, where the
        # checker decides and no relation is needed.
        self._library: List[Relation] = (
            [] if self._has_checker else get_catalogue(task, subtask, relations))
        self._catalogue_generated = False

        # Falsifier state (Next / NextSlot), reset per question.
        self._rel_cursor = 0
        self._slot_cursor = 0
        self._seen: Set[Any] = set()

        # Context split (persona scoping): the answer-producing operators (Solve,
        # Repair) receive the full task framing + answer-format directives so their
        # output stays parseable; the falsifier's own calls receive only a one-line
        # task description, so the format persona never conflicts with "propose a
        # probe" / "re-derive X" / "rewrite this problem".
        self._solve_ctx = ""   # system_prompt + instruction
        self._task_ctx = ""    # one-line task description only

    # ── helpers ────────────────────────────────────────────────────────────────
    def _gen(self, prompt: str, context: str, temperature: float):
        full = f"{context}\n\n{prompt}" if context else prompt
        return self.call_llm(full, temperature=temperature)

    def _parallel(self, call: Callable[[], str], n: int) -> List[str]:
        """Run ``n`` independent completions of the same instance concurrently."""
        if n <= 1:
            return [call()]
        with ThreadPoolExecutor(max_workers=n) as executor:
            return [f.result() for f in [executor.submit(call) for _ in range(n)]]

    def _attempts_per_round(self) -> int:
        """m for this regime.

        With a decisive checker a single probe already settles the verdict: every
        c_q in :mod:`baseline.FoT.checkers` is deterministic and computes its
        canonical verdict independently of the probe, so re-probing the same
        candidate could only repeat the same answer at the same cost.
        """
        return 1 if self._has_checker else max(1, self.survival)

    def _reset_falsifier(self) -> None:
        """Rewind Next / NextSlot at the start of a question."""
        self._rel_cursor = 0
        self._slot_cursor = 0
        self._seen = set()

    # ── Solve : q → a  (Algorithm 1, pi_solve) ──────────────────────────────────
    def solve(self, question: str, temperature: Optional[float] = None) -> str:
        """Produce a candidate with a CoT-style base reasoner.

        The same template serves the initial solve and every follow-up solve
        inside the falsifier. It has no slot for a candidate answer, which is what
        enforces the independence of rho_mr structurally rather than by convention.
        """
        r = self._gen(
            "Solve the following problem. Reason step by step and do not skip "
            "steps.\n\n"
            f"Problem:\n{question}\n\n"
            "When you are finished, write the final answer on its own line, "
            "exactly as:\nANSWER: <your answer>",
            context=self._solve_ctx,
            temperature=(self.solve_temperature if temperature is None else temperature))
        return r.content.strip()

    # ── Falsify : (q, a) → w | ⊥  (Algorithm 2) ─────────────────────────────────
    def falsify(self, question: str, candidate: str) -> Optional[Witness]:
        """Make ONE attempt to construct a witness that ``candidate`` is wrong.

        Dispatches on the fixed per-benchmark ``HasChecker(q)`` predicate. The
        regime does not change across rounds or instances.
        """
        answer = _extract_answer(candidate)
        if self._has_checker:
            return self._falsify_executable(question, answer)
        return self._falsify_relational(question, answer)

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
            text=(f"An external checker ran this probe against the answer "
                  f"{answer!r}: {probe or '(the checker chose its own check)'}\n"
                  f"It failed: {result.detail}"),
            sound=True,
            regime="executable",
            family="checker",
            relation="checker",
            expected="the check to pass",
            observed=result.detail,
            instance=probe,
        )

    # ── Relational regime (library R_q + voted derivations) ─────────────────────
    def _instantiate(self, relation: Relation, question: str,
                     answer: str) -> Optional[Variant]:
        """Realise one programmatic relation as a check instance, or skip it.

        A relation that does not fit this query returns None, which is *not* a
        violation — it is simply skipped, so an unregistered benchmark degrades
        to FoT ≡ Solve rather than to a misapplied relation.
        """
        if relation.family == "inv":
            if relation.invariant is None:
                return None
            computed = relation.invariant(question, answer)
            if computed is None:
                return None
            expected, observed, detail = computed
            return Variant(
                relation=relation.name,
                relation_text=relation.relation_text,
                question="",
                holds=lambda a, value, _e=expected: value == _e,
                family="inv",
                expected=expected,
                observed=observed,
                detail=detail,
                source="programmatic",
            )
        if relation.family == "bwd":
            if relation.apply_slot is None or not enumerate_slots(question):
                return None
            # NextSlot(q): each landing on rho_bwd masks the *next* stated
            # quantity, so successive backward checks are genuinely distinct.
            variant = relation.apply_slot(question, answer, self._slot_cursor)
            if variant is not None:
                self._slot_cursor += 1
            return variant
        if relation.apply is not None:
            return relation.apply(question, answer)
        return None

    def _next_instance(self, question: str, answer: str) -> Optional[Variant]:
        """Next(R_q): the next *distinct*, applicable check instance, or None.

        Programmatic transformations are applied here and cost no model call.
        Instances already used against the current candidate are skipped, so
        surviving m attempts means surviving m different checks; when the library
        runs out, the falsifier returns ⊥ and the candidate survives.
        """
        size = len(self._library)
        if size == 0:
            return None
        for _ in range(size * _MAX_LIBRARY_PASSES):
            relation = self._library[self._rel_cursor % size]
            self._rel_cursor += 1
            if not relation.programmatic:
                # pi_mr costs a model call, so dedupe *before* paying for it: one
                # instantiation per model-applied relation per candidate.
                if relation.name in self._seen:
                    continue
                if relation.applicable is not None and not relation.applicable(question):
                    self._seen.add(relation.name)
                    continue
                self._seen.add(relation.name)
                variant = self.build_variant(relation, question)
                if variant is not None:
                    return variant
                continue
            variant = self._instantiate(relation, question, answer)
            if variant is None:
                continue
            key = (variant.relation, variant.slot)
            if key in self._seen:
                continue
            self._seen.add(key)
            return variant
        return None

    def build_variant(self, relation: Relation, question: str) -> Optional[Variant]:
        """pi_mr: have the model apply a transformation it cannot get done in code.

        Used only for paraphrase-style relations. The prompt forbids solving
        either problem and never mentions the candidate, and the parsed
        ``RELATION:`` line is discarded — phi comes from the audited library.

        The result is then *mechanically validated*: a model-instantiated
        transform must carry over every numeric literal of the original, so a
        paraphrase that quietly drops or invents a quantity is rejected rather
        than checked against an invalid relation.
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

    def _voted_value(self, variant: Variant, answer: str) -> Tuple[Optional[str], int]:
        """The voted model value for a check instance, plus the vote size s.

        The single model-produced value in a relational check is the only place a
        hallucination could enter, so it is the majority over ``s`` independent
        completions of the *same* instance; ties and an UNDETERMINED majority
        abstain (None).
        """
        size = max(1, self.votes)
        if variant.family == "bwd":
            def call() -> str:
                return _parse_derived(self._mask_derivation(variant, answer))
        else:
            def call() -> str:
                return _extract_answer(
                    self.solve(variant.question, temperature=self.vote_temperature))
        return _majority(self._parallel(call, size)), size

    def _mask_derivation(self, variant: Variant, answer: str) -> str:
        """pi_mask: re-derive the hidden quantity, taking the candidate as given.

        The prompt is a forward derivation of one unknown with every other fact
        fixed — the setting in which LLM computation is most constrained — and it
        never asks for a judgement, so the eventual verdict is a pure Compare.
        """
        r = self._gen(
            "You are given a problem in which ONE stated quantity has been hidden "
            "and replaced by X, together with a candidate final answer to the "
            "original problem. Your job is NOT to judge the candidate answer. Take "
            "it as given, and re-derive the hidden quantity X from the candidate "
            "answer and the remaining information in the problem. Reason step by "
            "step.\n\n"
            f"Problem (one quantity hidden as X):\n{variant.question}\n\n"
            f"Candidate final answer (take as given):\n{answer}\n\n"
            "If X cannot be uniquely determined from the above, output exactly:\n"
            f"DERIVED: {UNDETERMINED}\n"
            "Otherwise, write the re-derived value on its own line, exactly as:\n"
            "DERIVED: <value of X>",
            context=self._task_ctx,
            temperature=self.vote_temperature)
        return r.content

    def _falsify_relational(self, question: str, answer: str) -> Optional[Witness]:
        """One relation instance: the model constructs, the comparator decides."""
        variant = self._next_instance(question, answer)
        if variant is None:
            return None                       # nothing applicable left: survives

        if variant.family == "inv":            # no model call: both sides computed
            observed = variant.observed or ""
            if variant.holds(answer, observed):
                return None
            return self._make_witness(variant, answer, observed, votes=0)

        observed, size = self._voted_value(variant, answer)
        if observed is None:
            return None                       # tie or UNDETERMINED: abstain
        if variant.holds(answer, observed):
            return None                       # phi holds: the candidate survives
        return self._make_witness(variant, answer, observed, votes=size)

    def _make_witness(self, variant: Variant, answer: str, observed: str,
                      votes: int) -> Witness:
        """Render w = ⟨rho, instance, (expected, observed)⟩ for pi_rep."""
        expected = variant.expected_value(answer) or "(not determined)"
        if variant.family == "bwd":
            text = (f"Backward check on the stated quantity {variant.slot!r}: "
                    f"taking the final answer {answer!r} as given, the hidden "
                    f"quantity X re-derives to {observed!r} by majority over "
                    f"{votes} independent derivations, but the problem states "
                    f"{expected!r}.\n\n"
                    f"The problem with that quantity hidden as X:\n{variant.question}")
        elif variant.family == "inv":
            text = (f"Invariant check [{variant.relation}]: {variant.detail}. "
                    f"The check required {expected!r} but the problem data gives "
                    f"{observed!r}.")
        else:
            text = (f"Metamorphic check [{variant.relation}]: {variant.relation_text}. "
                    f"The transformed problem below was solved independently "
                    f"{votes} time(s) without your answer being shown; the majority "
                    f"answer is {observed!r}, while your answer {answer!r} requires "
                    f"it to be {expected!r}.\n\n"
                    f"The transformed problem:\n{variant.question}")
        return Witness(
            text=text,
            sound=False,
            regime="relational",
            family=variant.family,
            relation=variant.relation,
            expected=expected,
            observed=observed,
            instance=variant.question,
            votes=votes,
        )

    # ── Repair : (q, a, w, H) → a'  (Algorithm 3, pi_rep) ───────────────────────
    def repair(self, question: str, candidate: str, witness: Witness,
               history: List[str]) -> str:
        """Produce a revised candidate that specifically resolves ``witness``.

        Because the witness localises the error — down to a named quantity and a
        pair of conflicting values — repair is targeted rather than a blind
        re-attempt: the model is told not only *that* the answer failed but
        exactly *on what*. The history H is carried forward so repairs do not
        reintroduce previously resolved failures, and for pair-implicating
        metamorphic witnesses the prompt explicitly licenses the model to defend
        its answer, so a rare wrong follow-up cannot force a wrong repair.
        """
        hist = "(none)"
        if self.carry_witness_history and history:
            hist = "\n".join(f"- {w}" for w in history)

        r = self._gen(
            "A previous answer to the problem below failed a mechanical check. The "
            "witness states the exact contradiction that was found: which value was "
            "checked, what the check expected, and what was observed instead. Use it "
            "to produce a corrected answer that specifically resolves this "
            "contradiction. If the witness compares your answer against a transformed "
            "variant of the problem and you conclude, after re-deriving both sides, "
            "that the variant's answer -- not yours -- is at fault, you may restate "
            "your previous answer. Make sure your answer does not reintroduce any of "
            "the past failures listed. Reason step by step.\n\n"
            f"Problem:\n{question}\n\n"
            f"Previous answer:\n{_extract_answer(candidate)}\n\n"
            f"Witness (the contradiction found by the check):\n{witness.text}\n\n"
            f"Past failures to avoid:\n{hist}\n\n"
            "When finished, write the final answer on its own line, exactly as:\n"
            "ANSWER: <your answer>",
            context=self._solve_ctx,
            temperature=self.repair_temperature)
        return r.content.strip()

    # ── pi_mr-gen: model-proposed library (ablation only) ───────────────────────
    def _generate_catalogue(self, question: str) -> None:
        """Build R_q with the model instead of by hand (library ablation).

        Generated once per run from one example problem, since R_q is a property
        of the task rather than of the query. Only proposals whose stated relation
        is answer-preserving are kept — phi must be evaluable by the driver.
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
            self._library = generated
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

        self._reset_falsifier()
        candidate = self.solve(question)                  # a ← Solve(q)
        initial_answer = _extract_answer(candidate)
        trace: List[str] = [f"[Solve] {initial_answer!r}"]
        history: List[str] = []                           # H ← ∅
        records: List[Dict[str, Any]] = []
        attempts = self._attempts_per_round()             # m
        # Optional deviation from Alg. 4 (``archive=True``): remember how many
        # checks each candidate withstood, so budget exhaustion can return the
        # best-corroborated one instead of the last step of the walk. A candidate
        # refuted by a SOUND witness is dropped outright: it is provably wrong.
        seen_candidates: List[Tuple[str, int]] = [(candidate, 0)]

        accepted = False
        for k in range(self.budget):                      # for k ← 1 to K
            witness: Optional[Witness] = None
            survived = 0
            for _ in range(attempts):                     # for j ← 1 to m
                witness = self.falsify(question, candidate)
                if witness is not None:
                    break
                survived += 1
            if witness is None:                           # survived m checks
                accepted = True
                trace.append(f"[Falsify k={k + 1}] survived {survived} distinct "
                             f"check(s) → accept fixpoint")
                break

            witness.survived = survived
            records.append(witness.record())
            trace.append(
                f"[Falsify k={k + 1}] witness ({witness.regime}/{witness.family}"
                f":{witness.relation}, sound={witness.sound}, survived={survived}): "
                f"{witness.text.splitlines()[0]}")
            if self.archive:
                if witness.sound:
                    seen_candidates.pop()                 # provably wrong
                else:
                    seen_candidates[-1] = (seen_candidates[-1][0], survived)

            history.append(witness.text)                  # H ← H ∪ {w}
            candidate = self.repair(question, candidate, witness, history[:-1])
            # A repaired candidate is a new candidate: re-open every check for it.
            self._seen.clear()
            if self.archive:
                seen_candidates.append((candidate, 0))
            trace.append(f"[Repair k={k + 1}] → {_extract_answer(candidate)!r}")

        returned = "fixpoint" if accepted else "last"
        if not accepted and self.archive and seen_candidates:
            best = max(range(len(seen_candidates)), key=lambda i: seen_candidates[i][1])
            if seen_candidates[best][0] != candidate:
                candidate = seen_candidates[best][0]
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
                "regime": "executable" if self._has_checker else "relational",
                "has_checker": self._has_checker,
                "library": [r.name for r in self._library],
                "accepted_fixpoint": accepted,
                "returned": returned,
                "budget": self.budget,
                "survival": attempts,
                "votes": self.votes,
                "repairs_used": len(records),
                "budget_exhausted": not accepted,
                "execute_code": self.execute_code,
                # Mechanism-level evaluation (§3.6): the initial/final pair gives
                # the flip matrix, and the witness records give witness precision
                # per regime and per relation family, once ground truth is joined.
                "initial_answer": initial_answer,
                "answer_changed": normalize_answer(initial_answer)
                                  != normalize_answer(final_answer),
                "witnesses": records,
                "falsification_regimes": [r["regime"] for r in records],
                "witness_history": history,
            },
        )

    def __repr__(self) -> str:
        return (f"FoT(baseline_name='{self.baseline_name}', "
                f"llm={self.llm.__class__.__name__}, task={self.task!r}, "
                f"subtask={self.subtask!r}, budget={self.budget}, "
                f"survival={self.survival}, votes={self.votes}, "
                f"has_checker={self._has_checker}, "
                f"library={[r.name for r in self._library]})")
