"""
Falsification-of-Thought (FoT) implementation.

FoT makes *self-refutation* the engine of reasoning. Instead of generating and
selecting, it generates a candidate and then repeatedly tries to *break* it,
using each successful break to drive a targeted repair. The guiding principle is
to **falsify, not critique**: feedback is a constructive *witness* (a concrete
counterexample, a failing input, or a violated necessary condition), never a
free-form opinion of the model's own work.

The driver loop (Algorithm 4 of the paper):

    a ← Solve(q)                          # initial candidate (e.g. CoT)
    H ← ∅                                 # witness history
    for k in 1..K:                        # budget K
        w ← ⊥
        for j in 1..m:                    # m independent falsification attempts
            w ← Falsify(q, a)             # executable or semantic witness
            if w ≠ ⊥: break
        if w = ⊥:
            return a                       # survived m attempts: accept fixpoint
        H ← H ∪ {w}
        a ← Repair(q, a, w, H)            # witness-guided, targeted repair
    return a                               # budget exhausted: last candidate

FoT composes three operators, each a prompt to a single frozen model M:

  * Solve  : q → a            produce an initial candidate (any base reasoner).
  * Falsify: (q, a) → w | ⊥   construct a witness that a is wrong, or ⊥.
  * Repair : (q, a, w) → a'   revise a to specifically resolve the witness w.

The falsifier (Algorithm 2) operates in one of two regimes, selected by the
task-level predicate ``HasChecker(q)`` — FIXED PER BENCHMARK, not decided at run
time by the model (see :mod:`baseline.FoT.checkers`):

  * Executable falsification — when the benchmark exposes a cheap, decisive
    checker c_q (arithmetic evaluation for Game of 24, program execution for
    CRUXEval, ``sat`` for Python Puzzles). The model only *proposes a probe*
    (pi_probe) — it is forbidden to give a verdict; a trusted, external c_q
    decides. A failure is therefore a SOUND witness: it certifies that the
    candidate is wrong, so a correct candidate is never discarded on a spurious
    witness, and survival is a sound pass.

  * Semantic falsification — when no cheap checker exists, the model first derives
    necessary conditions N(q) that any correct answer must satisfy (pi_nec, from
    the problem alone, *without* seeing the candidate), then a SEPARATE call
    checks whether the candidate violates them (pi_chk, instructed to re-derive
    every check and not trust the candidate). A semantic witness is not sound, so
    it is admitted only if it passes the ``ReCheckable`` guard (self-contained,
    re-derivable from q) and the candidate must survive m independent attempts
    before being accepted, reducing false refutations.

Acceptance is *corroboration* in the Popperian sense: a candidate is accepted not
because it was proven correct, but because honest attempts to break it failed.
FoT issues a single query per step for O(K·(1+m)) model calls — far below the
recursive expansion of ToT/GoT.

Reference: "Falsification-of-Thought: Reasoning by Self-Refutation" (2026).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from baseline.basebaseline import BaseBaseline, BaselineResponse
from baseline.FoT.checkers import CheckResult, Checker, get_checker
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


def _parse_conditions(text: str) -> Dict[str, str]:
    """Parse the numbered ``N1:/N2:/...`` necessary conditions (pi_nec output)."""
    conds: Dict[str, str] = {}
    for m in re.finditer(r"^\s*N(\d+)\s*[:.)]\s*(.+)$", text, re.MULTILINE):
        body = m.group(2).strip()
        if body:
            conds[f"N{m.group(1)}"] = body
    return conds


def _parse_check(text: str) -> tuple[bool, str, str]:
    """Parse pi_chk's ``VIOLATION:/CONDITION:/WITNESS:`` triple."""
    viol = re.search(r"VIOLATION:\s*([A-Za-z]+)", text, re.IGNORECASE)
    cond = re.search(r"CONDITION:\s*([^\n]+)", text, re.IGNORECASE)
    wit = re.search(r"WITNESS:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    viol_yes = bool(viol) and viol.group(1).strip().lower().startswith("y")
    cond_id = cond.group(1).strip() if cond else ""
    witness = wit.group(1).strip() if wit else ""
    return viol_yes, cond_id, witness


@dataclass
class Witness:
    """A constructive artifact demonstrating that a candidate is wrong.

    Attributes:
        text: human-readable, self-contained description of the failure.
        sound: True iff produced by the executable regime, where a trusted
            external checker c_q reported a real failure — in which case it
            certifies the candidate is incorrect. Semantic witnesses are not sound.
        regime: "executable" or "semantic".
    """

    text: str
    sound: bool
    regime: str


class FoT(BaseBaseline):
    """Falsification-of-Thought prompting baseline."""

    def __init__(
        self,
        llm: BaseLLM,
        task: Optional[str] = None,
        budget: int = 4,
        survival_threshold: int = 2,
        solve_temperature: float = 0.0,
        falsify_temperature: float = 0.7,
        repair_temperature: float = 0.0,
        execute_code: bool = True,
        code_timeout: float = 10.0,
        carry_witness_history: bool = True,
    ) -> None:
        """Initialize the FoT baseline.

        Args:
            llm: An instance of a BaseLLM subclass.
            task: Benchmark name (e.g. "gameof24", "cruxeval"). Selects the
                trusted checker c_q and fixes ``HasChecker(q)`` for the whole run.
                A benchmark with a registered checker uses the executable (sound)
                regime; any other uses the semantic regime.
            budget: K — maximum number of Falsify→Repair iterations.
            survival_threshold: m — independent falsification attempts a candidate
                must survive before it is accepted (mitigates false refutations in
                the semantic regime).
            solve_temperature: Sampling temperature for the initial Solve.
            falsify_temperature: Sampling temperature for falsification. Non-zero so
                the m semantic attempts are genuinely independent (different
                conditions / witnesses).
            repair_temperature: Sampling temperature for witness-guided Repair.
            execute_code: If True, use the executable regime on benchmarks that have
                a trusted checker c_q. If False, force the semantic regime
                everywhere. The checker runs the benchmark's own reference code
                (CRUXEval functions, Python-Puzzles ``sat``) in a timeout-bounded
                subprocess; disable on an untrusted host if that is a concern.
            code_timeout: Per-execution timeout in seconds for the checker.
            carry_witness_history: If True, carry the history of past witnesses into
                Repair so fixes do not reintroduce previously resolved failures
                (discouraging oscillation).
        """
        super().__init__(llm, baseline_name="FoT")
        self.task = task
        self.budget = budget
        self.survival_threshold = survival_threshold
        self.solve_temperature = solve_temperature
        self.falsify_temperature = falsify_temperature
        self.repair_temperature = repair_temperature
        self.execute_code = execute_code
        self.code_timeout = code_timeout
        self.carry_witness_history = carry_witness_history

        # HasChecker(q): fixed per benchmark, not decided at run time by the model.
        self._checker: Optional[Checker] = get_checker(task) if execute_code else None
        self._has_checker = self._checker is not None

        # Context split (persona scoping): the answer-producing operators (Solve,
        # Repair) receive the full task framing + answer-format directives so their
        # output stays parseable; the falsifier operators receive only a one-line
        # task description so the format persona never conflicts with "propose a
        # probe" / "list necessary conditions".
        self._solve_ctx = ""   # system_prompt + instruction
        self._task_ctx = ""    # one-line task description only

    # ── helper ─────────────────────────────────────────────────────────────────
    def _gen(self, prompt: str, context: str, temperature: float):
        full = f"{context}\n\n{prompt}" if context else prompt
        return self.call_llm(full, temperature=temperature)

    # ── Solve : q → a  (Algorithm 1, pi_solve) ──────────────────────────────────
    def solve(self, question: str) -> str:
        """Produce an initial candidate answer with a CoT-style base reasoner."""
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
    def falsify(self, question: str, candidate: str) -> Optional[Witness]:
        """Attempt to construct a witness that ``candidate`` is wrong.

        Dispatches on the fixed per-benchmark ``HasChecker(q)`` predicate: the
        executable regime when a trusted checker exists, the semantic regime
        otherwise. The regime does not change across attempts or instances.
        """
        answer = _extract_answer(candidate)
        if self._has_checker:
            return self._falsify_executable(question, candidate, answer)
        return self._falsify_semantic(question, candidate, answer)

    def _falsify_executable(
        self, question: str, candidate: str, answer: str
    ) -> Optional[Witness]:
        """Executable regime (pi_probe + trusted c_q).

        The model proposes ONE probe (it is forbidden to give a verdict); the
        external checker c_q decides. A failure is a sound witness; a pass (or an
        undecidable check) is a sound survival — c_q never reports a false failure.
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
        if result.verdict == "fail":
            return Witness(text=result.detail, sound=True, regime="executable")
        # "pass" or "undecided": the candidate survives this decisive check.
        return None

    def _falsify_semantic(
        self, question: str, candidate: str, answer: str
    ) -> Optional[Witness]:
        """Semantic regime: pi_nec then a SEPARATE pi_chk, guarded by ReCheckable.

        We ask "what must any correct answer satisfy, and does this one violate
        it?" — the construction of a discriminating test, which LLMs do more
        reliably than holistic self-judgement. pi_nec sees only the problem (not
        the candidate) so the conditions are not reverse-engineered from the
        answer; pi_chk then checks the candidate against them on a separate call.
        """
        # Step 1 (pi_nec): necessary conditions from the problem alone.
        r_nec = self._gen(
            "List the necessary conditions that ANY correct answer to the problem "
            "below must satisfy. Each condition must be (a) checkable using only "
            "the problem statement, and (b) stated precisely enough to test. Do "
            "not attempt to solve the problem.\n\n"
            f"Problem:\n{question}\n\n"
            "Output a numbered list, one condition per line:\n"
            "N1: <condition>\nN2: <condition>\n...",
            context=self._task_ctx,
            temperature=self.falsify_temperature)
        conditions = _parse_conditions(r_nec.content)
        if not conditions:
            return None  # no testable condition derived: nothing to falsify with

        cond_block = "\n".join(f"{cid}: {body}" for cid, body in conditions.items())

        # Step 2 (pi_chk): does the candidate violate a condition? Re-derive every
        # check from the problem; do not trust the candidate's own reasoning.
        r_chk = self._gen(
            "You are given a problem, a candidate answer, and necessary "
            "conditions. Decide whether the candidate VIOLATES any condition. "
            "Re-derive every check from the problem itself; do not trust the "
            "candidate's own reasoning. If it violates a condition, give a "
            "concrete, self-contained demonstration (numbers, a trace, or a "
            "specific case) that someone could re-verify from the problem "
            "statement alone.\n\n"
            f"Problem:\n{question}\n\n"
            f"Candidate answer:\n{answer}\n\n"
            f"Necessary conditions:\n{cond_block}\n\n"
            "Output exactly these three lines:\n"
            "VIOLATION: <yes | no>\n"
            "CONDITION: <id of the violated condition, or none>\n"
            "WITNESS: <concrete self-contained demonstration, or none>",
            context=self._task_ctx,
            temperature=self.falsify_temperature)

        viol, cond_id, witness = _parse_check(r_chk.content)
        if viol and self._recheckable(conditions, cond_id, witness):
            cond_text = conditions.get(_canon_cond_id(cond_id), "")
            label = f"violates {cond_id}" + (f" ({cond_text})" if cond_text else "")
            return Witness(text=f"{label}: {witness}", sound=False, regime="semantic")
        return None  # survives this attempt

    @staticmethod
    def _recheckable(conditions: Dict[str, str], cond_id: str, witness: str) -> bool:
        """ReCheckable(q, N, v): keep only self-contained, re-derivable witnesses.

        The first line of defence against hallucinated semantic refutations
        (Algorithm 2). A witness is admitted only if it is concrete (non-empty,
        not "none"), names one of the derived conditions, and does not merely defer
        to the candidate's own reasoning.
        """
        w = witness.strip()
        if not w or w.lower() in {"none", "n/a", "na", "-"}:
            return False
        canon = _canon_cond_id(cond_id)
        if canon and canon not in conditions:
            return False  # cites a condition that was never derived from q
        # A self-contained witness carries concrete, re-derivable content — a value,
        # a quoted token, a bracketed structure, or an (in)equality. Reject witnesses
        # that merely appeal to the candidate's own claims with no such content.
        concrete = bool(re.search(r"[\d\"'\[\]]|[<>=≠]", w))
        defers = bool(re.search(r"\b(the candidate|as claimed|trust|its reasoning|"
                                r"the model('s)? (answer|reasoning))\b", w, re.IGNORECASE))
        if defers and not concrete:
            return False
        return True

    # ── Repair : (q, a, w, H) → a'  (Algorithm 3, pi_rep) ───────────────────────
    def repair(
        self, question: str, candidate: str, witness: Witness, history: List[str]
    ) -> str:
        """Produce a revised candidate that specifically resolves ``witness``.

        Because the witness localizes the error, repair is targeted rather than a
        blind re-attempt: the model is told not only that the candidate failed but
        exactly on what. The history H of past witnesses is carried forward so
        fixes do not reintroduce previously resolved failures.
        """
        hist = "(none)"
        if self.carry_witness_history and history:
            hist = "\n".join(f"- {w}" for w in history)
        r = self._gen(
            "A previous answer to the problem below was shown to be wrong. Use the "
            "witness -- a concrete demonstration of the failure -- to produce a "
            "corrected answer that specifically resolves it. Make sure your new "
            "answer does not reintroduce any of the past failures listed. Reason "
            "step by step.\n\n"
            f"Problem:\n{question}\n\n"
            f"Previous (wrong) answer:\n{_extract_answer(candidate)}\n\n"
            f"Witness (why it fails):\n{witness.text}\n\n"
            f"Past failures to avoid:\n{hist}\n\n"
            "When finished, write the final answer on its own line, exactly as:\n"
            "ANSWER: <your answer>",
            context=self._solve_ctx,
            temperature=self.repair_temperature)
        return r.content.strip()

    # ── Driver loop (Algorithm 4) ───────────────────────────────────────────────
    def _falsify_round(self, question: str, candidate: str) -> Optional[Witness]:
        """One falsification round = the inner m-loop of Algorithm 4.

        With a decisive checker a single probe settles the verdict, so the
        executable regime runs ONE check (paper's Alg 4 note); the stochastic
        semantic regime runs up to m independent attempts and survival across all
        m licenses acceptance.
        """
        if self._has_checker:
            return self.falsify(question, candidate)
        witness: Optional[Witness] = None
        for _ in range(self.survival_threshold):
            witness = self.falsify(question, candidate)
            if witness is not None:
                break
        return witness

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
        # falsifier gets only the one-line task description.
        self._solve_ctx = "\n\n".join(p for p in (system_prompt, instruction) if p)
        self._task_ctx = (instruction or system_prompt or "").strip()

        candidate = self.solve(question)          # a ← Solve(q)
        trace: List[str] = [f"[Solve] {_extract_answer(candidate)!r}"]
        witness_history: List[str] = []           # H ← ∅

        accepted = False
        regimes: List[str] = []
        iterations = 0
        for k in range(self.budget):
            witness = self._falsify_round(question, candidate)
            if witness is None:
                # Survived the falsification round → accept fixpoint.
                accepted = True
                trace.append(f"[Falsify k={k + 1}] survived → accept")
                break
            iterations += 1
            regimes.append(witness.regime)
            trace.append(
                f"[Falsify k={k + 1}] witness ({witness.regime}, "
                f"sound={witness.sound}): {witness.text}")
            witness_history.append(witness.text)            # H ← H ∪ {w}
            candidate = self.repair(question, candidate, witness, witness_history[:-1])
            trace.append(f"[Repair k={k + 1}] → {_extract_answer(candidate)!r}")

        final_answer = _extract_answer(candidate)
        return self.create_response(
            final_answer=final_answer,
            reasoning_trace=candidate,
            intermediate_steps=trace,
            metadata={
                "task": self.task,
                "regime": "executable" if self._has_checker else "semantic",
                "has_checker": self._has_checker,
                "accepted_fixpoint": accepted,
                "budget": self.budget,
                "survival_threshold": self.survival_threshold,
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
                f"budget={self.budget}, survival_threshold={self.survival_threshold}, "
                f"has_checker={self._has_checker})")


def _canon_cond_id(cond_id: str) -> str:
    """Normalise a cited condition id (e.g. 'N2', 'condition N2', '2') to 'N<k>'."""
    m = re.search(r"N?\s*(\d+)", cond_id)
    return f"N{m.group(1)}" if m else ""
