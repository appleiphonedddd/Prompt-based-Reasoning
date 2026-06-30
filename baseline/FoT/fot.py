"""
Falsification-of-Thought (FoT) implementation.

FoT makes *self-refutation* the engine of reasoning. Instead of generating and
selecting, it generates a candidate and then repeatedly tries to *break* it,
using each successful break to drive a targeted repair. The guiding principle is
to **falsify, not critique**: feedback is a constructive *witness* (a concrete
counterexample, a failing input, or a violated necessary condition), never a
free-form opinion of the model's own work.

The loop (Algorithm 1 of the paper):

    a ← Solve(q)                          # initial candidate (e.g. CoT)
    for k in 1..K:                        # budget K
        w ← ⊥
        for j in 1..m:                    # m independent falsification attempts
            w ← Falsify(q, a)             # executable or semantic witness
            if w ≠ ⊥: break
        if w = ⊥:
            return a                       # survived m attempts: accept fixpoint
        a ← Repair(q, a, w)               # witness-guided, targeted repair
    return a                               # budget exhausted: last candidate

FoT composes three operators, each a prompt to a single frozen model M:

  * Solve  : q → a            produce an initial candidate (any base reasoner).
  * Falsify: (q, a) → w | ⊥   construct a witness that a is wrong, or ⊥.
  * Repair : (q, a, w) → a'   revise a to specifically resolve the witness w.

The falsifier operates in one of two regimes, chosen by whether the task admits
a cheap checker:

  * Executable falsification — when a candidate can be checked by running code
    (arithmetic re-evaluation for Game of 24, a constraint predicate for
    Geometric Shapes, program execution for code tasks). The model proposes a
    *checker* program that recomputes / re-checks the candidate; a deterministic
    executor — not the model — decides. A failure here is a SOUND witness: it
    certifies that the candidate is wrong, so a correct candidate is never
    discarded on a spurious witness. A clean pass is a sound survival.

  * Semantic falsification — when no cheap checker exists, the model derives a
    necessary condition N(q) that any correct answer must satisfy and exhibits a
    concrete violation of N by the candidate; the witness is the pair
    (condition, violation). A semantic witness is not sound (the model may posit
    a spurious condition), so we require the candidate to survive m independent
    attempts before accepting, reducing the rate of false refutations.

Acceptance is *corroboration* in the Popperian sense: a candidate is accepted
not because it was proven correct, but because honest attempts to break it
failed. FoT issues a single query per step for O(K·(1+m)) model calls — far
below the recursive expansion of ToT/GoT.

Reference: "Falsification-of-Thought: Reasoning by Self-Refutation" (2026).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from baseline.basebaseline import BaseBaseline, BaselineResponse
from models.base import BaseLLM

# Reuse BoT's hardened code extraction / sandboxed execution for the executable
# falsification regime (the cheap checker c_q runs the model-proposed program).
from baseline.BoT.bot import extract_code, run_code


# Sentinels the executable checker prints so a deterministic executor — not the
# model's prose — decides the verdict, side-stepping hallucinated judgements.
_WITNESS_MARKER = "__FOT_WITNESS__"   # candidate refuted; text after = the witness
_SURVIVES_MARKER = "__FOT_OK__"       # candidate passed the executable check

# Sentinels for the semantic regime.
_SEM_WITNESS = "### Witness"
_SEM_SURVIVES = "### Survives"


def _extract_answer(text: str) -> str:
    """Pull the final answer from an '### Answer' section, else the last line."""
    m = re.search(r"###\s*Answer\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


@dataclass
class Witness:
    """A constructive artifact demonstrating that a candidate is wrong.

    Attributes:
        text: human-readable description of the violation (the witness itself).
        sound: True iff produced by the executable regime (a deterministic
            checker reported a real failure), in which case it certifies the
            candidate is incorrect. Semantic witnesses are not sound.
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
            budget: K — maximum number of Falsify→Repair iterations.
            survival_threshold: m — independent falsification attempts a candidate
                must survive before it is accepted (mitigates false refutations in
                the semantic regime).
            solve_temperature: Sampling temperature for the initial Solve.
            falsify_temperature: Sampling temperature for falsification. Non-zero so
                the m attempts are genuinely independent (different witnesses).
            repair_temperature: Sampling temperature for witness-guided Repair.
            execute_code: If True, use the executable falsification regime when the
                model can write a checker (sound witnesses). If False, always use
                the semantic regime. WARNING: runs untrusted model-generated code in
                a timeout-bounded subprocess; disable on an untrusted host.
            code_timeout: Per-execution timeout in seconds for the checker program.
            carry_witness_history: If True, carry a short history of past witnesses
                into Repair so fixes do not reintroduce previously resolved failures
                (discouraging oscillation).
        """
        super().__init__(llm, baseline_name="FoT")
        self.budget = budget
        self.survival_threshold = survival_threshold
        self.solve_temperature = solve_temperature
        self.falsify_temperature = falsify_temperature
        self.repair_temperature = repair_temperature
        self.execute_code = execute_code
        self.code_timeout = code_timeout
        self.carry_witness_history = carry_witness_history
        self._ctx = ""   # system_prompt + instruction, set per run

    # ── helper ─────────────────────────────────────────────────────────────────
    def _gen(self, prompt: str, temperature: float, logprobs: bool = False):
        return self.call_llm(f"{self._ctx}\n\n{prompt}" if self._ctx else prompt,
                             temperature=temperature, logprobs=logprobs)

    # ── Solve : q → a ───────────────────────────────────────────────────────────
    def solve(self, question: str) -> str:
        """Produce an initial candidate answer with a CoT-style base reasoner."""
        r = self._gen(
            f"Problem:\n{question}\n\n"
            "Solve the problem. Think step by step, then end with a line:\n"
            "### Answer\n<final answer in the exact format the problem requires>",
            temperature=self.solve_temperature)
        return r.content.strip()

    # ── Falsify : (q, a) → w | ⊥ ────────────────────────────────────────────────
    def falsify(self, question: str, candidate: str) -> Optional[Witness]:
        """Attempt to construct a witness that ``candidate`` is wrong.

        Tries the executable regime first (when enabled); a clean, decisive
        executor verdict is returned immediately. Falls back to the semantic
        regime when no cheap checker can be written or the program is unusable.
        Returns a Witness if the candidate is refuted, else None (survives).
        """
        answer = _extract_answer(candidate)

        if self.execute_code:
            verdict = self._falsify_executable(question, candidate, answer)
            if verdict is not None:
                witness, decided = verdict
                if decided:
                    # A deterministic checker ran and gave a sound verdict
                    # (witness or survival) — trust it, skip the semantic regime.
                    return witness
        # No usable executable checker: derive a necessary condition and look for
        # a concrete violation (semantic regime).
        return self._falsify_semantic(question, candidate, answer)

    def _falsify_executable(
        self, question: str, candidate: str, answer: str
    ) -> Optional[Tuple[Optional[Witness], bool]]:
        """Executable regime: model writes a checker, a real executor decides.

        Returns:
            None  — no usable checker was produced (caller falls back to semantic).
            (Witness, True)  — the checker soundly refuted the candidate.
            (None, True)     — the checker soundly confirmed survival.
        """
        r = self._gen(
            f"Problem:\n{question}\n\n"
            f"Proposed answer:\n{answer}\n\n"
            "Your job is to FALSIFY the proposed answer, not to praise it. Write a "
            "COMPLETE, self-contained Python program that ATTEMPTS TO REFUTE the "
            "proposed answer by checking it against a necessary condition derived "
            "ONLY from the problem statement (e.g. the expression evaluates to the "
            "target and uses each number exactly once; the configuration satisfies "
            "the stated geometric constraints; the program returns the claimed "
            "output / satisfies sat(...) on the given input). Embed the proposed "
            "answer in the program.\n"
            "The program MUST print EXACTLY ONE of:\n"
            f"  {_WITNESS_MARKER}<one concrete reason the answer fails: the failing "
            "input, the recomputed value, or the violated condition>\n"
            f"    -- if and only if the check proves the answer is WRONG, or\n"
            f"  {_SURVIVES_MARKER}\n"
            "    -- if the check passes and the answer survives.\n"
            "If the problem admits NO such cheap, decisive check, print nothing and "
            "leave the program empty. Wrap the code in one ```python ... ``` block.",
            temperature=self.falsify_temperature)

        code = extract_code(r.content)
        if not code:
            return None  # no checker → fall back to semantic

        res = run_code(code, timeout=self.code_timeout)
        if not res.success:
            # The checker itself failed to run — it is not a sound oracle, so we
            # cannot trust either verdict. Fall back to the semantic regime.
            return None
        if _WITNESS_MARKER in res.output:
            reason = res.output.split(_WITNESS_MARKER, 1)[1].strip() or \
                "executable check failed"
            return Witness(text=reason, sound=True, regime="executable"), True
        if _SURVIVES_MARKER in res.output:
            return None, True   # sound survival
        # Ran cleanly but emitted no sentinel: not a decisive check.
        return None

    def _falsify_semantic(
        self, question: str, candidate: str, answer: str
    ) -> Optional[Witness]:
        """Semantic regime: derive a necessary condition, exhibit a violation.

        We ask "what must any correct answer satisfy, and does this one violate
        it?" — the construction of a discriminating test, which LLMs do more
        reliably than holistic self-judgement. The witness must be self-contained
        and re-checkable from the problem alone.
        """
        r = self._gen(
            f"Problem:\n{question}\n\n"
            f"Proposed answer:\n{answer}\n\n"
            "Try to FALSIFY the proposed answer. Do NOT ask 'is it correct?' — "
            "instead: (1) state ONE necessary condition that ANY correct answer to "
            "this problem must satisfy, and (2) check whether the proposed answer "
            "violates it. The condition must be self-contained and re-checkable "
            "from the problem statement alone.\n"
            "If the proposed answer concretely VIOLATES the condition, respond with:\n"
            f"{_SEM_WITNESS}\n<the necessary condition>: <the concrete violation by "
            "this answer>\n"
            "Otherwise, if it satisfies the condition and you cannot construct a "
            "genuine violation, respond with exactly:\n"
            f"{_SEM_SURVIVES}",
            temperature=self.falsify_temperature)

        text = r.content.strip()
        m = re.search(rf"{re.escape(_SEM_WITNESS)}\s*\n?(.*)", text, re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip() and _SEM_SURVIVES.lower() not in text.lower():
            return Witness(text=m.group(1).strip(), sound=False, regime="semantic")
        return None  # survives this attempt

    # ── Repair : (q, a, w) → a' ─────────────────────────────────────────────────
    def repair(
        self, question: str, candidate: str, witness: Witness, history: List[str]
    ) -> str:
        """Produce a revised candidate that specifically resolves ``witness``.

        Because the witness localizes the error, repair is targeted rather than a
        blind re-attempt: the model is told not only that the candidate failed but
        exactly on what. A short history of past witnesses is optionally carried
        forward so fixes do not reintroduce previously resolved failures.
        """
        hist = ""
        if self.carry_witness_history and history:
            hist = ("\nPreviously resolved failures (do NOT reintroduce these):\n"
                    + "\n".join(f"- {w}" for w in history) + "\n")
        r = self._gen(
            f"Problem:\n{question}\n\n"
            f"Previous answer (REFUTED):\n{_extract_answer(candidate)}\n\n"
            f"Falsifying witness — the previous answer fails because:\n{witness.text}\n"
            f"{hist}\n"
            "Repair the answer so that it specifically resolves this witness. Address "
            "exactly the failure above; do not merely restate the previous answer. "
            "Think step by step, then end with a line:\n"
            "### Answer\n<corrected answer in the exact format the problem requires>",
            temperature=self.repair_temperature)
        return r.content.strip()

    # ── Main loop (Algorithm 1) ─────────────────────────────────────────────────
    def run(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> BaselineResponse:
        self.reset_counters()
        self._ctx = "\n\n".join(p for p in (system_prompt, instruction) if p)

        candidate = self.solve(question)          # a ← Solve(q)
        trace: List[str] = [f"[Solve] {_extract_answer(candidate)!r}"]
        witness_history: List[str] = []

        accepted = False
        regimes: List[str] = []
        iterations = 0
        for k in range(self.budget):
            # Inner loop: m independent falsification attempts.
            witness: Optional[Witness] = None
            for _ in range(self.survival_threshold):
                witness = self.falsify(question, candidate)
                if witness is not None:
                    break
            if witness is None:
                # Survived m attempts (or one sound executable check) → fixpoint.
                accepted = True
                trace.append(f"[Falsify k={k + 1}] survived → accept")
                break
            iterations += 1
            regimes.append(witness.regime)
            trace.append(
                f"[Falsify k={k + 1}] witness ({witness.regime}, "
                f"sound={witness.sound}): {witness.text}")
            witness_history.append(witness.text)
            candidate = self.repair(question, candidate, witness, witness_history[:-1])
            trace.append(f"[Repair k={k + 1}] → {_extract_answer(candidate)!r}")

        final_answer = _extract_answer(candidate)
        return self.create_response(
            final_answer=final_answer,
            reasoning_trace=candidate,
            intermediate_steps=trace,
            metadata={
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
                f"llm={self.llm.__class__.__name__}, "
                f"budget={self.budget}, survival_threshold={self.survival_threshold}, "
                f"execute_code={self.execute_code})")
