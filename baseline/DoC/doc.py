"""
Dynamic Operator Composition (DOC) implementation.

DOC recasts reasoning as the *incremental, verification-guided synthesis* of a
per-instance reasoning program over a small algebra of typed operators, rather
than committing a-priori to one fixed topology (CoT chain, ToT tree, …).

The loop (Algorithm 1 of the proposal):

    s  ← InitState(x)
    τ  ← Probe(x)                       # initial obligation, typically Unanchored
    while budget>0 and τ≠∅ and stall<m:
        o  ← Policy(τ, T(τ))            # pick one *type-eligible* operator
        s' ← o.Apply(s)                # may call the LLM and/or an external tool
        (accept, τ') ← Verify(s', s)   # three gates; V never advances the state
        if accept and ψ(s') < ψ(s):    # commit only on certified progress
            s, τ ← s', τ'
        else:
            τ ← escalate(τ)            # e.g. Faulty→Repair, weak→re-anchor
    return Extract(best state seen)     # anytime

The verifier never advances the state — it only *types the residual gap*, and
that type restricts which operator may fire next (the obligation lattice,
Table 2). Two mechanisms absent from the baselines live here:

  * external grounding — G renders the state to a runnable program and Verify
    executes it; a counterexample becomes a located Faulty obligation that
    Repair fixes locally instead of restarting (lifts the execution-task ceiling);
  * a confidence gate — low-confidence proposals are distrusted and re-anchored
    toward planning/grounding, which is where fixed search methods collapse.

ψ (the potential) is the severity rank of the current obligation: every
committed operator strictly lowers it (Unanchored→…→Unverified→∅), giving the
termination argument. Existing methods are fixed compositions in this algebra
(CoT≡D+elaborate, ToT≡iterated D,B,V, GoT≡iterated B,A, BoT/RoT≡Φ then solve).

Reference: Egor, "DOC: Dynamic Operator Composition for Verification-Guided
Reasoning" (2026).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from baseline.basebaseline import BaseBaseline, BaselineResponse
from models.base import BaseLLM

# Reuse BoT's hardened code extraction / sandboxed execution (the G operator).
from baseline.BoT.bot import extract_code, run_code


# ── Obligation types (the lattice T of Table 2) ────────────────────────────────
# Each names the single most salient gap in the state; the severity rank doubles
# as the potential ψ, so committing an operator that discharges a gap strictly
# decreases ψ. ∅ (SOLVED) has rank 0 and halts the loop.
SOLVED = "∅"
UNVERIFIED = "Unverified"
FAULTY = "Faulty"
FRAGMENTED = "Fragmented"
UNDEREXPLORED = "Underexplored"
DECOMPOSABLE = "Decomposable"
UNANCHORED = "Unanchored"

SEVERITY = {
    SOLVED: 0,
    UNVERIFIED: 1,
    FAULTY: 1,
    FRAGMENTED: 2,
    UNDEREXPLORED: 3,
    DECOMPOSABLE: 4,
    UNANCHORED: 5,
}

# Type-directed map T: τ → eligible operators. An operator may fire only if the
# current gap-type licenses it — this is what prevents negative transfer (Branch
# cannot be applied to a state whose gap is merely "unverified").
TYPE_MAP = {
    UNANCHORED: ["Φ"],        # no plan → Abstract
    DECOMPOSABLE: ["D"],      # goal too coarse → Decompose
    UNDEREXPLORED: ["B"],     # brittle path → Branch
    FRAGMENTED: ["A"],        # partial results unreconciled → Aggregate
    UNVERIFIED: ["G", "V"],   # claim unchecked → Ground (or accept)
    FAULTY: ["R"],            # located error → Repair
}


@dataclass
class State:
    """Reasoning state s = ⟨goal, facts, steps, plan⟩, lightly extended."""

    goal: str
    plan: str = ""
    subgoals: List[str] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    draft: str = ""                       # current working solution (may hold code)
    answer: str = ""                      # extracted / executed final answer
    cex: str = ""                         # last executor counterexample (for Repair)
    tau: str = UNANCHORED                 # obligation this state was typed with
    last_op: str = ""                     # operator that produced this state
    gen_logprob: Optional[float] = None   # confidence of that generation
    repairs: int = 0

    def clone(self) -> "State":
        return replace(self, subgoals=list(self.subgoals), candidates=list(self.candidates))


def _extract_answer(text: str) -> str:
    """Pull the final answer from an '### Answer' section, else the last line."""
    m = re.search(r"###\s*Answer\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


class DoC(BaseBaseline):
    """Dynamic Operator Composition prompting baseline."""

    def __init__(
        self,
        llm: BaseLLM,
        max_iterations: int = 10,
        patience: int = 3,
        confidence_threshold: float = 0.30,
        branch_k: int = 3,
        gen_temperature: float = 0.7,
        execute_code: bool = True,
        max_code_repairs: int = 3,
        code_timeout: float = 10.0,
    ) -> None:
        super().__init__(llm, baseline_name="DoC")
        self.max_iterations = max_iterations          # token/effort budget B (in iterations)
        self.patience = patience                      # m: stop after m non-improving steps
        self.confidence_threshold = confidence_threshold  # θ for the confidence gate
        self.branch_k = branch_k
        self.gen_temperature = gen_temperature
        self.execute_code = execute_code
        self.max_code_repairs = max_code_repairs
        self.code_timeout = code_timeout
        self._ctx = ""                                # system_prompt + instruction, set per run

    # ── Operators Ω = {D, B, A, Φ, G, R} ──────────────────────────────────────
    # Each maps a state to a new state and may call M and/or an external tool.

    def _gen(self, prompt: str, temperature: float, logprobs: bool = False):
        return self.call_llm(f"{self._ctx}\n\n{prompt}" if self._ctx else prompt,
                             temperature=temperature, logprobs=logprobs)

    def op_abstract(self, s: State) -> State:
        """Φ — lift the instance to a high-level plan / underlying rule."""
        r = self._gen(
            f"Problem:\n{s.goal}\n\n"
            "Produce a concise high-level PLAN (the strategy or underlying rule) for "
            "solving this problem. Do NOT solve it yet — 3–6 short bullet steps.",
            temperature=self.gen_temperature, logprobs=True)
        n = s.clone(); n.plan = r.content.strip(); n.gen_logprob = r.avg_logprob
        return n

    def op_decompose(self, s: State) -> State:
        """D — split the goal into ordered subgoals."""
        r = self._gen(
            f"Problem:\n{s.goal}\n\nPlan:\n{s.plan}\n\n"
            "Break the problem into an ordered list of concrete sub-goals, one per "
            "line, numbered. Do not solve them yet.",
            temperature=0.0)
        subgoals = [re.sub(r"^\s*\d+[.)]\s*", "", ln).strip()
                    for ln in r.content.splitlines() if ln.strip()]
        n = s.clone(); n.subgoals = subgoals or [s.goal]
        return n

    def op_branch(self, s: State) -> State:
        """B — generate k candidate continuations of the current subgoal."""
        plan = f"Plan:\n{s.plan}\n\n" if s.plan else ""
        subs = ("Sub-goals:\n" + "\n".join(f"- {g}" for g in s.subgoals) + "\n\n") if s.subgoals else ""
        r = self._gen(
            f"Problem:\n{s.goal}\n\n{plan}{subs}"
            f"Propose {self.branch_k} DISTINCT candidate solutions (each with its full "
            "reasoning and a concrete final answer). Separate them with a line '---'.",
            temperature=self.gen_temperature, logprobs=True)
        cands = [c.strip() for c in r.content.split("---") if c.strip()]
        n = s.clone(); n.candidates = cands or [r.content.strip()]; n.gen_logprob = r.avg_logprob
        return n

    def op_aggregate(self, s: State) -> State:
        """A — merge multiple partial results into one coherent solution."""
        joined = "\n\n".join(f"[Candidate {i + 1}]\n{c}" for i, c in enumerate(s.candidates))
        r = self._gen(
            f"Problem:\n{s.goal}\n\n{joined}\n\n"
            "Compare and reconcile the candidate solutions, then produce a single best "
            "solution. End with a line:\n### Answer\n<final answer in the exact format "
            "the problem requires>",
            temperature=0.0)
        n = s.clone(); n.candidates = []; n.draft = r.content.strip()
        n.answer = _extract_answer(n.draft)
        return n

    def op_ground(self, s: State) -> State:
        """G — render the state to an executable program (Verify runs it)."""
        guide = f"Plan / partial reasoning:\n{s.plan or s.draft}\n\n" if (s.plan or s.draft) else ""
        r = self._gen(
            f"Problem:\n{s.goal}\n\n{guide}"
            "Write a COMPLETE, self-contained Python program that SOLVES the problem by "
            "searching/computing (never guessing a single candidate) and prints ONLY the "
            "final answer to stdout, on one line, formatted EXACTLY as the problem "
            "requires (print the expression/equation/move itself, not merely its value). "
            "Wrap it in a single ```python ... ``` block.",
            temperature=0.0)
        n = s.clone(); n.draft = r.content.strip(); n.gen_logprob = r.avg_logprob
        return n

    def op_repair(self, s: State) -> State:
        """R — locally edit the step located as faulty by the executor."""
        code = extract_code(s.draft) or s.draft
        r = self._gen(
            "The following program failed when executed. Using the code and the error, "
            "return a corrected, COMPLETE Python program that runs and prints ONLY the "
            "final answer on one line, in the exact required format. Preserve the "
            "program's search intent; never hard-code a single value.\n\n"
            f"[Code]\n```python\n{code}\n```\n\n[Execution Error]\n{s.cex}",
            temperature=0.0)
        n = s.clone(); n.draft = r.content.strip(); n.repairs = s.repairs + 1
        return n

    _DISPATCH = {
        "Φ": op_abstract, "D": op_decompose, "B": op_branch,
        "A": op_aggregate, "G": op_ground, "R": op_repair,
    }

    # ── Policy: pick one operator from the type-eligible set Ω_τ = T(τ) ─────────
    def policy(self, tau: str) -> str:
        ops = TYPE_MAP[tau]
        if tau == UNVERIFIED:
            # Ground the checkable claim when execution is enabled, else accept (V).
            return "G" if self.execute_code else "V"
        return ops[0]

    # ── Verify: three gates in order; never advances the state ──────────────────
    def verify(self, s: State) -> Tuple[bool, str]:
        # (1) External grounding — a deterministic executor, not the model, decides
        # correctness, so it does not co-degrade with model quality.
        if self.execute_code and extract_code(s.draft):
            res = run_code(extract_code(s.draft), timeout=self.code_timeout)
            if res.success:
                s.answer = res.output
                return True, SOLVED
            s.cex = res.error
            return False, FAULTY
        # (2) Confidence gate — distrust low-confidence generations and re-anchor
        # toward Φ/G, away from Branch whose value hinges on strong generation.
        if (s.last_op in ("Φ", "B") and s.gen_logprob is not None
                and s.gen_logprob < self.confidence_threshold):
            return False, UNANCHORED
        # (3) Self-check — emit one typed residual obligation for s.
        return True, self.type_gap(s)

    def type_gap(self, s: State) -> str:
        """Type the residual gap structurally; the lattice encodes 'what is missing'."""
        if not s.plan:
            return UNANCHORED
        if not s.subgoals and not s.draft:
            return DECOMPOSABLE
        if s.candidates:
            return FRAGMENTED
        if not s.draft:
            return UNDEREXPLORED
        # A draft exists: ask the model whether it is final, or hides a checkable
        # computation worth grounding (only meaningful when execution is enabled).
        if not self.execute_code:
            return SOLVED
        r = self._gen(
            f"Problem:\n{s.goal}\n\nCandidate solution:\n{s.draft}\n\n"
            "Reply with ONE word: 'SOLVED' if this is already a complete, final answer "
            "needing no external checking, or 'UNVERIFIED' if it contains a computation, "
            "search, or arithmetic that should be executed to confirm it.",
            temperature=0.0)
        return UNVERIFIED if "UNVERIFIED" in r.content.upper() else SOLVED

    @staticmethod
    def escalate(tau: str) -> str:
        """Soft no-progress: widen toward a more structural (higher-severity) gap."""
        order = [SOLVED, UNVERIFIED, FRAGMENTED, UNDEREXPLORED, DECOMPOSABLE, UNANCHORED]
        i = order.index(tau) if tau in order else 0
        return order[min(i + 1, len(order) - 1)]

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

        s = State(goal=question)                 # InitState(x)
        tau = UNANCHORED                          # Probe(x)
        best = s                                  # anytime: best state seen
        best_psi = SEVERITY[tau]
        stall = 0
        trace: List[str] = [f"[Probe] τ={tau}"]

        budget = self.max_iterations
        while budget > 0 and tau != SOLVED and stall < self.patience:
            op = self.policy(tau)
            if op == "V":                         # verifier-only: accept current draft
                break
            s_new = self._DISPATCH[op](self, s)
            s_new.last_op = op
            accept, tau_new = self.verify(s_new)
            budget -= 1
            trace.append(f"[{op}] τ={tau} → accept={accept}, τ'={tau_new}")

            if accept and SEVERITY[tau_new] < SEVERITY[tau]:
                s_new.tau = tau_new
                s, tau = s_new, tau_new           # commit; advance the obligation
                if SEVERITY[tau] < best_psi:
                    best, best_psi, stall = s, SEVERITY[tau], 0
                else:
                    stall += 1
            elif tau_new == FAULTY and s_new.repairs < self.max_code_repairs:
                # Carry the faulty (executable) draft forward so Repair can fix it
                # locally — the located counterexample beats restarting.
                s_new.tau, s, tau = FAULTY, s_new, FAULTY
                stall += 1
            elif not accept:
                tau = tau_new                     # hard-fail escalation (weak → Unanchored)
                stall += 1
            else:
                tau = self.escalate(tau)          # soft no-progress: widen
                stall += 1

        final_answer = best.answer or _extract_answer(best.draft) or best.draft
        reasoning_trace = (
            f"[Plan]\n{best.plan}\n\n[Sub-goals]\n" + "\n".join(best.subgoals)
            + f"\n\n[Solution]\n{best.draft}"
        )
        return self.create_response(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            intermediate_steps=trace,
            metadata={
                "final_obligation": tau,
                "iterations_used": self.max_iterations - budget,
                "operator_trace": [t.split("]")[0].lstrip("[") for t in trace[1:]],
                "best_psi": best_psi,
                "execute_code": self.execute_code,
                "confidence_threshold": self.confidence_threshold,
            },
        )

    def __repr__(self) -> str:
        return (f"DoC(baseline_name='{self.baseline_name}', "
                f"llm={self.llm.__class__.__name__}, "
                f"max_iterations={self.max_iterations}, "
                f"execute_code={self.execute_code})")
