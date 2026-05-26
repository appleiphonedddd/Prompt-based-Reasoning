"""
Graph of Thoughts (GoT) Prompting Implementation.

GoT models LLM reasoning as an arbitrary directed graph G = (V, E),
where each vertex is an LLM thought and each directed edge represents
a dependency between thoughts.  This generalises Chain-of-Thought (CoT)
and Tree-of-Thoughts (ToT) by enabling three graph-native transformations:

  1. Generate  – branch one thought into k new thoughts.
  2. Aggregate – merge k thoughts into one combined thought.
  3. Refine    – loop over a thought to iteratively improve it.

The concrete execution schedule is determined by a Graph of Operations
(GoO).  For general question-answering the default GoO is:

  Phase 0 – Generate k independent reasoning paths.
  Phase 1 – Score every path.
  Phase 2 – KeepBest N paths.
  Phase 3 – Aggregate the survivors into a single answer.
  Phase 4 – Refine the aggregate answer (optional, repeat r times).

This is the same decompose → solve → merge pattern described in §4 of
the paper, adapted to generic Q&A rather than the sorting use-case.

Reference:
- Besta, M. et al. (2024). Graph of Thoughts: Solving Elaborate
  Problems with Large Language Models. AAAI-24.

Author: (your name here)
"""

from __future__ import annotations

import re
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from baseline.basebaseline import BaseBaseline, BaselineResponse
from models.base import BaseLLM

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Domain objects
# ─────────────────────────────────────────────────────────────────────────────

class ThoughtStatus(Enum):
    """Lifecycle status of a single Thought vertex."""
    PENDING   = auto()   # created, not yet scored
    SCORED    = auto()   # score has been assigned
    KEPT      = auto()   # survived KeepBest selection
    PRUNED    = auto()   # eliminated by KeepBest
    REFINED   = auto()   # has been through at least one Refine pass
    AGGREGATED = auto()  # this thought was used as input to Aggregate


@dataclass
class Thought:
    """
    A single vertex in the Graph of Thoughts.

    Attributes:
        id:        Unique identifier.
        content:   Raw text produced by the LLM for this thought.
        score:     Quality score (higher is better).
        status:    Current lifecycle status.
        parent_ids: IDs of thoughts this thought was derived from.
        metadata:  Arbitrary extra information (phase, round, …).
    """
    id:         str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content:    str             = ""
    score:      float           = 0.0
    status:     ThoughtStatus   = ThoughtStatus.PENDING
    parent_ids: List[str]       = field(default_factory=list)
    metadata:   Dict[str, Any]  = field(default_factory=dict)

    def __repr__(self) -> str:
        snip = self.content[:60].replace("\n", " ")
        return (
            f"Thought(id={self.id}, score={self.score:.3f}, "
            f"status={self.status.name}, content='{snip}…')"
        )


class GraphReasoningState:
    """
    Dynamic store that tracks the full graph G = (V, E) during execution.

    Vertices are Thought objects; edges are encoded implicitly through
    each Thought's ``parent_ids`` list.
    """

    def __init__(self) -> None:
        self._thoughts: Dict[str, Thought] = {}

    # ── mutation ──────────────────────────────────────────────────────────

    def add(self, thought: Thought) -> None:
        """Insert a new thought vertex."""
        self._thoughts[thought.id] = thought

    def update_score(self, thought_id: str, score: float) -> None:
        """Assign / overwrite the score of an existing thought."""
        self._thoughts[thought_id].score  = score
        self._thoughts[thought_id].status = ThoughtStatus.SCORED

    def mark_kept(self, thought_id: str) -> None:
        self._thoughts[thought_id].status = ThoughtStatus.KEPT

    def mark_pruned(self, thought_id: str) -> None:
        self._thoughts[thought_id].status = ThoughtStatus.PRUNED

    def mark_refined(self, thought_id: str) -> None:
        self._thoughts[thought_id].status = ThoughtStatus.REFINED

    def mark_aggregated(self, thought_id: str) -> None:
        self._thoughts[thought_id].status = ThoughtStatus.AGGREGATED

    # ── queries ───────────────────────────────────────────────────────────

    def get(self, thought_id: str) -> Thought:
        return self._thoughts[thought_id]

    def all_thoughts(self) -> List[Thought]:
        return list(self._thoughts.values())

    def kept_thoughts(self) -> List[Thought]:
        return [t for t in self._thoughts.values()
                if t.status in (ThoughtStatus.KEPT, ThoughtStatus.SCORED)]

    def scored_thoughts(self) -> List[Thought]:
        return [t for t in self._thoughts.values()
                if t.status != ThoughtStatus.PENDING]

    def volume(self, thought_id: str) -> int:
        """
        Number of ancestor thoughts that could have contributed to this one
        (the 'volume' metric introduced in §2 of the paper).
        """
        visited: set = set()

        def _dfs(tid: str) -> None:
            if tid in visited:
                return
            visited.add(tid)
            for pid in self._thoughts[tid].parent_ids:
                if pid in self._thoughts:
                    _dfs(pid)

        _dfs(thought_id)
        visited.discard(thought_id)      # exclude the thought itself
        return len(visited)

    def to_dict(self) -> Dict[str, Any]:
        return {
            t.id: {
                "content":    t.content,
                "score":      t.score,
                "status":     t.status.name,
                "parent_ids": t.parent_ids,
                "metadata":   t.metadata,
                "volume":     self.volume(t.id),
            }
            for t in self._thoughts.values()
        }


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_GENERATE_PROMPT = """\
{system_prompt}

You are an expert problem-solver.  Your task is to reason step-by-step and \
produce a complete, precise solution to the problem below.

Problem:
{question}

Think carefully and write your full reasoning followed by your final answer.
Format your response as:
Reasoning: <your step-by-step reasoning>
Answer: <concise final answer>"""

_SCORE_PROMPT = """\
You are an impartial evaluator.  Rate the quality of the following solution \
on a scale from 0.0 (completely wrong) to 1.0 (perfect).

Problem:
{question}

Proposed solution:
{content}

Respond with ONLY a single decimal number between 0.0 and 1.0.
Score:"""

_AGGREGATE_PROMPT = """\
{system_prompt}

You are an expert problem-solver.  Below are {n} candidate solutions to the \
same problem.  Your task is to synthesize them into ONE definitive, complete, \
and accurate answer, combining the best insights from all candidates while \
eliminating errors or contradictions.

Problem:
{question}

Candidate solutions:
{candidates}

Write your synthesised reasoning and final answer.
Format your response as:
Reasoning: <your step-by-step reasoning>
Answer: <concise final answer>"""

_REFINE_PROMPT = """\
{system_prompt}

You are an expert problem-solver.  Review the solution below and improve it \
if needed.  Fix any errors, fill gaps in reasoning, and make the answer \
clearer and more precise.

Problem:
{question}

Current solution:
{content}

Write your improved reasoning and final answer.
Format your response as:
Reasoning: <your step-by-step reasoning>
Answer: <concise final answer>"""


# ─────────────────────────────────────────────────────────────────────────────
# GoT baseline
# ─────────────────────────────────────────────────────────────────────────────

class GoT(BaseBaseline):
    """
    Graph of Thoughts (GoT) prompting baseline.

    Execution schedule (Graph of Operations):

      Phase 0 – Generate:   produce ``num_branches`` independent reasoning paths.
      Phase 1 – Score:      ask the LLM to rate each path (0.0 – 1.0).
      Phase 2 – KeepBest:   retain only the top ``keep_best`` paths.
      Phase 3 – Aggregate:  merge the survivors into one combined answer.
      Phase 4 – Refine:     iteratively improve the aggregate (``refine_rounds`` times).

    Attributes:
        num_branches:    k – number of independent thoughts to generate (≥ 1).
        keep_best:       N – number of thoughts to keep after scoring (≥ 1).
        refine_rounds:   r – number of refine loops after aggregation (≥ 0).
        gen_temperature: Sampling temperature for Generate / Refine steps.
        score_temperature: Temperature for Scoring LLM calls.
        agg_temperature: Temperature for Aggregate step.

    Example:
        >>> llm = GeminiClient()
        >>> baseline = GoT(llm, num_branches=5, keep_best=2, refine_rounds=1)
        >>> response = baseline.run("Use 2, 3, 4, 8 to make 24.")
        >>> print(response.final_answer)
    """

    def __init__(
        self,
        llm: BaseLLM,
        num_branches:      int   = 5,
        keep_best:         int   = 2,
        refine_rounds:     int   = 1,
        gen_temperature:   float = 0.7,
        score_temperature: float = 0.0,
        agg_temperature:   float = 0.3,
    ) -> None:
        """
        Initialise the GoT baseline.

        Args:
            llm:               An instance of a BaseLLM subclass.
            num_branches:      Number of independent thoughts to generate (k).
            keep_best:         Number of top-scored thoughts to keep (N).
            refine_rounds:     Number of post-aggregation refinement loops (r).
            gen_temperature:   Temperature for the Generate & Refine LLM calls.
            score_temperature: Temperature for Scoring LLM calls.
            agg_temperature:   Temperature for the Aggregate LLM call.
        """
        super().__init__(llm, baseline_name="GoT")

        if num_branches < 1:
            raise ValueError("num_branches must be ≥ 1")
        if keep_best < 1:
            raise ValueError("keep_best must be ≥ 1")
        if refine_rounds < 0:
            raise ValueError("refine_rounds must be ≥ 0")

        self.num_branches      = num_branches
        self.keep_best         = min(keep_best, num_branches)
        self.refine_rounds     = refine_rounds
        self.gen_temperature   = gen_temperature
        self.score_temperature = score_temperature
        self.agg_temperature   = agg_temperature
        self.system_prompt     = ""

    # ─────────────────────────────────────────────────────────────────────
    # Prompter helpers (§3 – Prompter module)
    # ─────────────────────────────────────────────────────────────────────

    def _build_generate_prompt(self, question: str) -> str:
        return _GENERATE_PROMPT.format(system_prompt=self.system_prompt, question=question)

    def _build_score_prompt(self, question: str, content: str) -> str:
        return _SCORE_PROMPT.format(question=question, content=content)

    def _build_aggregate_prompt(
        self, question: str, thoughts: List[Thought]
    ) -> str:
        numbered = "\n\n".join(
            f"[Solution {i + 1}]\n{t.content}"
            for i, t in enumerate(thoughts)
        )
        return _AGGREGATE_PROMPT.format(
            system_prompt=self.system_prompt,
            n=len(thoughts),
            question=question,
            candidates=numbered,
        )

    def _build_refine_prompt(self, question: str, content: str) -> str:
        return _REFINE_PROMPT.format(system_prompt=self.system_prompt, question=question, content=content)

    # ─────────────────────────────────────────────────────────────────────
    # Parser helpers (§3 – Parser module)
    # ─────────────────────────────────────────────────────────────────────

    def _parse_answer(self, raw: str) -> Tuple[str, str]:
        """
        Extract (reasoning, answer) from a Generate / Aggregate / Refine reply.

        Falls back gracefully when the model omits the structured format.
        """
        reasoning_match = re.search(
            r"Reasoning\s*:\s*(.*?)(?=Answer\s*:|$)",
            raw, re.DOTALL | re.IGNORECASE,
        )
        answer_match = re.search(
            r"Answer\s*:\s*(.*?)$",
            raw, re.DOTALL | re.IGNORECASE,
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # fall back: use the last non-empty line
            lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
            answer = lines[-1] if lines else raw.strip()

        return reasoning, answer

    def _parse_score(self, raw: str) -> float:
        """
        Parse a floating-point score from the LLM reply.

        Normalises everything onto [0, 1]:
          - Explicit fraction "x/y" (e.g. "8.5/10") → x / y.
          - Bare value in [0, 1]                    → used as-is.
          - Bare value > 1 (e.g. "7", "8.5", "10")  → rescaled from a 0-10 scale (÷ 10).
          - No number found                         → defaults to 0.0 with a warning.
        The final result is always clamped to [0, 1].
        """
        # Prefer an explicit "x / y" rating (e.g. "8.5/10") when present.
        frac = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", raw)
        if frac and float(frac.group(2)) != 0.0:
            value = float(frac.group(1)) / float(frac.group(2))
            return max(0.0, min(1.0, value))

        numbers = re.findall(r"\d+(?:\.\d+)?", raw)
        if not numbers:
            logger.warning("Could not parse score from: %r – defaulting to 0.0", raw)
            return 0.0
        value = float(numbers[0])
        # Any value above 1 is assumed to be on a 0-10 scale (handles "7", "8.5", "3.0").
        if value > 1.0:
            value = value / 10.0
        return max(0.0, min(1.0, value))

    # ─────────────────────────────────────────────────────────────────────
    # Thought transformations (§2.2 of the paper)
    # ─────────────────────────────────────────────────────────────────────

    def _transform_generate(
        self,
        question: str,
        grs: GraphReasoningState,
        parent_ids: Optional[List[str]] = None,
    ) -> List[Thought]:
        """
        Generate k new thoughts (branching step).

        Each thought is produced by an independent LLM call so that
        temperature-driven diversity is maximised.
        """
        prompt    = self._build_generate_prompt(question)
        new_thoughts: List[Thought] = []

        for branch_idx in range(self.num_branches):
            raw      = self.call_llm(prompt, temperature=self.gen_temperature)
            reasoning, answer = self._parse_answer(raw.content)

            thought = Thought(
                content    = raw.content,
                parent_ids = list(parent_ids or []),
                metadata   = {
                    "phase":     "generate",
                    "branch":    branch_idx,
                    "reasoning": reasoning,
                    "answer":    answer,
                },
            )
            grs.add(thought)
            new_thoughts.append(thought)
            logger.debug("Generated thought %s (branch %d)", thought.id, branch_idx)

        return new_thoughts

    def _transform_score(
        self,
        question: str,
        thoughts: List[Thought],
        grs: GraphReasoningState,
    ) -> None:
        """
        Score every thought in *thoughts* by calling the LLM as a judge.

        Scores are stored back into the GRS (in-place mutation of the graph).
        """
        for thought in thoughts:
            prompt = self._build_score_prompt(question, thought.content)
            raw    = self.call_llm(prompt, temperature=self.score_temperature)
            score  = self._parse_score(raw.content)
            grs.update_score(thought.id, score)
            logger.debug("Scored thought %s → %.3f", thought.id, score)

    def _transform_keep_best(
        self,
        thoughts: List[Thought],
        grs: GraphReasoningState,
    ) -> List[Thought]:
        """
        Retain the top-N thoughts by score (KeepBest operation).

        Pruned thoughts remain in the GRS with status PRUNED so that the
        full graph history is preserved for the 'volume' metric.
        """
        ranked = sorted(thoughts, key=lambda t: t.score, reverse=True)
        kept   = ranked[: self.keep_best]
        pruned = ranked[self.keep_best :]

        for t in kept:
            grs.mark_kept(t.id)
        for t in pruned:
            grs.mark_pruned(t.id)

        logger.debug(
            "KeepBest: kept %d, pruned %d",
            len(kept), len(pruned),
        )
        return kept

    def _transform_aggregate(
        self,
        question: str,
        thoughts: List[Thought],
        grs: GraphReasoningState,
    ) -> Thought:
        """
        Merge multiple thoughts into a single aggregated thought.

        In graph terms this creates a new vertex with incoming edges from
        all *thoughts* — the core novelty of GoT over ToT (§1 of paper).
        """
        for t in thoughts:
            grs.mark_aggregated(t.id)

        prompt              = self._build_aggregate_prompt(question, thoughts)
        raw                 = self.call_llm(prompt, temperature=self.agg_temperature)
        reasoning, answer   = self._parse_answer(raw.content)

        agg_thought = Thought(
            content    = raw.content,
            parent_ids = [t.id for t in thoughts],
            metadata   = {
                "phase":     "aggregate",
                "reasoning": reasoning,
                "answer":    answer,
            },
        )
        grs.add(agg_thought)
        logger.debug(
            "Aggregated %d thoughts → new thought %s",
            len(thoughts), agg_thought.id,
        )
        return agg_thought

    def _transform_refine(
        self,
        question: str,
        thought: Thought,
        grs: GraphReasoningState,
    ) -> Thought:
        """
        Refine a single thought by looping it through the LLM.

        In graph terms this creates a self-loop edge (v → v') where v'
        inherits all connections of v (§2.2 of the paper).
        """
        prompt            = self._build_refine_prompt(question, thought.content)
        raw               = self.call_llm(prompt, temperature=self.gen_temperature)
        reasoning, answer = self._parse_answer(raw.content)

        refined = Thought(
            content    = raw.content,
            parent_ids = [thought.id],
            metadata   = {
                "phase":     "refine",
                "reasoning": reasoning,
                "answer":    answer,
            },
        )
        grs.mark_refined(thought.id)
        grs.add(refined)
        logger.debug("Refined thought %s → new thought %s", thought.id, refined.id)
        return refined

    # ─────────────────────────────────────────────────────────────────────
    # Controller – Graph of Operations executor (§3)
    # ─────────────────────────────────────────────────────────────────────

    def _execute_graph_of_operations(
        self,
        question: str,
        grs: GraphReasoningState,
    ) -> Thought:
        """
        Execute the static GoO (Graph of Operations) defined for this baseline:

            Phase 0 – Generate  (k independent thoughts)
            Phase 1 – Score     (LLM-as-judge for every thought)
            Phase 2 – KeepBest  (retain top N)
            Phase 3 – Aggregate (merge survivors into one)
            Phase 4 – Refine    (r improvement loops)

        Returns the final Thought that carries the definitive answer.
        """
        # ── Phase 0: Generate ────────────────────────────────────────────
        logger.info("[GoT] Phase 0 – Generate (%d branches)", self.num_branches)
        generated = self._transform_generate(question, grs)

        # ── Phase 1: Score ───────────────────────────────────────────────
        logger.info("[GoT] Phase 1 – Score")
        self._transform_score(question, generated, grs)

        # ── Phase 2: KeepBest ────────────────────────────────────────────
        logger.info("[GoT] Phase 2 – KeepBest (N=%d)", self.keep_best)
        survivors = self._transform_keep_best(generated, grs)

        # ── Phase 3: Aggregate ───────────────────────────────────────────
        if len(survivors) == 1:
            # No aggregation needed if only one thought survived
            aggregate = survivors[0]
            logger.info("[GoT] Phase 3 – skipping Aggregate (only 1 survivor)")
        else:
            logger.info("[GoT] Phase 3 – Aggregate (%d → 1)", len(survivors))
            aggregate = self._transform_aggregate(question, survivors, grs)

        # ── Phase 4: Refine ──────────────────────────────────────────────
        current = aggregate
        for round_idx in range(self.refine_rounds):
            logger.info(
                "[GoT] Phase 4 – Refine (round %d / %d)",
                round_idx + 1, self.refine_rounds,
            )
            current = self._transform_refine(question, current, grs)

        return current

    # ─────────────────────────────────────────────────────────────────────
    # Public API – BaseBaseline.run()
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction:   Optional[str] = None,
        temperature:   float         = 0.0,
        **kwargs,
    ) -> BaselineResponse:
        """
        Execute the GoT pipeline on *question* and return a BaselineResponse.

        Args:
            question:      The input problem to solve.
            system_prompt: System-level instruction provided to the model.
            instruction:   Optional extra context prepended to the question.
            temperature:   If non-zero, overrides ``gen_temperature``.
            **kwargs:      Forwarded but otherwise ignored.

        Returns:
            BaselineResponse with the final answer, full reasoning trace,
            and GoT-specific metadata (graph snapshot, volume, …).
        """
        self.reset_counters()

        # Store system_prompt for use in all prompt templates
        self.system_prompt = system_prompt or ""

        # Allow caller-side temperature override without mutating instance state.
        _orig_gen_temp = self.gen_temperature
        if temperature != 0.0:
            self.gen_temperature = temperature

        # Prepend any extra instruction to the question text
        full_question = question
        if instruction:
            full_question = f"{instruction}\n\n{question}"

        # Initialise the Graph Reasoning State (dynamic graph G)
        grs = GraphReasoningState()

        # Run the static Graph of Operations (GoO)
        try:
            final_thought = self._execute_graph_of_operations(full_question, grs)
        finally:
            self.gen_temperature = _orig_gen_temp

        # Extract structured answer from the final thought
        _, final_answer = self._parse_answer(final_thought.content)

        # Build a human-readable reasoning trace from all intermediate steps
        trace_parts: List[str] = []
        for thought in grs.all_thoughts():
            phase  = thought.metadata.get("phase", "?")
            branch = thought.metadata.get("branch", "")
            label  = f"[{phase}" + (f" branch={branch}" if branch != "" else "") + f"] (id={thought.id}, score={thought.score:.3f}, vol={grs.volume(thought.id)})"
            trace_parts.append(f"{label}\n{thought.content}")
        reasoning_trace = "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(trace_parts)

        # Collect intermediate step summaries (one per thought)
        intermediate: List[str] = [
            f"{t.metadata.get('phase','?')} | id={t.id} | score={t.score:.3f} | {t.content[:120]}…"
            for t in grs.all_thoughts()
        ]

        # Compute global volume of the final thought (paper's §5 metric)
        final_volume = grs.volume(final_thought.id)

        return self.create_response(
            final_answer       = final_answer,
            reasoning_trace    = reasoning_trace,
            intermediate_steps = intermediate,
            metadata           = {
                "got_params": {
                    "num_branches":  self.num_branches,
                    "keep_best":     self.keep_best,
                    "refine_rounds": self.refine_rounds,
                },
                "final_thought_id":     final_thought.id,
                "final_thought_volume": final_volume,
                "graph_snapshot":       grs.to_dict(),
                "total_thoughts":       len(grs.all_thoughts()),
            },
        )