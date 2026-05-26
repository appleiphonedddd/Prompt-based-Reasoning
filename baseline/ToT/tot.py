"""
Tree of Thoughts (ToT) Prompting Implementation.

ToT generalises Chain-of-Thought by maintaining a *tree* of intermediate
reasoning steps ("thoughts").  At each node the LM proposes k candidate
next thoughts, evaluates every candidate, and a search algorithm (BFS or
DFS) decides which branches to keep exploring.

This implementation targets **Game of 24** (the primary benchmark in the
paper) but the class is designed to be task-agnostic: the prompt strings
and thought/value parsing can be overridden for other tasks.

Reference:
    Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large
    Language Models", NeurIPS 2023.  https://arxiv.org/abs/2305.10601

Author: (your name)
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from baseline.basebaseline import BaseBaseline, BaselineResponse
from models.base import BaseLLM

# ─────────────────────────────────────────────
#  Prompt templates (Game of 24)
# ─────────────────────────────────────────────

# Few-shot propose prompt: given remaining numbers, list plausible next steps.
PROPOSE_PROMPT = """Let's play Game of 24. Use basic arithmetic (+, -, *, /) on \
the given numbers to reach 24.  Each step, pick two numbers, apply one operation, \
and write the result (e.g. "6 - 1 = 5 (left: 3 5 8)").
Important: ONLY use the numbers listed in Input below. Do NOT introduce any other numbers.

Input: {state}
Possible next steps:
"""

# Few-shot value prompt: judge whether remaining numbers can still reach 24.
VALUE_PROMPT = """Evaluate whether the following remaining numbers can reach 24 \
using basic arithmetic.  Reply with one of: sure / likely / impossible.

Remaining: {state}
Evaluation:"""

# Final answer prompt: extract a complete equation from the solved state.
FINAL_ANSWER_PROMPT = """You are solving the Game of 24.
The numbers {state} were given.  Here are the intermediate steps taken:
{steps}
Write a single arithmetic expression using each of the original numbers exactly once \
that equals 24.  Reply with the expression only, no "= 24" suffix \
(e.g. "(10 - 4) * (9 - 5)").
Expression:"""

# ─────────────────────────────────────────────
#  Plan-level ToT prompts (generic, non-Game-of-24 tasks)
#
#  Follows the generic zero-shot ToT-BFS the ToT authors describe in
#  Appendix B.1 of the paper: sample k complete strategies → vote for the
#  best → sample k complete solutions following it → vote for the best.
#  The thought unit is a *whole* strategy/solution, which keeps the reasoning
#  coherent — unlike a per-fragment tree whose pieces are selected by a
#  value function that, for free-form tasks, cannot discriminate.
# ─────────────────────────────────────────────

STRATEGY_PROMPT = """{task_ctx}

Problem:
{question}

Propose ONE concise high-level strategy for solving this problem: outline the \
key steps or approach you would take. Do NOT carry out the full computation or \
give the final answer yet — describe only the plan."""

SOLUTION_PROMPT = """{task_ctx}

Problem:
{question}

Strategy to follow:
{strategy}

Now solve the problem step by step, following the strategy above. Show your \
work, then state the final answer clearly on the last line."""

VOTE_PROMPT = """{task_ctx}

Problem:
{question}

Below are several candidate {kind}s for solving the problem. Decide which one \
is most promising for reaching the correct answer.

{listing}

Briefly analyze each choice, then conclude on the LAST line with exactly:
"The best choice is X" — where X is the integer id of the best choice."""

FINALIZE_PROMPT = """{task_ctx}

Problem:
{question}

Worked solution:
{solution}

Based on the worked solution above, output ONLY the final answer — no \
explanation, no restating of the question."""

# Matches a Game of 24 input: exactly four space-separated positive integers.
_GAME_OF_24_RE = re.compile(r"^\d+(?:\s+\d+){3}$")


# ─────────────────────────────────────────────
#  Value categories
# ─────────────────────────────────────────────

class Value(Enum):
    SURE       = 3
    LIKELY     = 2
    IMPOSSIBLE = 0

    @staticmethod
    def from_text(text: str) -> "Value":
        t = text.strip().lower()
        if "sure"       in t: return Value.SURE
        if "likely"     in t: return Value.LIKELY
        if "impossible" in t: return Value.IMPOSSIBLE
        return Value.LIKELY          # default to "likely" on ambiguity


# ─────────────────────────────────────────────
#  Tree node
# ─────────────────────────────────────────────

@dataclass
class ThoughtNode:
    """One node in the Tree of Thoughts.

    Attributes:
        state:       Remaining numbers at this node (e.g. "4 6 10").
        thought:     The equation step that produced this node.
        depth:       Distance from the root (0 = root).
        parent:      Reference to the parent node (None for root).
        value_score: Numeric score assigned by the state evaluator.
        children:    Child nodes generated from this node.
    """

    state:       str
    thought:     str          = ""
    depth:       int          = 0
    parent:      Optional["ThoughtNode"] = field(default=None, repr=False)
    value_score: float        = 0.0
    children:    List["ThoughtNode"] = field(default_factory=list, repr=False)

    # ── helpers ──────────────────────────────

    def path_thoughts(self) -> List[str]:
        """Collect all thoughts from root to this node (oldest first)."""
        node, path = self, []
        while node is not None:
            if node.thought:
                path.append(node.thought)
            node = node.parent
        return list(reversed(path))

    def is_terminal(self, max_depth: int) -> bool:
        return self.depth >= max_depth


# ─────────────────────────────────────────────
#  ToT baseline
# ─────────────────────────────────────────────

class ToT(BaseBaseline):
    """Tree of Thoughts (ToT) prompting baseline.

    Implements the BFS and DFS variants described in Algorithm 1 & 2 of
    the paper.  Default hyper-parameters reproduce the Game-of-24 setup
    (BFS, breadth=5, 3 thought steps, 3 value samples per thought).

    Attributes:
        n_generate_sample:    k — number of candidate thoughts per node.
        n_evaluate_sample:    Number of times to sample the value prompt
                              (majority vote gives a more robust estimate).
        breadth_limit:        b — BFS keeps the top-b states per level.
        max_steps:            T — maximum tree depth (= number of thought steps).
        search_algorithm:     "bfs" (default) or "dfs".
        value_threshold:      DFS prunes states whose score falls below this.
        propose_temperature:  Sampling temperature for thought generation.
        value_temperature:    Sampling temperature for state evaluation.
        propose_prompt:       Prompt template for thought generation.
        value_prompt:         Prompt template for state evaluation.
        final_answer_prompt:  Prompt template for extracting the final answer.

    Example:
        >>> llm = GeminiClient()
        >>> baseline = ToT(llm, breadth_limit=5, n_generate_sample=5)
        >>> response = baseline.run("4 9 10 13")
        >>> print(response.final_answer)
        "(13 - 9) * (10 - 4) = 24"
    """

    def __init__(
        self,
        llm: BaseLLM,
        n_generate_sample:   int   = 5,
        n_evaluate_sample:   int   = 3,
        breadth_limit:       int   = 5,
        max_steps:           int   = 3,
        search_algorithm:    str   = "bfs",
        value_threshold:     float = 1.0,
        propose_temperature: float = 0.7,
        value_temperature:   float = 0.0,
        propose_prompt:      Optional[str] = None,
        value_prompt:        Optional[str] = None,
        final_answer_prompt: Optional[str] = None,
    ) -> None:
        """Initialise the ToT baseline.

        Args:
            llm:                  Any BaseLLM subclass.
            n_generate_sample:    Candidate thoughts to propose at each node (k).
            n_evaluate_sample:    Value-prompt samples per thought (majority vote).
            breadth_limit:        BFS: states kept per level (b).
            max_steps:            Maximum tree depth / thought steps (T).
            search_algorithm:     "bfs" or "dfs".
            value_threshold:      DFS: prune states whose score < threshold.
            propose_temperature:  Temperature for thought-generation calls.
            value_temperature:    Temperature for state-evaluation calls.
            propose_prompt:       Override default propose-prompt template.
            value_prompt:         Override default value-prompt template.
            final_answer_prompt:  Override default answer-extraction template.
        """
        super().__init__(llm, baseline_name="ToT")

        self.n_generate_sample   = n_generate_sample
        self.n_evaluate_sample   = n_evaluate_sample
        self.breadth_limit       = breadth_limit
        self.max_steps           = max_steps
        self.search_algorithm    = search_algorithm.lower()
        self.value_threshold     = value_threshold
        self.propose_temperature = propose_temperature
        self.value_temperature   = value_temperature

        self.propose_prompt     = propose_prompt or PROPOSE_PROMPT
        self.value_prompt       = value_prompt   or VALUE_PROMPT
        self.final_answer_prompt = final_answer_prompt or FINAL_ANSWER_PROMPT

    # ══════════════════════════════════════════
    #  Thought Generator  G(pθ, s, k)
    # ══════════════════════════════════════════

    def generate_thoughts(self, node: ThoughtNode) -> List[str]:
        """Propose k candidate next thoughts from the current state.

        Uses a single "propose prompt" call that asks the LM to list
        several plausible next steps (sequential proposal strategy from
        the paper, suited for constrained spaces like Game of 24).

        Args:
            node: Current tree node whose state we expand.

        Returns:
            List of raw thought strings (one per candidate).
        """
        task_ctx = getattr(self, "_task_context", None)
        if task_ctx:
            if node.depth == 0:
                # Root node: state is the original question; ask for first steps.
                prompt = (
                    task_ctx + "\n\n"
                    "Problem:\n" + node.state + "\n\n"
                    "Propose " + str(self.n_generate_sample) + " distinct candidate "
                    "FIRST reasoning steps toward solving this problem.\n"
                    "Each step should be on its own line. Be specific and concrete."
                )
            else:
                # Deeper node: state is the accumulated reasoning so far.
                prompt = (
                    task_ctx + "\n\n"
                    "Reasoning so far:\n" + node.state + "\n\n"
                    "Propose " + str(self.n_generate_sample) + " distinct candidate "
                    "NEXT reasoning steps that continue from the above.\n"
                    "Each step should be on its own line. Be specific and concrete."
                )
        else:
            prompt = self.propose_prompt.format(state=node.state)
        response = self.call_llm(prompt, temperature=self.propose_temperature)
        thoughts = self.parse_thoughts(response.content)
        # Return up to k candidates (pad with empty strings if fewer returned)
        return (thoughts + [""] * self.n_generate_sample)[: self.n_generate_sample]

    def parse_thoughts(self, text: str) -> List[str]:
        """Extract individual thought steps from the LM's raw output.

        For Game of 24 (numeric state), only lines containing digits are
        kept.  For other tasks the full non-empty line set is returned so
        that textual reasoning steps are not silently discarded.

        Args:
            text: Raw LM output containing proposed steps.

        Returns:
            List of non-empty thought strings.
        """
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        is_game_of_24 = not getattr(self, "_task_context", None)
        if is_game_of_24:
            lines = [ln for ln in lines if re.search(r"\d", ln)]
        return lines

    def extract_remaining(self, thought: str, current_state: str) -> str:
        """Extract the state that follows from applying a thought step.

        For Game of 24 looks for structured "(left: x y z)" or "[state: ...]"
        markers.  For all other tasks, the new state is the *accumulated*
        reasoning so far: current_state appended with the new thought.  This
        ensures that each deeper tree level builds on previous steps rather
        than re-starting from the original question.

        Args:
            thought:       The thought string produced by the LM.
            current_state: The state of the parent node.

        Returns:
            A string representing the updated state after this thought.
        """
        # The "(left: …)" and "= 24" markers are Game-of-24-specific terminal
        # heuristics and MUST stay scoped to Game of 24.  Leaking them into
        # other numeric tasks (e.g. multistep arithmetic) made any step that
        # merely mentioned "= 24" get mis-declared the solution, derailing the
        # search and inflating/deflating accuracy by luck.
        is_game_of_24 = getattr(self, "_task_context", None) is None
        if is_game_of_24:
            # Pattern: "(left: 4 4 10)" or "left: 4 4 10"
            match = re.search(r"left[:\s]+([0-9 ]+)", thought, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            # Terminal: "= 24" means Game of 24 is solved
            if re.search(r"=\s*24\b", thought):
                return "24"
            return current_state

        # Non-Game-of-24 (legacy fragment path): accumulate reasoning so the
        # next level receives context instead of the original question again.
        match = re.search(r"\[state:\s*([^\]]+)\]", thought, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return current_state + "\n" + thought

    # ══════════════════════════════════════════
    #  State Evaluator  V(pθ, S)
    # ══════════════════════════════════════════

    def evaluate_state(self, node: ThoughtNode) -> float:
        """Score a state via majority-vote across n_evaluate_sample calls.

        Each call asks the LM to rate the remaining numbers as
        "sure / likely / impossible" to reach 24.  The average numeric
        value (Value enum) across samples is returned.

        Args:
            node: The node whose state is to be evaluated.

        Returns:
            A numeric score (higher = more promising).
        """
        if node.state == "24":
            # Terminal success: highest possible score
            return float(Value.SURE.value)

        task_ctx = getattr(self, "_task_context", None)
        if task_ctx:
            prompt = (
                task_ctx + "\n\n"
                "Current reasoning state:\n" + node.state + "\n\n"
                "Is this state on track toward a valid final answer for the problem above?\n"
                "Reply with exactly one word: sure / likely / impossible"
            )
        else:
            prompt = self.value_prompt.format(state=node.state)
        scores: List[float] = []
        for _ in range(self.n_evaluate_sample):
            response = self.call_llm(prompt, temperature=self.value_temperature)
            scores.append(Value.from_text(response.content).value)

        return sum(scores) / len(scores) if scores else 0.0

    # ══════════════════════════════════════════
    #  Search Algorithms
    # ══════════════════════════════════════════

    # ── BFS (Algorithm 1) ───────────────────

    def bfs(self, root: ThoughtNode) -> Tuple[Optional[ThoughtNode], List[str]]:
        """Breadth-First Search over the thought tree.

        Maintains the b most-promising frontier states at each depth level.

        Args:
            root: Root node (initial numbers).

        Returns:
            (best_terminal_node, intermediate_step_log)
        """
        frontier: List[ThoughtNode] = [root]
        log:      List[str]         = []
        best_terminal: Optional[ThoughtNode] = None

        for step in range(1, self.max_steps + 1):
            # ── Expand every frontier node ──
            candidates: List[ThoughtNode] = []
            for parent in frontier:
                thoughts = self.generate_thoughts(parent)
                log.append(f"[BFS step {step}] Node '{parent.state}' → {len(thoughts)} thoughts")

                for thought in thoughts:
                    if not thought:
                        continue
                    new_state = self.extract_remaining(thought, parent.state)
                    child = ThoughtNode(
                        state=new_state,
                        thought=thought,
                        depth=step,
                        parent=parent,
                    )
                    parent.children.append(child)
                    candidates.append(child)

            if not candidates:
                log.append(f"[BFS step {step}] No candidates generated — stopping early.")
                break

            # ── Evaluate all candidates ──
            for node in candidates:
                node.value_score = self.evaluate_state(node)
                log.append(f"  Thought: '{node.thought[:60]}...' | Score: {node.value_score:.2f}")

            # ── Check for solved states ──
            # Compare numeric scores directly to avoid ValueError from Value(int(avg))
            # when the average score falls on an unmapped integer (e.g. 1).
            sure_val = float(Value.SURE.value)
            solved = [
                n for n in candidates
                if n.state == "24"
                or (n.value_score >= sure_val and step == self.max_steps)
            ]
            if solved:
                best_terminal = max(solved, key=lambda n: n.value_score)
                log.append(f"[BFS step {step}] ✓ Solution node found.")
                break

            # ── Prune to top-b frontier ──
            candidates.sort(key=lambda n: n.value_score, reverse=True)
            frontier = candidates[: self.breadth_limit]
            log.append(f"[BFS step {step}] Keeping top-{self.breadth_limit} states.")

        # If no explicit solution node found, pick the best leaf reached
        if best_terminal is None and frontier:
            best_terminal = max(frontier, key=lambda n: n.value_score)

        return best_terminal, log

    # ── DFS (Algorithm 2) ───────────────────

    def dfs(self, root: ThoughtNode) -> Tuple[Optional[ThoughtNode], List[str]]:
        """Depth-First Search with pruning over the thought tree.

        Explores the most promising branch first; backtracks when the state
        evaluator deems a node impossible (score < value_threshold).

        Args:
            root: Root node.

        Returns:
            (best_terminal_node, intermediate_step_log)
        """
        log:   List[str]                   = []
        stack: List[ThoughtNode]           = [root]
        best_terminal: Optional[ThoughtNode] = None

        while stack:
            node = stack.pop()

            # ── Terminal check ──
            if node.is_terminal(self.max_steps) or node.state == "24":
                log.append(f"[DFS] Terminal node reached: '{node.state}'")
                if best_terminal is None or node.value_score > best_terminal.value_score:
                    best_terminal = node
                continue

            # ── Pruning check ──
            if node.depth > 0 and node.value_score < self.value_threshold:
                log.append(f"[DFS] Pruning node '{node.state}' (score {node.value_score:.2f})")
                continue

            # ── Expand: generate & evaluate thoughts ──
            thoughts = self.generate_thoughts(node)
            log.append(f"[DFS depth {node.depth}] '{node.state}' → {len(thoughts)} thoughts")

            children: List[ThoughtNode] = []
            for thought in thoughts:
                if not thought:
                    continue
                new_state = self.extract_remaining(thought, node.state)
                child = ThoughtNode(
                    state=new_state,
                    thought=thought,
                    depth=node.depth + 1,
                    parent=node,
                )
                child.value_score = self.evaluate_state(child)
                node.children.append(child)
                children.append(child)
                log.append(f"  Thought: '{thought[:60]}' | Score: {child.value_score:.2f}")

            # Push highest-scoring children last so they're popped first
            children.sort(key=lambda n: n.value_score)
            stack.extend(children)

        return best_terminal, log

    # ══════════════════════════════════════════
    #  Answer extraction
    # ══════════════════════════════════════════

    def extract_final_answer(
        self,
        root: ThoughtNode,
        best_node: Optional[ThoughtNode],
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """Ask the LM to write a clean final equation.

        Provides the original numbers and the discovered reasoning steps
        as context.

        Args:
            root:          Root node (holds the original numbers).
            best_node:     The best terminal or leaf node found by search.
            question:      Original question string (= initial numbers).
            system_prompt: Optional benchmark system prompt (enforces output format).
            instruction:   Optional benchmark instruction (enforces number usage).

        Returns:
            The final equation string, or the best thought if extraction fails.
        """
        if best_node is None:
            return "No solution found."

        steps = "\n".join(best_node.path_thoughts()) or "No steps."

        # For non-Game of 24 tasks, build a generic final-answer prompt that
        # embeds the task context directly instead of the Game of 24 template.
        task_ctx: Optional[str] = None
        if (system_prompt or instruction) and self.final_answer_prompt is FINAL_ANSWER_PROMPT:
            if not _GAME_OF_24_RE.match(question.strip()):
                task_ctx = "\n\n".join(p for p in [system_prompt, instruction] if p)

        if task_ctx:
            prompt = (
                task_ctx + "\n\n"
                "Original problem:\n" + question + "\n\n"
                "Reasoning path:\n" + steps + "\n\n"
                "Provide ONLY the final answer. No explanations or extra text."
            )
        else:
            prompt = self.final_answer_prompt.format(state=question, steps=steps)
            # Prepend benchmark context for Game of 24
            prefix_parts = []
            if system_prompt:
                prefix_parts.append(system_prompt)
            if instruction:
                prefix_parts.append(instruction)
            if prefix_parts:
                prompt = "\n\n".join(prefix_parts) + "\n\n" + prompt

        response = self.call_llm(prompt, temperature=0.0)
        answer = response.content.strip()

        # Sanitise: take the first non-empty line
        for line in answer.splitlines():
            line = line.strip()
            if line:
                return line
        return answer or "No solution found."

    # ══════════════════════════════════════════
    #  Plan-level ToT  (generic non-Game-of-24 tasks)
    # ══════════════════════════════════════════

    def _sample_n(self, prompt: str, k: int, temperature: float) -> List[str]:
        """Sample k i.i.d. completions for a prompt; keep non-empty texts."""
        out: List[str] = []
        for _ in range(k):
            resp = self.call_llm(prompt, temperature=temperature)
            text = resp.content.strip()
            if text:
                out.append(text)
        return out

    @staticmethod
    def _parse_vote(text: str, n: int) -> Optional[int]:
        """Parse a 0-based winning index from a vote response.

        Looks for "the best choice is X"; otherwise falls back to the last
        integer that is in range.  Returns None if nothing usable is found.
        """
        ids = re.findall(r"best\s+choice\s+is\s*\**\s*#?\s*(\d+)", text, re.IGNORECASE)
        if not ids:
            ids = re.findall(r"\b(\d+)\b", text)
        for raw in reversed(ids):
            idx = int(raw) - 1
            if 0 <= idx < n:
                return idx
        return None

    def _vote_best(
        self, task_ctx: str, question: str,
        candidates: List[str], kind: str, log: List[str],
    ) -> int:
        """Majority-vote across n_evaluate_sample calls for the best candidate.

        Unlike the per-state sure/likely/impossible evaluator (which cannot
        discriminate between free-form reasoning steps), voting *compares*
        whole candidates against each other and yields a real choice signal.
        """
        if len(candidates) <= 1:
            return 0
        listing = "\n\n".join(
            f"Choice {i + 1}:\n{c}" for i, c in enumerate(candidates)
        )
        prompt = VOTE_PROMPT.format(
            task_ctx=task_ctx, question=question, kind=kind, listing=listing,
        )
        votes: List[int] = []
        for _ in range(self.n_evaluate_sample):
            resp = self.call_llm(prompt, temperature=self.value_temperature)
            idx = self._parse_vote(resp.content, len(candidates))
            if idx is not None:
                votes.append(idx)
        best_idx = max(set(votes), key=votes.count) if votes else 0
        log.append(
            f"[plan-ToT] {kind} votes={[v + 1 for v in votes]} → choice {best_idx + 1}"
        )
        return best_idx

    def _finalize_answer(self, question: str, solution: str, task_ctx: str) -> str:
        """Ask the LM to emit only the final answer from a worked solution."""
        prompt = FINALIZE_PROMPT.format(
            task_ctx=task_ctx, question=question, solution=solution,
        )
        resp = self.call_llm(prompt, temperature=0.0)
        answer = resp.content.strip()
        for line in answer.splitlines():
            line = line.strip()
            if line:
                return line
        return answer or "No solution found."

    def _plan_level_solve(
        self, question: str, task_ctx: str, propose_temp: float,
    ) -> Tuple[str, str, str, List[str]]:
        """Generic plan-level ToT-BFS for non-Game-of-24 tasks.

        Two voting levels over *complete* candidates:
          1. sample k strategies      → vote best strategy
          2. sample k full solutions  → vote best solution
        then extract a clean final answer from the winning solution.

        Returns:
            (final_answer, best_strategy, best_solution, log)
        """
        log: List[str] = []
        k = max(1, self.n_generate_sample)

        # ── Level 1: strategies ──
        strategies = self._sample_n(
            STRATEGY_PROMPT.format(task_ctx=task_ctx, question=question),
            k, propose_temp,
        )
        log.append(f"[plan-ToT] sampled {len(strategies)} strategy candidates")
        best_strategy = (
            strategies[self._vote_best(task_ctx, question, strategies, "strategy", log)]
            if strategies else ""
        )

        # ── Level 2: full solutions following the chosen strategy ──
        solutions = self._sample_n(
            SOLUTION_PROMPT.format(
                task_ctx=task_ctx, question=question, strategy=best_strategy,
            ),
            k, propose_temp,
        )
        log.append(f"[plan-ToT] sampled {len(solutions)} solution candidates")
        best_solution = (
            solutions[self._vote_best(task_ctx, question, solutions, "solution", log)]
            if solutions else best_strategy
        )

        # ── Finalize: clean answer extraction from the winning solution ──
        final_answer = self._finalize_answer(question, best_solution, task_ctx)
        return final_answer, best_strategy, best_solution, log

    # ══════════════════════════════════════════
    #  Public interface
    # ══════════════════════════════════════════

    def run(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> BaselineResponse:
        """Execute the full ToT pipeline on the given problem.

        Args:
            question:      The input problem (e.g. "4 9 10 13" for Game of 24).
            system_prompt: Kept for interface compatibility; not used internally.
            instruction:   Kept for interface compatibility; not used internally.
            temperature:   If non-zero, overrides propose_temperature.
            **kwargs:      Additional arguments (ignored; for interface compat).

        Returns:
            BaselineResponse with the final answer, reasoning trace, and metrics.
        """
        self.reset_counters()

        # Use a local variable so self.propose_temperature is never mutated.
        propose_temp = temperature if temperature > 0.0 else self.propose_temperature

        # Decide the task family.  Game of 24 (four bare integers, default
        # prompts) uses the fragment-tree search; every other task uses the
        # generic plan-level ToT, whose voting comparison gives a real search
        # signal where the per-state sure/likely/impossible evaluator cannot.
        is_non_game_of_24 = (
            bool(system_prompt or instruction)
            and self.propose_prompt is PROPOSE_PROMPT
            and not _GAME_OF_24_RE.match(question.strip())
        )

        # ── Plan-level ToT path (generic tasks) ──
        if is_non_game_of_24:
            task_ctx = "\n\n".join(p for p in [system_prompt, instruction] if p)
            final_answer, best_strategy, best_solution, search_log = (
                self._plan_level_solve(question, task_ctx, propose_temp)
            )
            reasoning_trace = (
                "[ToT — plan-level BFS]\n"
                f"  Strategy: {best_strategy}\n"
                f"  Solution: {best_solution}"
            )
            return self.create_response(
                final_answer=final_answer,
                reasoning_trace=reasoning_trace,
                intermediate_steps=search_log,
                metadata={
                    "search_algorithm":    "plan-level",
                    "n_generate_sample":   self.n_generate_sample,
                    "n_evaluate_sample":   self.n_evaluate_sample,
                    "breadth_limit":       self.breadth_limit,
                    "max_steps":           self.max_steps,
                    "value_threshold":     self.value_threshold,
                    "propose_temperature": propose_temp,
                    "value_temperature":   self.value_temperature,
                    "best_path":           [best_strategy, best_solution],
                    "best_node_state":     None,
                    "best_node_score":     None,
                    "best_strategy":       best_strategy,
                    "best_solution":       best_solution,
                },
            )

        # ── Game-of-24 fragment-tree path ──
        # _task_context = None keeps generate_thoughts()/evaluate_state()/
        # extract_remaining() on their Game-of-24 branches.
        self._task_context = None
        root = ThoughtNode(state=question.strip(), depth=0)

        # Temporarily swap temperature so generate_thoughts picks it up,
        # then restore the original value afterwards.
        _orig_propose_temp = self.propose_temperature
        self.propose_temperature = propose_temp
        try:
            if self.search_algorithm == "dfs":
                best_node, search_log = self.dfs(root)
            else:
                best_node, search_log = self.bfs(root)
        finally:
            self.propose_temperature = _orig_propose_temp

        # ── Answer extraction ────────────────
        final_answer = self.extract_final_answer(
            root, best_node, question,
            system_prompt=system_prompt,
            instruction=instruction,
        )

        # ── Build reasoning trace ────────────
        best_path = best_node.path_thoughts() if best_node else []
        reasoning_trace = (
            f"[ToT — {self.search_algorithm.upper()}]\n"
            + "\n".join(f"  Step {i+1}: {t}" for i, t in enumerate(best_path))
        )

        return self.create_response(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            intermediate_steps=search_log,
            metadata={
                "search_algorithm":    self.search_algorithm,
                "n_generate_sample":   self.n_generate_sample,
                "n_evaluate_sample":   self.n_evaluate_sample,
                "breadth_limit":       self.breadth_limit,
                "max_steps":           self.max_steps,
                "value_threshold":     self.value_threshold,
                "propose_temperature": self.propose_temperature,
                "value_temperature":   self.value_temperature,
                "best_path":           best_path,
                "best_node_state":     best_node.state if best_node else None,
                "best_node_score":     best_node.value_score if best_node else None,
            },
        )
