"""
Reversal of Thought (RoT) Prompting Implementation.

RoT enhances LLM reasoning through a four-stage pipeline:
1. Reverse Reasoning Warm-up: Generate multiple candidate task definitions
   by asking the LLM to reverse-engineer the task from demonstrations.
2. Pairwise Preference Selection: Use the LLM as a judge to compare
   candidates pairwise, then apply transitive closure to strengthen
   preference scores and select the optimal candidate.
3. Cognitive Preference Manager (CPM): Assess knowledge boundaries via
   embedding similarity, then aggregate the original prompt with the
   LLM-taste prompt using task-appropriate strategies (solution logic
   for known tasks, stylistic template for unknown tasks).
4. Instantiation: Apply the CPM-refined prompt to solve the actual
   input question with structured reasoning.

Reference:
- Yuan, J., Du, D., Zhang, H., Di, Z., & Naseem, U. (2025).
  "Reversal of Thought: Enhancing Large Language Models with
  Preference-Guided Reverse Reasoning Warm-up."
  Proceedings of ACL 2025 (Main), pp. 19442-19459.

Author: Egor Morozov
"""

import re
import math
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple

from baseline.basebaseline import BaseBaseline, BaselineResponse
from models.base import BaseLLM

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Prompt templates (adapted from the official RoT release)
# ─────────────────────────────────────────────────────────

REVERSAL_OF_THOUGHT_PROMPT = """###Instruction###
You are a highly distinguished expert in mathematics and information reasoning. \
Based on the given example, define the specific task, including the task definition, \
pseudocode, logical pseudocode, case examples, and input-output format.
 1. Understand Task Description:
 Meticulously study demonstrations to deeply understand generic task description.
 2. Plan Generic Pseudocode:
 Provide pseudocode in text form and plan an efficient algorithm to complete the task with your experiences.
 3. Formulate Logical Pseudocode:
 Convert the pseudocode into generic logical algorithm pseudocode using ONLY logical symbols:
 Logical Operators:
 Conjunction: A ∧ B ; Disjunction: A ∨ B
 equivalence: A ≡ B , Negation: ¬A
 Quantifiers:
 Universal quantifier: ∀x ; Existential quantifier: ∃x
 Inequalities:
 Less than: x < y ; Greater than: x > y 
 Less than or equal to: x ≤ y
 Greater than or equal to: x ≥ y
 Equals: x = y ; Not equals: x ≠ y
 Conditional Statements:
 If A then B: A ⊃ B
 If A ∧B then C: (A∧B) ⊃ C
 If A ∨B then C: (A∨B) ⊃ C
 If ∀x(P(x)) then Q: ∀x(P(x)) ⊃ Q
 If ∃x(P(x)) then Q: ∃x(P(x)) ⊃ Q etc.
 Input: [Demonstration] Output: [Output]"""

PAIRWISE_PREFERENCE_PROMPT = (
    "Please choose your more preferred instruction: A or B ?\nInput:\n"
)

INSTANTIATION_PROMPT = """You are an expert-level LLM specialized in structured problem solving across \
domains including mathematics, programming, logic, and reasoning.

Your reasoning and response style should follow your internal preference:
\"{llm_taste}\"

Your task is to generate a complete and precise solution that strictly adheres to the \
provided thought template, while adapting it to the specifics of the task.

Follow these output rules:
- If the solution involves Python code, output exactly one code block, with no explanations, headers, or extra text.
- All Python code must be self-contained, correct, and ready to run.
- For non-code solutions, return a clean, extractable final answer in plain text.
- Do not output multiple code blocks or add commentary.
- Always align input parameters, variable names, and data formats with the user task.

Begin generating your solution only after fully reading both the task and thought template. \
Think step by step, but reflect your reasoning only through the final output.
### Output Format ###
** Thinking **: {{your internal reasoning trace here, if needed}}
** Answer **: {{final answer}}"""


# ─────────────────────────────────────────────────────────
# CPM Prompt Templates (from Appendix A.2, Figure 5)
# ─────────────────────────────────────────────────────────

CPM_KNOWN_PROMPT = """You are an expert in information synthesis, proficient in combining \
complementary insights and extracting essential details from the viewpoints of the distilled \
task definition, detailed generic logical pseudocode, case example, and input-output format.

For Known Task:
The Reversal Prompt and Benchmark Prompt should complement each other. Analyze the content \
and structure of both prompts to identify their accuracy, similarities and differences. \
Synthesize the key points and integrate them into a unified and coherent output.

Input:
LLM-Taste Prompt:{llm_taste}
Benchmark Prompt:{task_prompt}
Output:"""

CPM_UNKNOWN_PROMPT = """You are an expert in information synthesis, proficient in combining \
complementary insights and extracting essential details from the viewpoints of the distilled \
task definition, detailed generic logical pseudocode, case example, and input-output format.

For Unknown Task:
Extract a cognitive preference template T from any inaccuracies in the LLM-taste prompt. \
Integrate meta-cognitive elements from the original prompt P into this template to enhance T.

Input:
LLM-Taste Prompt:{llm_taste}
Benchmark Prompt:{task_prompt}
Output:"""


# ─────────────────────────────────────────────────────────
# Embedding Model Interface for CPM (Algorithm 2)
# ─────────────────────────────────────────────────────────

class BaseEmbeddingModel(ABC):
    """Abstract interface for computing text similarity.

    CPM (Cognitive Preference Manager) requires an embedding model to
    assess knowledge boundaries by comparing the original task definition
    with the reverse-reasoned task definition.

    The paper uses ``dunzhang/stella_en_1.5B_v5`` via sentence-transformers;
    use ``BAAI/bge-large-en-v1.5`` for transformers ≥5.x compatibility.
    """

    @abstractmethod
    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text_a: First text (typically the original task definition).
            text_b: Second text (typically the LLM-taste task definition).

        Returns:
            Cosine similarity score in [0, 1].
        """
        pass


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Embedding model using sentence-transformers (paper's default).

    Uses ``BAAI/bge-large-en-v1.5`` as default (paper recommends
    ``dunzhang/stella_en_1.5B_v5`` but it requires transformers <4.x).

    Requires:
        pip install sentence-transformers

    Example:
        >>> emb = SentenceTransformerEmbedding("BAAI/bge-large-en-v1.5")
        >>> score = emb.compute_similarity("math problem", "arithmetic task")
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install with: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self._util = util

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        emb_a = self._model.encode(text_a, convert_to_tensor=True)
        emb_b = self._model.encode(text_b, convert_to_tensor=True)
        return float(self._util.pytorch_cos_sim(emb_a, emb_b).item())



# ─────────────────────────────────────────────────────────
# RoT Baseline (with CPM)
# ─────────────────────────────────────────────────────────

class RoT(BaseBaseline):
    """Reversal of Thought (RoT) prompting baseline.

    Implements the full pipeline from the paper (Algorithms 1 & 2):
      1. Generate K candidate reverse-reasoned task definitions (PGRR).
      2. Evaluate all pairwise preferences via LLM-as-judge.
      3. Apply transitive closure to the preference matrix.
      4. Select the optimal candidate (highest combined score).
      5. Run CPM to assess knowledge boundary and refine the prompt.
      6. Instantiate the refined prompt to solve the target question.

    Attributes:
        warmup: Number of reverse reasoning candidates (K in paper).
        candidate_temperature: Sampling temperature for candidate generation.
        instantiation_temperature: Sampling temperature for final reasoning.
        demos: Few-shot demonstrations for reverse reasoning.
        embedding_model: Optional embedding model for CPM similarity.
        similarity_threshold: δ threshold for known/unknown classification.
        task_prompt: Original benchmark/task prompt for CPM comparison.

    Example:
        >>> llm = GeminiClient()
        >>> emb = SentenceTransformerEmbedding()
        >>> baseline = RoT(
        ...     llm, warmup=5,
        ...     demos="Input:1,5,5,5; Output:5*(5-1/5)=24",
        ...     embedding_model=emb,
        ...     task_prompt="Let's play a game called 24..."
        ... )
        >>> response = baseline.run("2, 5, 8, 11")
        >>> print(response.final_answer)
    """

    def __init__(
        self,
        llm: BaseLLM,
        warmup: int = 5,
        candidate_temperature: float = 0.7,
        instantiation_temperature: float = 0.1,
        demos: Optional[str] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        similarity_threshold: float = 0.7,
        task_prompt: Optional[str] = None,
    ):
        """Initialize RoT baseline.

        Args:
            llm: An instance of a BaseLLM subclass.
            warmup: Number of reverse reasoning candidates to generate (K).
            candidate_temperature: Temperature for candidate generation
                                   (higher → more diverse candidates).
            instantiation_temperature: Temperature for final answer generation
                                       (lower → more deterministic reasoning).
            demos: Few-shot demonstration string used for reverse reasoning.
                   Format: "Input:... Output:..."
            embedding_model: Embedding model for CPM knowledge boundary
                            assessment. If None, CPM is skipped and the
                            pipeline falls back to direct instantiation
                            (equivalent to w/o CPM in the ablation study).
            similarity_threshold: δ threshold for known/unknown boundary
                                  (paper recommends 0.6–0.8, default 0.7).
            task_prompt: Original benchmark/task prompt P used by CPM to
                        compare against the reverse-reasoned P*. Required
                        when embedding_model is provided.
        """
        if warmup < 1:
            raise ValueError(f"warmup must be ≥ 1, got {warmup}")
        super().__init__(llm, baseline_name="RoT")
        self.warmup = warmup
        self.candidate_temperature = candidate_temperature
        self.instantiation_temperature = instantiation_temperature
        self.demos = demos or ""
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.task_prompt = task_prompt
        # Cached results of Stage 1 & 2 — these don't depend on the question,
        # only on the demos/task, so we run them once and reuse across all questions.
        self._cached_llm_taste: Optional[str] = None
        self._cached_optimal_idx: int = 0
        self._cached_pref_scores: Dict[str, float] = {}
        # Cached Stage 3 (CPM) results, keyed by effective task prompt. CPM only
        # depends on (task_prompt, llm_taste); llm_taste is itself cached above
        # and constant per instance, so the task prompt alone is a safe key.
        self._cached_cpm: Dict[str, Tuple[str, str, float]] = {}

    # ──────────────────────────────────────────
    # Stage 1: Reverse Reasoning Warm-up
    # ──────────────────────────────────────────

    def generate_candidates(self) -> Tuple[List[str], List[float]]:
        """Generate K candidate task definitions via reverse reasoning.

        Each candidate is produced by prompting the LLM with the
        reversal-of-thought system prompt and the few-shot demos,
        using a higher temperature to encourage diversity.

        Returns:
            Tuple of (candidates, p_res) where p_res[i] is the average token
            probability (Eq. 2) for candidate i. Falls back to 1.0 when the
            backend does not support logprobs, preserving correct ranking via
            preference scores alone.
        """
        prompt = REVERSAL_OF_THOUGHT_PROMPT
        if self.demos:
            prompt = f"{prompt}\n\n{self.demos}"

        with ThreadPoolExecutor(max_workers=self.warmup) as executor:
            futures = [
                executor.submit(self.call_llm, prompt, self.candidate_temperature, True)
                for _ in range(self.warmup)
            ]
            responses = [f.result() for f in futures]

        candidates = [r.content.strip() for r in responses]
        # Use real avg token probability (Eq. 2); fall back to 1.0 if backend
        # does not support logprobs so ranking is still driven by preference scores.
        p_res = [r.avg_logprob if r.avg_logprob is not None else 1.0 for r in responses]
        return candidates, p_res

    # ──────────────────────────────────────────
    # Stage 2: Pairwise Preference Evaluation
    # ──────────────────────────────────────────

    def evaluate_preference(self, candidate_a: str, candidate_b: str) -> float:
        """Evaluate pairwise preference between two candidates.

        Uses the LLM as a judge to decide which candidate instruction
        is preferred ("A" or "B").

        Args:
            candidate_a: First candidate instruction.
            candidate_b: Second candidate instruction.

        Returns:
            Preference score for candidate A (1.0 if A preferred,
            0.0 if B preferred, 0.5 on ambiguity).
        """
        prompt = f"{PAIRWISE_PREFERENCE_PROMPT}A:{candidate_a}\nB:{candidate_b}"
        response = self.call_llm(prompt, temperature=0.0)
        choice = response.content.strip().upper()

        # Parse the LLM's preference decision
        if choice.startswith("A"):
            return 1.0
        elif choice.startswith("B"):
            return 0.0
        else:
            # Fallback: search for A or B anywhere in the response
            a_count = choice.count("A")
            b_count = choice.count("B")
            if a_count > b_count:
                return 1.0
            elif b_count > a_count:
                return 0.0
            return 0.5

    def build_preference_matrix(
        self, candidates: List[str]
    ) -> Dict[Tuple[int, int], float]:
        """Build the full pairwise preference matrix with transitive closure.

        Steps (following Algorithm 1 from the paper):
        1. Evaluate all (K choose 2) pairwise comparisons.
        2. Apply transitive closure:
           P_pre[(i,k)] = max(P_pre[(i,k)], P_pre[(i,j)] * P_pre[(j,k)])

        Args:
            candidates: List of K candidate strings.

        Returns:
            Preference matrix as a dict mapping (i, j) → score.
        """
        k = len(candidates)
        p_pre: Dict[Tuple[int, int], float] = {}

        # Step 1: Direct pairwise comparisons (all pairs in parallel)
        pairs = [(i, j) for i in range(k - 1) for j in range(i + 1, k)]

        def _compare(i: int, j: int) -> Tuple[int, int, float]:
            return i, j, self.evaluate_preference(candidates[i], candidates[j])

        with ThreadPoolExecutor(max_workers=len(pairs) or 1) as executor:
            futures = [executor.submit(_compare, i, j) for i, j in pairs]
            for future in as_completed(futures):
                i, j, score = future.result()
                p_pre[(i, j)] = score
                p_pre[(j, i)] = 1.0 - score

        # Step 2: Transitive closure (Floyd–Warshall–style propagation)
        for j in range(k):
            for i in range(k):
                if i == j:
                    continue
                for m in range(k):
                    if m == i or m == j:
                        continue
                    transitive_score = p_pre.get((i, j), 0.0) * p_pre.get((j, m), 0.0)
                    p_pre[(i, m)] = max(p_pre.get((i, m), 0.0), transitive_score)

        return p_pre

    def select_optimal(
        self,
        candidates: List[str],
        p_pre: Dict[Tuple[int, int], float],
        p_res: List[float],
    ) -> Tuple[int, str]:
        """Select the optimal candidate using Eq. 6 from the paper.

        Computes the combined score (P_res[i] + P_pre_avg[i]) / 2 for each
        candidate and returns the argmax, matching the reference implementation.

        Args:
            candidates: List of K candidate strings.
            p_pre: Preference matrix from build_preference_matrix.
            p_res: Generation probability scores from generate_candidates.
                   Uniform 1.0 when logprobs are unavailable.

        Returns:
            Tuple of (optimal_index, optimal_candidate_string).
        """
        k = len(candidates)
        scores = []

        for i in range(k):
            if k > 1:
                avg_pref = sum(
                    p_pre.get((i, j), 0.0) for j in range(k) if j != i
                ) / (k - 1)
            else:
                avg_pref = 1.0
            # Eq. 6: P_opt = argmax_Ri [ (P_res^i + P_pre_avg(Ri)) / 2 ]
            scores.append((p_res[i] + avg_pref) / 2)

        optimal_idx = max(range(k), key=lambda i: scores[i])
        return optimal_idx, candidates[optimal_idx]

    # ──────────────────────────────────────────
    # Stage 3: Cognitive Preference Manager
    # ──────────────────────────────────────────

    @staticmethod
    def extract_task_definition(text: str) -> str:
        """Extract the task definition section from a prompt or LLM output.

        Looks for common markers like "Task Definition:", "Task Defination:",
        or falls back to the full text if no marker is found.

        Args:
            text: The full prompt or reverse-reasoned output.

        Returns:
            The extracted task definition string.
        """
        # Try multiple common patterns from the paper's outputs
        patterns = [
            r"(?:Task\s+Defin[ai]tion)\s*[:：]\s*(.*?)(?=\n\s*(?:Logical|Pseudocode|Case|Input-Output|$))",
            r"(?:Task\s+Defin[ai]tion)\s*[:：]\s*(.*?)(?:\n\n|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 10:  # Ensure it's meaningful
                    return extracted

        # Fallback: return first ~500 chars (enough for similarity)
        return text[:500].strip()

    def compute_knowledge_boundary(
        self,
        task_prompt: str,
        llm_taste: str,
    ) -> Tuple[str, float]:
        """Assess whether the task is known or unknown (Eq. 7 from paper).

        Computes embedding similarity between the original task definition
        (P_task) and the reverse-reasoned task definition (P*_task).

        Args:
            task_prompt: The original benchmark/task prompt P.
            llm_taste: The optimal reverse-reasoned LLM-taste prompt P*.

        Returns:
            Tuple of (boundary_signal, similarity_score) where
            boundary_signal is "known" or "unknown".
        """
        p_task = self.extract_task_definition(task_prompt)
        p_star_task = self.extract_task_definition(llm_taste)

        similarity = self.embedding_model.compute_similarity(p_task, p_star_task)

        if similarity >= self.similarity_threshold:
            return "known", similarity
        else:
            return "unknown", similarity

    def aggregate_known(self, task_prompt: str, llm_taste: str) -> str:
        """Solution Logic Aggregation for known tasks (Algorithm 2, line 4-6).

        Merges beneficial aspects from the original prompt P with the
        LLM-taste prompt P* to create the final refined prompt P_final.

        Args:
            task_prompt: Original benchmark/task prompt P.
            llm_taste: Optimal reverse-reasoned LLM-taste prompt P*.

        Returns:
            The aggregated P_final string.
        """
        prompt = CPM_KNOWN_PROMPT.format(
            llm_taste=llm_taste,
            task_prompt=task_prompt,
        )
        response = self.call_llm(prompt, temperature=0.0)
        return response.content.strip()

    def aggregate_unknown(self, task_prompt: str, llm_taste: str) -> str:
        """Stylistic Template Aggregation for unknown tasks (Algorithm 2, line 8-10).

        Extracts a cognitive preference template T from the LLM-taste prompt
        and integrates meta-cognitive elements from the original prompt P
        into T to construct the final prompt P_final.

        Args:
            task_prompt: Original benchmark/task prompt P.
            llm_taste: Optimal reverse-reasoned LLM-taste prompt P*.

        Returns:
            The aggregated P_final string.
        """
        prompt = CPM_UNKNOWN_PROMPT.format(
            llm_taste=llm_taste,
            task_prompt=task_prompt,
        )
        response = self.call_llm(prompt, temperature=0.0)
        return response.content.strip()

    def _run_cpm(
        self,
        task_prompt: str,
        llm_taste: str,
    ) -> Tuple[str, str, float]:
        """Execute the full Cognitive Preference Manager (Algorithm 2).

        Assesses the knowledge boundary and applies the appropriate
        aggregation strategy to produce the final refined prompt.

        Args:
            task_prompt: Original benchmark/task prompt P.
            llm_taste: Optimal reverse-reasoned LLM-taste prompt P*.

        Returns:
            Tuple of (p_final, boundary_signal, similarity_score).
        """
        boundary, similarity = self.compute_knowledge_boundary(
            task_prompt, llm_taste
        )

        if boundary == "known":
            p_final = self.aggregate_known(task_prompt, llm_taste)
        else:
            p_final = self.aggregate_unknown(task_prompt, llm_taste)

        return p_final, boundary, similarity

    @property
    def cpm_enabled(self) -> bool:
        """Whether CPM is active (requires embedding_model and task_prompt)."""
        return self.embedding_model is not None and self.task_prompt is not None

    # ──────────────────────────────────────────
    # Stage 4: Instantiation
    # ──────────────────────────────────────────

    def build_instantiation_prompt(self, llm_taste: str) -> str:
        """Build the instantiation system prompt with the optimal LLM taste.

        Args:
            llm_taste: The optimal (possibly CPM-refined) task definition.

        Returns:
            Formatted instantiation prompt string.
        """
        return INSTANTIATION_PROMPT.format(llm_taste=llm_taste)

    def parse_instantiation_response(self, response_text: str) -> Tuple[str, str]:
        """Parse the instantiation response to extract thinking and answer.

        Args:
            response_text: Raw LLM response following the RoT output format.

        Returns:
            Tuple of (final_answer, thinking_trace).
        """
        # Extract thinking trace
        thinking_pattern = re.compile(
            r"\*\*\s*Thinking\s*\*\*\s*:\s*(.*?)(?=\*\*\s*Answer|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        thinking_match = thinking_pattern.search(response_text)
        thinking = thinking_match.group(1).strip() if thinking_match else ""

        # Extract final answer — greedy match to end-of-string so that
        # multi-line or multi-paragraph answers are not truncated at the
        # first blank line (fixes Issue 9).
        answer_pattern = re.compile(
            r"\*\*\s*Answer\s*\*\*\s*:\s*(.*)\Z",
            re.DOTALL | re.IGNORECASE,
        )
        answer_match = answer_pattern.search(response_text)
        if answer_match:
            final_answer = answer_match.group(1).strip()
        else:
            # Fallback: use the last non-empty line
            lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
            final_answer = lines[-1] if lines else response_text.strip()

        return final_answer, thinking

    # ──────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────

    def run(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        temperature: float = 0.0,
        task_prompt: Optional[str] = None,
        **kwargs,
    ) -> BaselineResponse:
        """Execute the full RoT pipeline on the given question.

        Pipeline stages:
        1. Generate K candidate task definitions (Reverse Reasoning Warm-up)
        2. Evaluate pairwise preferences and select optimal candidate
        3. Run CPM to assess knowledge boundary and refine prompt (if enabled)
        4. Instantiate the (refined) prompt to solve the question

        Args:
            question: The input question or problem to solve.
            system_prompt: Optional system-level instruction (unused in RoT,
                          kept for interface compatibility).
            instruction: Optional task-specific instruction (unused in RoT).
            temperature: Overrides instantiation_temperature if provided
                        and non-zero.
            task_prompt: Per-call override for self.task_prompt. The original
                        benchmark prompt P used by CPM for knowledge boundary
                        assessment. If provided, overrides the instance-level
                        task_prompt for this run only.
            **kwargs: Additional arguments for interface compatibility.

        Returns:
            BaselineResponse containing the answer, full reasoning trace,
            and detailed metadata about the PGRR + CPM process.
        """
        self.reset_counters()
        intermediate_steps = []
        inst_temp = temperature if temperature > 0 else self.instantiation_temperature

        # Resolve task_prompt: per-call override > instance-level
        effective_task_prompt = task_prompt or self.task_prompt

        # ── Stage 1 & 2: run once, then reuse cached llm_taste ──
        if self._cached_llm_taste is None:
            candidates, p_res = self.generate_candidates()
            intermediate_steps.append(
                f"[Stage 1: Reverse Reasoning] Generated {len(candidates)} candidates"
            )
            for idx, cand in enumerate(candidates):
                intermediate_steps.append(f"[Candidate {idx + 1}]\n{cand}")

            p_pre = self.build_preference_matrix(candidates)
            optimal_idx, llm_taste = self.select_optimal(candidates, p_pre, p_res)

            k = len(candidates)
            self._cached_pref_scores = {
                f"candidate_{i + 1}": round(
                    sum(p_pre.get((i, j), 0.0) for j in range(k) if j != i) / max(k - 1, 1), 4
                )
                for i in range(k)
            }
            self._cached_optimal_idx = optimal_idx
            self._cached_llm_taste = llm_taste
            intermediate_steps.append(
                f"[Stage 2: Preference Selection] Selected candidate {optimal_idx + 1}"
            )
        else:
            llm_taste = self._cached_llm_taste
            optimal_idx = self._cached_optimal_idx
            intermediate_steps.append("[Stage 1 & 2: Skipped] Using cached llm_taste")

        # ── Stage 3: Cognitive Preference Manager ──
        cpm_boundary = None
        cpm_similarity = None

        if self.embedding_model is not None and effective_task_prompt is not None:
            cached = self._cached_cpm.get(effective_task_prompt)
            if cached is not None:
                p_final, cpm_boundary, cpm_similarity = cached
                intermediate_steps.append(
                    f"[Stage 3: CPM] Boundary={cpm_boundary} "
                    f"(similarity={cpm_similarity:.4f}, δ={self.similarity_threshold}) [cached]"
                )
            else:
                p_final, cpm_boundary, cpm_similarity = self._run_cpm(
                    effective_task_prompt, llm_taste
                )
                self._cached_cpm[effective_task_prompt] = (
                    p_final, cpm_boundary, cpm_similarity
                )
                intermediate_steps.append(
                    f"[Stage 3: CPM] Boundary={cpm_boundary} "
                    f"(similarity={cpm_similarity:.4f}, δ={self.similarity_threshold})"
                )
            intermediate_steps.append(f"[CPM P_final]\n{p_final}")
        else:
            # CPM disabled: use raw LLM-taste (equivalent to w/o CPM ablation)
            p_final = llm_taste
            intermediate_steps.append(
                "[Stage 3: CPM] Skipped (no embedding_model or task_prompt)"
            )

        # ── Stage 4: Instantiation ──
        instantiation_sys = self.build_instantiation_prompt(p_final)
        task_context = ""
        if system_prompt:
            task_context += f"{system_prompt}\n"
        if instruction:
            task_context += f"{instruction}\n"
        question_prompt = f"{instantiation_sys}\n\n{task_context}{question}"
        response = self.call_llm(question_prompt, temperature=inst_temp)
        final_answer, thinking = self.parse_instantiation_response(
            response.content
        )

        intermediate_steps.append(
            f"[Stage 4: Instantiation]\n{response.content.strip()}"
        )

        # Build reasoning trace
        reasoning_trace = (
            f"[RoT Warm-up] Selected candidate {optimal_idx + 1}/{self.warmup}\n"
            f"[LLM Taste]\n{llm_taste}\n\n"
        )
        if cpm_boundary is not None:
            reasoning_trace += (
                f"[CPM] Boundary: {cpm_boundary} "
                f"(similarity: {cpm_similarity:.4f})\n"
                f"[P_final]\n{p_final}\n\n"
            )
        reasoning_trace += f"[Thinking]\n{thinking}"

        return self.create_response(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            intermediate_steps=intermediate_steps,
            metadata={
                "model": response.model_name,
                "warmup": self.warmup,
                "candidate_temperature": self.candidate_temperature,
                "instantiation_temperature": inst_temp,
                "optimal_candidate_index": optimal_idx,
                "preference_scores": self._cached_pref_scores,
                "demos": self.demos,
                "cpm_enabled": self.cpm_enabled,
                "cpm_boundary": cpm_boundary,
                "cpm_similarity": cpm_similarity,
                "similarity_threshold": self.similarity_threshold,
            },
        )