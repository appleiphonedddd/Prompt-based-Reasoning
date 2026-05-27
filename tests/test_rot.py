"""
Tests for the Reversal of Thought (RoT) baseline implementation.

Tests cover:
- Candidate generation (Stage 1) with logprob-derived P_res
- Pairwise preference evaluation (Stage 2)
- Transitive closure in the preference matrix
- Optimal candidate selection (Eq. 6)
- Cognitive Preference Manager / CPM (Stage 3)
  - Task definition extraction
  - Knowledge boundary assessment (known vs unknown)
  - Solution logic aggregation (known tasks)
  - Stylistic template aggregation (unknown tasks)
  - CPM integration + result caching in the full pipeline
- Instantiation response parsing (Stage 4)
- Full pipeline end-to-end execution (with and without CPM)

Stages 1–2 and the preference matrix are parallelised with thread pools, so
tests that exercise them use a thread-safe mock that routes responses by prompt
content (rather than by call order) to stay deterministic.

Author: Egor Morozov
"""

import threading
import unittest
from typing import Callable, Dict, List, Optional, Union

from models.base import BaseLLM, LLMResponse
from baseline.RoT.rot import (
    RoT,
    BaseEmbeddingModel,
    SentenceTransformerEmbedding,
)


# ─────────────────────────────────────────────────
# Mocks
# ─────────────────────────────────────────────────

# Distinct substrings identifying each prompt type (see rot.py templates).
_REVERSAL_MARKER = "highly distinguished expert"
_PREFERENCE_MARKER = "preferred instruction"
_CPM_MARKER = "information synthesis"
_INSTANTIATION_MARKER = "expert-level LLM"


class SequentialMockLLM(BaseLLM):
    """Mock LLM returning pre-defined responses in sequence (single-threaded use)."""

    def __init__(self, responses: List[str]):
        super().__init__(api_key="dummy", model="mock-rot")
        self.responses = responses
        self.call_counter = 0

    def generate(
        self, prompt: str, temperature: float = 0, logprobs: bool = False
    ) -> LLMResponse:
        if self.call_counter < len(self.responses):
            content = self.responses[self.call_counter]
            self.call_counter += 1
        else:
            content = ""
        return LLMResponse(
            content=content,
            model_name="mock-rot",
            input_tokens=10,
            output_tokens=10,
            avg_logprob=0.9 if logprobs else None,
        )


class RoutedMockLLM(BaseLLM):
    """Thread-safe mock that routes responses by prompt content.

    Routing keeps end-to-end tests deterministic despite the thread pools used
    in candidate generation and preference evaluation.

    Args:
        candidate_responses: Pool of reverse-reasoning candidate definitions
            (handed out round-robin; ordering is irrelevant to assertions).
        prefer: Either ``"A"``/``"B"`` (always prefer that position) or a
            callable taking the preference prompt and returning ``"A"``/``"B"``.
        cpm_response: Response for CPM aggregation prompts.
        instantiation_response: Response for the final instantiation prompt.
    """

    def __init__(
        self,
        candidate_responses: List[str],
        prefer: Union[str, Callable[[str], str]] = "A",
        cpm_response: str = "CPM refined prompt",
        instantiation_response: str = "** Answer **: 42",
    ):
        super().__init__(api_key="dummy", model="mock-routed")
        self.candidate_responses = candidate_responses
        self.prefer = prefer
        self.cpm_response = cpm_response
        self.instantiation_response = instantiation_response
        self._cand_idx = 0
        self._lock = threading.Lock()

    def generate(
        self, prompt: str, temperature: float = 0, logprobs: bool = False
    ) -> LLMResponse:
        if _PREFERENCE_MARKER in prompt:
            choice = self.prefer(prompt) if callable(self.prefer) else self.prefer
            content = choice
        elif _CPM_MARKER in prompt:
            content = self.cpm_response
        elif _INSTANTIATION_MARKER in prompt:
            content = self.instantiation_response
        elif _REVERSAL_MARKER in prompt:
            with self._lock:
                content = self.candidate_responses[
                    self._cand_idx % len(self.candidate_responses)
                ]
                self._cand_idx += 1
        else:
            content = self.instantiation_response

        return LLMResponse(
            content=content,
            model_name="mock-routed",
            input_tokens=10,
            output_tokens=10,
            avg_logprob=0.9 if logprobs else None,
        )


class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock embedding model that returns a fixed similarity score."""

    def __init__(self, fixed_score: float = 0.8):
        self.fixed_score = fixed_score
        self.call_count = 0
        self.last_text_a = None
        self.last_text_b = None

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        self.call_count += 1
        self.last_text_a = text_a
        self.last_text_b = text_b
        return self.fixed_score


def _prefer_b(_prompt: str) -> str:
    """Preference router that always prefers candidate B (the higher index)."""
    return "B"


# ─────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────

class TestRoTInitialization(unittest.TestCase):
    """Test RoT initialization and configuration."""

    def test_default_initialization(self):
        llm = SequentialMockLLM([])
        baseline = RoT(llm)

        self.assertEqual(baseline.baseline_name, "RoT")
        self.assertEqual(baseline.warmup, 5)
        self.assertEqual(baseline.candidate_temperature, 0.7)
        self.assertEqual(baseline.instantiation_temperature, 0.1)
        # With no demos supplied, demos defaults to an empty string.
        self.assertEqual(baseline.demos, "")
        # CPM defaults
        self.assertIsNone(baseline.embedding_model)
        self.assertEqual(baseline.similarity_threshold, 0.7)
        self.assertIsNone(baseline.task_prompt)
        self.assertFalse(baseline.cpm_enabled)

    def test_custom_initialization(self):
        llm = SequentialMockLLM([])
        baseline = RoT(
            llm,
            warmup=3,
            candidate_temperature=0.9,
            instantiation_temperature=0.2,
            demos="Input: test\nOutput: result",
        )

        self.assertEqual(baseline.warmup, 3)
        self.assertEqual(baseline.candidate_temperature, 0.9)
        self.assertEqual(baseline.instantiation_temperature, 0.2)
        self.assertEqual(baseline.demos, "Input: test\nOutput: result")

    def test_invalid_warmup_raises(self):
        llm = SequentialMockLLM([])
        with self.assertRaises(ValueError):
            RoT(llm, warmup=0)

    def test_initialization_with_cpm(self):
        llm = SequentialMockLLM([])
        emb = MockEmbeddingModel(0.8)
        baseline = RoT(
            llm,
            embedding_model=emb,
            similarity_threshold=0.75,
            task_prompt="Solve the math problem.",
        )

        self.assertIs(baseline.embedding_model, emb)
        self.assertEqual(baseline.similarity_threshold, 0.75)
        self.assertEqual(baseline.task_prompt, "Solve the math problem.")
        self.assertTrue(baseline.cpm_enabled)

    def test_cpm_not_enabled_without_task_prompt(self):
        llm = SequentialMockLLM([])
        emb = MockEmbeddingModel(0.8)
        baseline = RoT(llm, embedding_model=emb)
        self.assertFalse(baseline.cpm_enabled)

    def test_cpm_not_enabled_without_embedding_model(self):
        llm = SequentialMockLLM([])
        baseline = RoT(llm, task_prompt="Some task")
        self.assertFalse(baseline.cpm_enabled)


# ─────────────────────────────────────────────────
# Stage 1: Candidate generation
# ─────────────────────────────────────────────────

class TestRoTCandidateGeneration(unittest.TestCase):
    """Test reverse-reasoning candidate generation (returns candidates + P_res)."""

    def test_returns_candidates_and_pres(self):
        llm = RoutedMockLLM(["def A", "def B", "def C"])
        baseline = RoT(llm, warmup=3, demos="Input: x\nOutput: y")

        candidates, p_res = baseline.generate_candidates()

        self.assertEqual(len(candidates), 3)
        self.assertEqual(len(p_res), 3)
        # avg_logprob=0.9 is requested via logprobs=True for candidates.
        self.assertTrue(all(p == 0.9 for p in p_res))

    def test_pres_fallback_when_no_logprobs(self):
        """When the backend returns no logprob, P_res falls back to 1.0."""

        class NoLogprobLLM(RoutedMockLLM):
            def generate(self, prompt, temperature=0, logprobs=False):
                resp = super().generate(prompt, temperature, logprobs=False)
                return resp  # avg_logprob stays None

        llm = NoLogprobLLM(["def A", "def B"])
        baseline = RoT(llm, warmup=2)
        _, p_res = baseline.generate_candidates()
        self.assertEqual(p_res, [1.0, 1.0])


# ─────────────────────────────────────────────────
# Stage 2: Preference evaluation
# ─────────────────────────────────────────────────

class TestRoTPreferenceEvaluation(unittest.TestCase):
    """Test the pairwise preference evaluation logic."""

    def test_preference_chooses_a(self):
        llm = SequentialMockLLM(["A"])
        baseline = RoT(llm, warmup=1)
        score = baseline.evaluate_preference("candidate_a", "candidate_b")
        self.assertEqual(score, 1.0)

    def test_preference_chooses_b(self):
        llm = SequentialMockLLM(["B"])
        baseline = RoT(llm, warmup=1)
        score = baseline.evaluate_preference("candidate_a", "candidate_b")
        self.assertEqual(score, 0.0)

    def test_preference_ambiguous_fallback(self):
        llm = SequentialMockLLM(["nothing decisive here"])
        baseline = RoT(llm, warmup=1)
        score = baseline.evaluate_preference("candidate_a", "candidate_b")
        self.assertEqual(score, 0.5)

    def test_preference_with_verbose_response(self):
        llm = SequentialMockLLM(["A is clearly better because..."])
        baseline = RoT(llm, warmup=1)
        score = baseline.evaluate_preference("candidate_a", "candidate_b")
        self.assertEqual(score, 1.0)


class TestRoTPreferenceMatrix(unittest.TestCase):
    """Test preference matrix construction and transitive closure.

    A fixed candidate list is passed directly (rather than generated) so that
    matrix indices map to known candidates regardless of thread scheduling.
    """

    def test_two_candidates_matrix(self):
        llm = RoutedMockLLM([], prefer="A")
        baseline = RoT(llm, warmup=2)

        p_pre = baseline.build_preference_matrix(["c0", "c1"])
        self.assertEqual(p_pre[(0, 1)], 1.0)
        self.assertEqual(p_pre[(1, 0)], 0.0)

    def test_three_candidates_transitivity(self):
        """Direct 0≻2 is missing (c2 wins head-to-head) but transitivity via
        0≻1≻2 fills p_pre[(0,2)] back to 1.0."""

        def router(prompt: str) -> str:
            # prompt tail: "...A:<a>\nB:<b>"
            a = prompt.split("A:")[-1].split("\nB:")[0].strip()
            b = prompt.split("\nB:")[-1].strip()
            pair = {a, b}
            if pair == {"c0", "c1"}:
                return "A" if a == "c0" else "B"
            if pair == {"c1", "c2"}:
                return "A" if a == "c1" else "B"
            if pair == {"c0", "c2"}:  # c2 wins the direct comparison
                return "A" if a == "c2" else "B"
            return "A"

        llm = RoutedMockLLM([], prefer=router)
        baseline = RoT(llm, warmup=3)

        p_pre = baseline.build_preference_matrix(["c0", "c1", "c2"])
        self.assertEqual(p_pre[(0, 1)], 1.0)
        self.assertEqual(p_pre[(1, 2)], 1.0)
        # Transitive closure recovers the 0≻2 edge.
        self.assertEqual(p_pre[(0, 2)], 1.0)


class TestRoTOptimalSelection(unittest.TestCase):
    """Test optimal candidate selection (Eq. 6: argmax (P_res + P̄_pre)/2)."""

    def test_selects_preferred_candidate(self):
        llm = SequentialMockLLM([])
        baseline = RoT(llm, warmup=2)

        candidates = ["weak_candidate", "strong_candidate"]
        p_pre = {(0, 1): 0.0, (1, 0): 1.0}
        p_res = [1.0, 1.0]

        idx, text = baseline.select_optimal(candidates, p_pre, p_res)
        self.assertEqual(idx, 1)
        self.assertEqual(text, "strong_candidate")

    def test_pres_breaks_preference_tie(self):
        """With equal preference scores, higher P_res wins."""
        llm = SequentialMockLLM([])
        baseline = RoT(llm, warmup=2)

        candidates = ["cand0", "cand1"]
        p_pre = {(0, 1): 0.5, (1, 0): 0.5}
        p_res = [0.2, 0.9]

        idx, text = baseline.select_optimal(candidates, p_pre, p_res)
        self.assertEqual(idx, 1)
        self.assertEqual(text, "cand1")

    def test_single_candidate(self):
        llm = SequentialMockLLM([])
        baseline = RoT(llm, warmup=1)

        idx, text = baseline.select_optimal(["only_candidate"], {}, [1.0])
        self.assertEqual(idx, 0)
        self.assertEqual(text, "only_candidate")


# ─────────────────────────────────────────────────
# Stage 4: Response parsing
# ─────────────────────────────────────────────────

class TestRoTResponseParsing(unittest.TestCase):
    """Test the instantiation response parsing."""

    def test_parse_standard_format(self):
        baseline = RoT(SequentialMockLLM([]))
        response = (
            "** Thinking **: First we add 2+5=7, then 8-7=1, then 11*1=11.\n\n"
            "** Answer **: (2+5) * 8 - 11 = 29"
        )
        answer, thinking = baseline.parse_instantiation_response(response)
        self.assertIn("2+5", thinking)
        self.assertIn("(2+5)", answer)

    def test_parse_no_thinking(self):
        baseline = RoT(SequentialMockLLM([]))
        answer, thinking = baseline.parse_instantiation_response("** Answer **: 42")
        self.assertEqual(answer, "42")
        self.assertEqual(thinking, "")

    def test_parse_multiline_answer_not_truncated(self):
        baseline = RoT(SequentialMockLLM([]))
        response = "** Answer **: line one\n\nline two\n\nline three"
        answer, _ = baseline.parse_instantiation_response(response)
        self.assertIn("line one", answer)
        self.assertIn("line three", answer)

    def test_parse_fallback_no_markers(self):
        baseline = RoT(SequentialMockLLM([]))
        answer, _ = baseline.parse_instantiation_response("The solution is 24.")
        self.assertEqual(answer, "The solution is 24.")


# ─────────────────────────────────────────────────
# Stage 3: CPM
# ─────────────────────────────────────────────────

class TestRoTTaskDefinitionExtraction(unittest.TestCase):
    """Test the task definition extraction used by CPM."""

    def test_extract_with_marker(self):
        text = (
            "Task Definition: The task is to solve a math problem.\n\n"
            "Logical Pseudocode: For each permutation..."
        )
        result = RoT(SequentialMockLLM([])).extract_task_definition(text)
        self.assertIn("solve a math problem", result)
        self.assertNotIn("Logical Pseudocode", result)

    def test_extract_with_misspelled_marker(self):
        text = (
            "Task Defination: Find a feasible mathematical expression.\n\n"
            "Pseudocode: step 1..."
        )
        result = RoT(SequentialMockLLM([])).extract_task_definition(text)
        self.assertIn("feasible mathematical expression", result)

    def test_extract_fallback_no_marker(self):
        text = "This is just a plain prompt without any task definition marker."
        result = RoT(SequentialMockLLM([])).extract_task_definition(text)
        self.assertIn("plain prompt", result)


class TestRoTKnowledgeBoundary(unittest.TestCase):
    """Test knowledge boundary assessment (CPM Algorithm 2, Eq. 7)."""

    def test_known_boundary(self):
        emb = MockEmbeddingModel(fixed_score=0.85)
        baseline = RoT(
            SequentialMockLLM([]),
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Solve 24 game",
        )
        boundary, score = baseline.compute_knowledge_boundary(
            "Solve the 24 game",
            "Task Definition: Find expression equaling 24.\n\nLogical Pseudocode: ...",
        )
        self.assertEqual(boundary, "known")
        self.assertAlmostEqual(score, 0.85)
        self.assertEqual(emb.call_count, 1)

    def test_unknown_boundary(self):
        emb = MockEmbeddingModel(fixed_score=0.5)
        baseline = RoT(
            SequentialMockLLM([]),
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Solve math word problems",
        )
        boundary, score = baseline.compute_knowledge_boundary(
            "Solve math word problems",
            "Task Definition: Construct a Python function.\n\nLogical Pseudocode: ...",
        )
        self.assertEqual(boundary, "unknown")
        self.assertAlmostEqual(score, 0.5)

    def test_boundary_at_threshold(self):
        emb = MockEmbeddingModel(fixed_score=0.7)
        baseline = RoT(
            SequentialMockLLM([]), embedding_model=emb, similarity_threshold=0.7
        )
        boundary, _ = baseline.compute_knowledge_boundary("A", "B")
        self.assertEqual(boundary, "known")

    def test_custom_threshold(self):
        emb = MockEmbeddingModel(fixed_score=0.65)
        baseline = RoT(
            SequentialMockLLM([]), embedding_model=emb, similarity_threshold=0.6
        )
        boundary, _ = baseline.compute_knowledge_boundary("A", "B")
        self.assertEqual(boundary, "known")


class TestRoTAggregation(unittest.TestCase):
    """Test CPM aggregation methods (known and unknown paths)."""

    def test_aggregate_known(self):
        llm = SequentialMockLLM(["Merged prompt: solve 24 with arithmetic ops."])
        result = RoT(llm).aggregate_known(
            task_prompt="Let's play 24 game.",
            llm_taste="Task Definition: Find expression equaling 24.",
        )
        self.assertIn("Merged prompt", result)
        self.assertEqual(llm.call_counter, 1)

    def test_aggregate_unknown(self):
        llm = SequentialMockLLM(["Enhanced template with meta-cognitive elements."])
        result = RoT(llm).aggregate_unknown(
            task_prompt="Solve multilingual math problems.",
            llm_taste="Task Definition: Parse equations.",
        )
        self.assertIn("Enhanced template", result)
        self.assertEqual(llm.call_counter, 1)

    def test_run_cpm_known_path(self):
        llm = SequentialMockLLM(["Refined known prompt"])
        emb = MockEmbeddingModel(fixed_score=0.85)
        baseline = RoT(
            llm, embedding_model=emb, similarity_threshold=0.7, task_prompt="Solve 24"
        )
        p_final, boundary, similarity = baseline._run_cpm(
            "Solve 24",
            "Task Definition: Make 24 from numbers.\n\nLogical Pseudocode: ...",
        )
        self.assertEqual(boundary, "known")
        self.assertAlmostEqual(similarity, 0.85)
        self.assertEqual(p_final, "Refined known prompt")

    def test_run_cpm_unknown_path(self):
        llm = SequentialMockLLM(["Refined unknown prompt"])
        emb = MockEmbeddingModel(fixed_score=0.4)
        baseline = RoT(
            llm,
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Write Python puzzles",
        )
        p_final, boundary, similarity = baseline._run_cpm(
            "Write Python puzzles",
            "Task Definition: Sort words alphabetically.\n\nLogical Pseudocode: ...",
        )
        self.assertEqual(boundary, "unknown")
        self.assertAlmostEqual(similarity, 0.4)
        self.assertEqual(p_final, "Refined unknown prompt")


class TestEmbeddingInterface(unittest.TestCase):
    """Sanity checks for the embedding-model interface used by CPM."""

    def test_base_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseEmbeddingModel()  # abstract — cannot instantiate

    def test_sentence_transformer_is_subclass(self):
        self.assertTrue(issubclass(SentenceTransformerEmbedding, BaseEmbeddingModel))

    def test_mock_embedding_similarity(self):
        emb = MockEmbeddingModel(0.42)
        self.assertAlmostEqual(emb.compute_similarity("a", "b"), 0.42)
        self.assertEqual(emb.call_count, 1)


# ─────────────────────────────────────────────────
# End-to-end
# ─────────────────────────────────────────────────

class TestRoTEndToEnd(unittest.TestCase):
    """End-to-end tests for the full RoT pipeline."""

    def test_full_pipeline_warmup_2_no_cpm(self):
        """2 candidates + 1 pairwise + 1 instantiation = 4 calls.

        Preference always prefers B, so the 2nd candidate (index 1) is optimal
        regardless of thread scheduling (the single pair is always A=c0, B=c1).
        """
        llm = RoutedMockLLM(
            ["Task: make 24 (v0)", "Task: make 24 (v1)"],
            prefer=_prefer_b,
            instantiation_response="** Answer **: (11 - 5) * (8 / 2) = 24",
        )
        baseline = RoT(llm, warmup=2)

        response = baseline.run("2, 5, 8, 11")

        self.assertIn("24", response.final_answer)
        self.assertEqual(response.baseline_type, "RoT")
        self.assertEqual(response.metadata["warmup"], 2)
        self.assertEqual(response.metadata["optimal_candidate_index"], 1)
        self.assertEqual(response.num_llm_calls, 4)
        self.assertEqual(response.total_tokens, 80)
        self.assertFalse(response.metadata["cpm_enabled"])
        self.assertIsNone(response.metadata["cpm_boundary"])
        self.assertIsNone(response.metadata["cpm_similarity"])

    def test_full_pipeline_warmup_2_with_cpm_known(self):
        """2 candidates + 1 pairwise + 1 CPM + 1 instantiation = 5 calls."""
        llm = RoutedMockLLM(
            ["Task Definition: make 24 (v0)", "Task Definition: make 24 (v1)"],
            prefer="A",
            cpm_response="Refined: find an expression using four numbers to equal 24.",
            instantiation_response="** Answer **: (11 - 5) * (8 / 2) = 24",
        )
        emb = MockEmbeddingModel(fixed_score=0.85)
        baseline = RoT(
            llm,
            warmup=2,
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Let's play a game called 24.",
        )

        response = baseline.run("2, 5, 8, 11")

        self.assertIn("24", response.final_answer)
        self.assertEqual(response.num_llm_calls, 5)
        self.assertEqual(response.total_tokens, 100)
        self.assertTrue(response.metadata["cpm_enabled"])
        self.assertEqual(response.metadata["cpm_boundary"], "known")
        self.assertAlmostEqual(response.metadata["cpm_similarity"], 0.85)
        self.assertTrue(
            any("CPM" in s and "known" in s for s in response.intermediate_steps)
        )

    def test_full_pipeline_warmup_2_with_cpm_unknown(self):
        llm = RoutedMockLLM(
            ["Task Definition: parse code (v0)", "Task Definition: parse code (v1)"],
            prefer="A",
            cpm_response="Enhanced template with meta-cognitive elements.",
            instantiation_response="** Answer **: def solve(): return True",
        )
        emb = MockEmbeddingModel(fixed_score=0.4)
        baseline = RoT(
            llm,
            warmup=2,
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Solve Python programming puzzles.",
        )

        response = baseline.run("Find x such that x*2 == 10")

        self.assertEqual(response.num_llm_calls, 5)
        self.assertEqual(response.metadata["cpm_boundary"], "unknown")
        self.assertAlmostEqual(response.metadata["cpm_similarity"], 0.4)

    def test_full_pipeline_warmup_3(self):
        """3 candidates + 3 pairwise + 1 instantiation = 7 calls.

        Preference always prefers A (lower index), so candidate 0 is optimal.
        """
        llm = RoutedMockLLM(
            ["def 0", "def 1", "def 2"],
            prefer="A",
            instantiation_response="** Answer **: 42",
        )
        baseline = RoT(llm, warmup=3)

        response = baseline.run("test question")

        self.assertEqual(response.final_answer, "42")
        self.assertEqual(response.num_llm_calls, 7)
        self.assertEqual(response.metadata["optimal_candidate_index"], 0)

    def test_per_call_task_prompt_override(self):
        llm = RoutedMockLLM(
            ["Task Definition: Generic task."],
            cpm_response="Refined with per-call task prompt",
            instantiation_response="** Answer **: 99",
        )
        emb = MockEmbeddingModel(fixed_score=0.9)
        baseline = RoT(
            llm, warmup=1, embedding_model=emb, task_prompt="Default task prompt"
        )

        response = baseline.run("Q", task_prompt="Overridden task prompt")

        self.assertEqual(response.final_answer, "99")
        self.assertEqual(response.metadata["cpm_boundary"], "known")

    def test_cpm_skipped_when_no_embedding_model(self):
        llm = RoutedMockLLM(["candidate_def"], instantiation_response="** Answer **: done")
        baseline = RoT(llm, warmup=1, task_prompt="Some task")

        response = baseline.run("test")

        # 1 candidate + 1 instantiation (no CPM, no pairwise for warmup=1).
        self.assertEqual(response.num_llm_calls, 2)
        self.assertFalse(response.metadata["cpm_enabled"])
        self.assertTrue(any("Skipped" in s for s in response.intermediate_steps))

    def test_counter_reset_between_runs(self):
        llm = RoutedMockLLM(["c0", "c1"], prefer="A", instantiation_response="** Answer **: x")
        baseline = RoT(llm, warmup=2)

        r1 = baseline.run("q1")
        # First run: 2 candidates + 1 pairwise + 1 instantiation = 4.
        self.assertEqual(r1.num_llm_calls, 4)

        r2 = baseline.run("q2")
        # Stages 1–2 are cached, so the second run only re-instantiates.
        self.assertEqual(r2.num_llm_calls, 1)

    def test_cpm_result_is_cached_across_runs(self):
        """CPM aggregation is computed once and reused on later runs (fix #3)."""
        llm = RoutedMockLLM(
            ["Task Definition: t"],
            cpm_response="Aggregated",
            instantiation_response="** Answer **: ok",
        )
        emb = MockEmbeddingModel(fixed_score=0.9)
        baseline = RoT(llm, warmup=1, embedding_model=emb, task_prompt="Task")

        r1 = baseline.run("q1")
        # 1 candidate + 1 CPM + 1 instantiation = 3.
        self.assertEqual(r1.num_llm_calls, 3)

        r2 = baseline.run("q2")
        # Candidate + CPM cached; only instantiation runs.
        self.assertEqual(r2.num_llm_calls, 1)
        self.assertEqual(r2.metadata["cpm_boundary"], "known")
        self.assertTrue(any("[cached]" in s for s in r2.intermediate_steps))
        self.assertEqual(emb.call_count, 1)  # embedding only computed once

    def test_intermediate_steps_populated_with_cpm(self):
        llm = RoutedMockLLM(
            ["candidate_def"],
            cpm_response="Aggregated result",
            instantiation_response="** Answer **: done",
        )
        emb = MockEmbeddingModel(fixed_score=0.9)
        baseline = RoT(llm, warmup=1, embedding_model=emb, task_prompt="Task")

        response = baseline.run("test")

        steps = response.intermediate_steps
        self.assertTrue(any("Stage 1" in s for s in steps))
        self.assertTrue(any("Stage 2" in s for s in steps))
        self.assertTrue(any("Stage 3" in s or "CPM" in s for s in steps))
        self.assertTrue(any("Stage 4" in s for s in steps))

    def test_intermediate_steps_without_cpm(self):
        llm = RoutedMockLLM(["candidate_def"], instantiation_response="** Answer **: done")
        baseline = RoT(llm, warmup=1)

        response = baseline.run("test")

        steps = response.intermediate_steps
        self.assertTrue(any("Stage 1" in s for s in steps))
        self.assertTrue(any("Stage 2" in s for s in steps))
        self.assertTrue(any("Skipped" in s for s in steps))
        self.assertTrue(any("Stage 4" in s for s in steps))

    def test_reasoning_trace_includes_cpm_info(self):
        llm = RoutedMockLLM(
            ["Task Definition: Something."],
            cpm_response="Aggregated prompt",
            instantiation_response="** Thinking **: step by step\n\n** Answer **: 42",
        )
        emb = MockEmbeddingModel(fixed_score=0.85)
        baseline = RoT(llm, warmup=1, embedding_model=emb, task_prompt="Original task")

        response = baseline.run("Q")

        self.assertIn("CPM", response.reasoning_trace)
        self.assertIn("known", response.reasoning_trace)
        self.assertIn("0.8500", response.reasoning_trace)
        self.assertIn("P_final", response.reasoning_trace)


class TestRoTEndToEndWithCPMWarmup3(unittest.TestCase):
    """End-to-end with warmup=3 and CPM enabled.

    3 candidate + 3 pairwise + 1 CPM + 1 instantiation = 8 calls.
    Preference always prefers A, so candidate 0 is optimal.
    """

    def test_full_pipeline_warmup_3_with_cpm(self):
        llm = RoutedMockLLM(
            ["Task Definition: A", "Task Definition: B", "Task Definition: C"],
            prefer="A",
            cpm_response="Final refined prompt combining best of both.",
            instantiation_response="** Answer **: 42",
        )
        emb = MockEmbeddingModel(fixed_score=0.75)
        baseline = RoT(
            llm,
            warmup=3,
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Original benchmark prompt",
        )

        response = baseline.run("test question")

        self.assertEqual(response.final_answer, "42")
        self.assertEqual(response.num_llm_calls, 8)
        self.assertEqual(response.total_tokens, 160)
        self.assertEqual(response.metadata["optimal_candidate_index"], 0)
        self.assertEqual(response.metadata["cpm_boundary"], "known")
        self.assertTrue(response.metadata["cpm_enabled"])


if __name__ == "__main__":
    unittest.main()
