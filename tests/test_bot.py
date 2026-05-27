"""
Unit tests for the Buffer of Thoughts (BoT) baseline.

Test organisation
─────────────────
 1.  TestEmbedding                — _embed() helper (SentenceTransformer backend)
 2.  TestCosineSim                — _cosine_sim() helper
 3.  TestThoughtTemplate          — dataclass round-trip
 4.  TestMetaBufferPublicAttrs    — buffer/buffer_path are plain public attrs
 5.  TestMetaBufferRetrieval      — retrieve()
 6.  TestMetaBufferAdd            — add() novelty gate (Eq. 5)
 7.  TestMetaBufferPersistence    — load()/save() + FIX-F empty-file guard
 8.  TestBufferManagerInit        — public attrs; counters NOT @property (FIX-A)
 9.  TestBufferManagerParseTemplate — static parse_template() is public (FIX-C)
10.  TestBufferManagerDistilAndUpdate — distil_and_update() (FIX-B, FIX-C)
11.  TestBoTInitialisation        — public meta_buffer/buffer_manager (FIX-E)
12.  TestBoTPublicStageMethods    — all stage methods are public
13.  TestBoTRunNewTask            — full pipeline, empty buffer
14.  TestBoTRunWithTemplate       — full pipeline, template hit (FIX-D)
15.  TestBoTRunReadOnly           — update_buffer=False
16.  TestBoTCounterAccumulation   — token/call counters across stages
17.  TestBoTBufferGrowth          — multi-question buffer evolution
18.  TestBoTResponseShape         — BaselineResponse fields + metadata keys
19.  TestBoTPersistenceIntegration— templates survive BoT re-instantiation
20.  TestBugRegressions           — explicit regression guard for all six bugs

Reference:
    Yang et al., "Buffer of Thoughts: Thought-Augmented Reasoning with Large
    Language Models", NeurIPS 2024.

Author: (your name)
"""

import json
import os
import tempfile
import unittest
from typing import List

import numpy as np

from models.base import BaseLLM, LLMResponse
from baseline.BoT.bot import (
    BoT,
    BufferManager,
    MetaBuffer,
    ThoughtTemplate,
    _embed,                   # SentenceTransformer embedding backend
    _cosine_sim,              # cosine similarity over embeddings
    extract_distilled_task,   # isolates the '### Distilled Task' query for retrieval
    extract_code,             # pulls a ```python``` block from a response
    run_code,                 # executes generated code in a subprocess
    CodeExecutionResult,
    SEED_TEMPLATES,           # paper Appendix B.1 templates auto-loaded by default
    DISTILLATION_SYSTEM,
    PROBLEM_DISTILLER_SYSTEM,
    INSTANTIATION_SYSTEM,
    NEW_TASK_SYSTEM,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared mock infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

class FixedMockLLM(BaseLLM):
    """Always returns the same response; records every prompt received."""

    def __init__(self, response: str = "", input_tokens: int = 10, output_tokens: int = 20):
        super().__init__(api_key="dummy", model="mock-fixed")
        self._response = response
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.call_count = 0
        self.prompts_received: List[str] = []

    def generate(self, prompt: str, temperature: float = 0, logprobs: bool = False) -> LLMResponse:
        self.call_count += 1
        self.prompts_received.append(prompt)
        return LLMResponse(
            content=self._response,
            model_name="mock-fixed",
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


class SequentialMockLLM(BaseLLM):
    """Returns pre-defined responses in order; empty string once exhausted."""

    def __init__(self, responses: List[str], input_tokens: int = 10, output_tokens: int = 20):
        super().__init__(api_key="dummy", model="mock-seq")
        self._responses = list(responses)
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.call_count = 0
        self.prompts_received: List[str] = []

    def generate(self, prompt: str, temperature: float = 0, logprobs: bool = False) -> LLMResponse:
        content = self._responses[self.call_count] if self.call_count < len(self._responses) else ""
        self.call_count += 1
        self.prompts_received.append(prompt)
        return LLMResponse(
            content=content,
            model_name="mock-seq",
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


# ── canned response strings ───────────────────────────────────────────────────

DISTILLED_INFO = (
    "### Key Information\n- variable x\n\n"
    "### Constraints\n- must be positive\n\n"
    "### Distilled Task\nSolve quadratic equation ax^2+bx+c=0."
)

INSTANTIATION_RESPONSE = (
    "### Reasoning\n"
    "Step 1: D = 25 - 24 = 1\n"
    "Step 2: roots = (5 \u00b1 1) / 2\n\n"
    "### Answer\nx = 3 or x = 2"
)

NEW_TASK_RESPONSE = (
    "### Reasoning Structure Chosen\nPrompt-based\n\n"
    "### Reasoning\nProceed step by step.\n\n"
    "### Answer\n42"
)

DISTILLATION_JSON = json.dumps({
    "category": "Mathematical Reasoning",
    "description": "Solve quadratic equations by computing the discriminant.",
    "template": "Step 1: compute D = b^2 - 4ac.\nStep 2: compute roots.",
})


def _make_template(idx: int = 0, description: str = "quadratic equation solve discriminant") -> ThoughtTemplate:
    return ThoughtTemplate(
        index=idx,
        category="Mathematical Reasoning",
        description=description,
        template="Step 1: compute D.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. _embed — SentenceTransformer embedding backend
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbedding(unittest.TestCase):

    def test_returns_1d_float_array(self):
        v = _embed("solve a quadratic equation")
        self.assertIsInstance(v, np.ndarray)
        self.assertEqual(v.ndim, 1)
        self.assertGreater(v.shape[0], 0)

    def test_l2_normalised(self):
        # _embed normalises embeddings, so the L2 norm must be ~1.0.
        v = _embed("any sentence here")
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=4)

    def test_deterministic(self):
        a = _embed("chess board checkmate in one move")
        b = _embed("chess board checkmate in one move")
        self.assertTrue(np.allclose(a, b))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _cosine_sim — cosine similarity over embeddings
# ═══════════════════════════════════════════════════════════════════════════════

class TestCosineSim(unittest.TestCase):

    def test_identical_text_near_one(self):
        v = _embed("solve quadratic equation discriminant roots")
        self.assertAlmostEqual(_cosine_sim(v, v), 1.0, places=4)

    def test_related_beats_unrelated(self):
        query = _embed("solve a quadratic equation")
        related = _embed("compute the roots of ax^2 + bx + c = 0")
        unrelated = _embed("bake a chocolate cake with flour and sugar")
        self.assertGreater(_cosine_sim(query, related), _cosine_sim(query, unrelated))

    def test_in_unit_range(self):
        a = _embed("a list of numbers and arithmetic operations")
        b = _embed("write a sonnet with a rhyme scheme")
        score = _cosine_sim(a, b)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_symmetric(self):
        a = _embed("update the chess board state")
        b = _embed("infer the date from a description")
        self.assertAlmostEqual(_cosine_sim(a, b), _cosine_sim(b, a), places=6)


# ═══════════════════════════════════════════════════════════════════════════════
# 2b. extract_distilled_task — retrieval query isolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractDistilledTask(unittest.TestCase):

    def test_returns_only_task_section(self):
        out = extract_distilled_task(DISTILLED_INFO)
        self.assertEqual(out, "Solve quadratic equation ax^2+bx+c=0.")
        self.assertNotIn("Key Information", out)
        self.assertNotIn("Constraints", out)

    def test_stops_at_next_header(self):
        text = (
            "### Distilled Task\nCombine numbers to reach a target.\n\n"
            "### Notes\nignore me"
        )
        self.assertEqual(extract_distilled_task(text), "Combine numbers to reach a target.")

    def test_case_insensitive_header(self):
        text = "### distilled task\nlower case header body"
        self.assertEqual(extract_distilled_task(text), "lower case header body")

    def test_fallback_to_full_text_when_absent(self):
        text = "no headers at all, just a plain question about chess."
        self.assertEqual(extract_distilled_task(text), text)

    def test_fallback_when_section_empty(self):
        text = "### Distilled Task\n\n### Constraints\n- something"
        # Empty task body → fall back to the whole text rather than returning "".
        self.assertEqual(extract_distilled_task(text), text.strip())

    def test_isolated_query_improves_match_when_noise_is_offtopic(self):
        # When the Key Information / Constraints sections are off-topic noise,
        # embedding the whole dump drags similarity to the relevant description
        # down; isolating the task paragraph recovers it. (Isolation is not a
        # universal win — if the noise happens to share keywords it can help the
        # full dump — but for off-topic noise it should clearly improve.)
        full = (
            "### Key Information\n- assorted trivia about weather, traffic and lunch\n"
            "### Constraints\n- respond politely using complete sentences\n"
            "### Distilled Task\nWrite a sonnet that follows a fixed rhyme scheme."
        )
        desc = _embed("Generate a sonnet following a rhyme scheme with required words.")
        sim_full = _cosine_sim(_embed(full), desc)
        sim_task = _cosine_sim(_embed(extract_distilled_task(full)), desc)
        self.assertGreater(sim_task, sim_full)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ThoughtTemplate
# ═══════════════════════════════════════════════════════════════════════════════

class TestThoughtTemplate(unittest.TestCase):

    def _make(self, **kw):
        base = dict(index=0, category="Mathematical Reasoning",
                    description="quadratic solver", template="Step 1")
        base.update(kw)
        return ThoughtTemplate(**base)

    def test_to_dict_roundtrip(self):
        t = self._make(index=7)
        t2 = ThoughtTemplate.from_dict(t.to_dict())
        self.assertEqual(t.index, t2.index)
        self.assertEqual(t.category, t2.category)
        self.assertEqual(t.description, t2.description)
        self.assertEqual(t.template, t2.template)

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        for key in ("index", "category", "description", "template"):
            self.assertIn(key, d)

    def test_from_dict_type(self):
        t = ThoughtTemplate.from_dict(
            {"index": 3, "category": "Code Programming",
             "description": "sort", "template": "def sort():"}
        )
        self.assertIsInstance(t, ThoughtTemplate)
        self.assertEqual(t.index, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MetaBuffer — public attributes
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetaBufferPublicAttrs(unittest.TestCase):
    """buffer and buffer_path must be plain public attributes, not properties."""

    def test_buffer_is_list(self):
        self.assertIsInstance(MetaBuffer().buffer, list)

    def test_buffer_path_none_by_default(self):
        self.assertIsNone(MetaBuffer().buffer_path)

    def test_buffer_path_stored(self):
        mb = MetaBuffer(buffer_path="/tmp/x.json")
        self.assertEqual(mb.buffer_path, "/tmp/x.json")

    def test_buffer_directly_assignable(self):
        mb = MetaBuffer()
        new_list = [_make_template(0)]
        mb.buffer = new_list
        self.assertIs(mb.buffer, new_list)

    def test_load_and_save_are_public(self):
        mb = MetaBuffer()
        self.assertTrue(callable(getattr(mb, "load", None)))
        self.assertTrue(callable(getattr(mb, "save", None)))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MetaBuffer — retrieve
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetaBufferRetrieval(unittest.TestCase):

    def test_empty_buffer_returns_none(self):
        self.assertIsNone(MetaBuffer().retrieve("any query", threshold=0.5))

    def test_exact_match_above_threshold(self):
        desc = "quadratic equation discriminant roots"
        mb = MetaBuffer(init_templates=[_make_template(0, desc)])
        self.assertIsNotNone(mb.retrieve(desc, threshold=0.5))

    def test_dissimilar_query_returns_none(self):
        mb = MetaBuffer(init_templates=[_make_template(0, "chess board checkmate")])
        self.assertIsNone(mb.retrieve("baking chocolate cake flour", threshold=0.5))

    def test_returns_best_among_multiple(self):
        templates = [
            ThoughtTemplate(0, "Mathematical Reasoning", "quadratic algebra roots solve", "T1"),
            ThoughtTemplate(1, "Code Programming", "sorting algorithm binary search list", "T2"),
        ]
        mb = MetaBuffer(init_templates=templates)
        result = mb.retrieve("solve quadratic algebra roots", threshold=0.3)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "Mathematical Reasoning")

    def test_threshold_boundary(self):
        mb = MetaBuffer(init_templates=[_make_template(0, "apple banana cherry")])
        self.assertIsNotNone(mb.retrieve("apple banana cherry", threshold=0.99))
        self.assertIsNone(mb.retrieve("apple banana cherry", threshold=1.01))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MetaBuffer — add
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetaBufferAdd(unittest.TestCase):

    def test_first_template_added(self):
        mb = MetaBuffer()
        self.assertTrue(mb.add(_make_template(description="novel unique phrase here"), threshold=0.9))
        self.assertEqual(mb.size, 1)

    def test_index_assigned_as_zero(self):
        mb = MetaBuffer()
        t = _make_template(idx=-1, description="first entry alpha")
        mb.add(t, threshold=0.9)
        self.assertEqual(mb.buffer[0].index, 0)

    def test_duplicate_rejected(self):
        mb = MetaBuffer()
        desc = "quadratic equation discriminant formula roots"
        mb.add(_make_template(description=desc), threshold=0.5)
        self.assertFalse(mb.add(_make_template(description=desc), threshold=0.5))
        self.assertEqual(mb.size, 1)

    def test_novel_template_accepted(self):
        mb = MetaBuffer()
        mb.add(_make_template(description="quadratic math algebra"), threshold=0.5)
        novel = ThoughtTemplate(-1, "Code Programming", "sorting binary search algorithm", "T")
        self.assertTrue(mb.add(novel, threshold=0.5))
        self.assertEqual(mb.size, 2)

    def test_sequential_indices(self):
        mb = MetaBuffer()
        # High threshold (0.9) so the novelty gate admits all three distinct topics:
        # under semantic embeddings unrelated texts still have a small positive
        # similarity, so a low gate would wrongly reject them.
        for desc in ("apple orchard fruit", "chess piece board", "integral calculus derivative"):
            mb.add(ThoughtTemplate(-1, "Common Sense Reasoning", desc, "T"), threshold=0.9)
        self.assertEqual([t.index for t in mb.buffer], [0, 1, 2])

    def test_all_templates_returns_copy(self):
        mb = MetaBuffer(init_templates=[_make_template(0)])
        lst = mb.all_templates()
        lst.clear()
        self.assertEqual(mb.size, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MetaBuffer — persistence + FIX-F
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetaBufferPersistence(unittest.TestCase):

    def test_save_and_reload(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mb1 = MetaBuffer(buffer_path=path)
            mb1.add(_make_template(description="unique alpha topic entry"), threshold=0.9)
            mb1.add(ThoughtTemplate(-1, "Code Programming", "sorting algorithms data", "T2"), threshold=0.9)
            mb2 = MetaBuffer(buffer_path=path)
            self.assertEqual(mb2.size, 2)
            self.assertEqual(mb2.size, mb1.size)
        finally:
            os.unlink(path)

    def test_load_preexisting_file(self):
        data = [{"index": 0, "category": "Mathematical Reasoning",
                 "description": "quadratic", "template": "Step 1"}]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            path = f.name
        try:
            mb = MetaBuffer(buffer_path=path)
            self.assertEqual(mb.size, 1)
            self.assertEqual(mb.buffer[0].description, "quadratic")
        finally:
            os.unlink(path)

    def test_fix_f_empty_file_does_not_raise(self):
        """FIX-F: MetaBuffer.load() must not raise JSONDecodeError on an empty file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mb = MetaBuffer(buffer_path=path)   # must NOT raise
            self.assertEqual(mb.size, 0)
        finally:
            os.unlink(path)

    def test_no_path_in_memory_only(self):
        mb = MetaBuffer()
        mb.add(_make_template(description="ephemeral content here"), threshold=0.1)
        self.assertEqual(mb.size, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. BufferManager — init + FIX-A
# ═══════════════════════════════════════════════════════════════════════════════

class TestBufferManagerInit(unittest.TestCase):

    def _make(self):
        return BufferManager(MetaBuffer(), FixedMockLLM(),
                             similarity_threshold=0.5, distill_temperature=0.3)

    def test_public_attributes(self):
        bm = self._make()
        self.assertIsInstance(bm.meta_buffer, MetaBuffer)
        self.assertIsInstance(bm.llm, FixedMockLLM)
        self.assertAlmostEqual(bm.threshold, 0.5)
        self.assertAlmostEqual(bm.distill_temperature, 0.3)

    def test_fix_a_counters_are_plain_ints(self):
        """FIX-A: counters must NOT be @property — they shadow instance attrs."""
        bm = self._make()
        bm.total_input_tokens = 5
        bm.total_output_tokens = 7
        bm.num_calls = 2
        self.assertEqual(bm.total_input_tokens, 5)
        self.assertEqual(bm.total_output_tokens, 7)
        self.assertEqual(bm.num_calls, 2)

    def test_fix_a_no_recursion_on_read(self):
        """Accessing counter before assignment must not cause RecursionError."""
        bm = self._make()
        try:
            _ = bm.total_input_tokens + bm.total_output_tokens + bm.num_calls
        except RecursionError:
            self.fail("FIX-A: @property shadowing instance attr causes RecursionError")

    def test_counters_start_at_zero(self):
        bm = self._make()
        self.assertEqual(bm.total_input_tokens, 0)
        self.assertEqual(bm.total_output_tokens, 0)
        self.assertEqual(bm.num_calls, 0)

    def test_reset_counters(self):
        bm = self._make()
        bm.total_input_tokens = 99
        bm.total_output_tokens = 88
        bm.num_calls = 7
        bm.reset_counters()
        self.assertEqual(bm.total_input_tokens, 0)
        self.assertEqual(bm.total_output_tokens, 0)
        self.assertEqual(bm.num_calls, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. BufferManager.parse_template (FIX-C: public name)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBufferManagerParseTemplate(unittest.TestCase):

    def test_is_public_static_method(self):
        self.assertTrue(callable(BufferManager.parse_template))

    def test_valid_json(self):
        result = BufferManager.parse_template(DISTILLATION_JSON)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "Mathematical Reasoning")

    def test_markdown_fences_stripped(self):
        self.assertIsNotNone(
            BufferManager.parse_template(f"```json\n{DISTILLATION_JSON}\n```")
        )

    def test_json_in_prose(self):
        self.assertIsNotNone(
            BufferManager.parse_template(f"Output:\n{DISTILLATION_JSON}\nDone.")
        )

    def test_invalid_json_returns_none(self):
        self.assertIsNone(BufferManager.parse_template("not json"))

    def test_missing_description_returns_none(self):
        bad = json.dumps({"category": "Mathematical Reasoning", "template": "Step 1"})
        self.assertIsNone(BufferManager.parse_template(bad))

    def test_missing_template_returns_none(self):
        bad = json.dumps({"category": "Mathematical Reasoning", "description": "solve"})
        self.assertIsNone(BufferManager.parse_template(bad))

    def test_index_initialised_minus_one(self):
        result = BufferManager.parse_template(DISTILLATION_JSON)
        self.assertEqual(result.index, -1)

    def test_private_name_does_not_exist(self):
        bm = BufferManager(MetaBuffer(), FixedMockLLM())
        self.assertFalse(hasattr(bm, "_parse_template"),
                         "_parse_template must not exist; use parse_template (FIX-C)")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. BufferManager.distil_and_update (FIX-B, FIX-C)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBufferManagerDistilAndUpdate(unittest.TestCase):

    def _make(self, response: str, threshold: float = 0.3):
        mb = MetaBuffer()
        llm = FixedMockLLM(response=response)
        bm = BufferManager(mb, llm, similarity_threshold=threshold, distill_temperature=0.2)
        return bm, mb, llm

    def test_fix_b_prompt_references_distillation_system(self):
        """FIX-B: prompt must contain DISTILLATION_SYSTEM not _DISTILLATION_SYSTEM."""
        bm, _, llm = self._make(DISTILLATION_JSON)
        bm.distil_and_update(DISTILLED_INFO, INSTANTIATION_RESPONSE)
        self.assertIn(DISTILLATION_SYSTEM, llm.prompts_received[0])

    def test_valid_json_adds_template(self):
        bm, mb, _ = self._make(DISTILLATION_JSON)
        result = bm.distil_and_update(DISTILLED_INFO, INSTANTIATION_RESPONSE)
        self.assertIsNotNone(result)
        self.assertEqual(mb.size, 1)

    def test_invalid_json_returns_none(self):
        bm, mb, _ = self._make("not json")
        self.assertIsNone(bm.distil_and_update(DISTILLED_INFO, INSTANTIATION_RESPONSE))
        self.assertEqual(mb.size, 0)

    def test_duplicate_rejected(self):
        bm, mb, _ = self._make(DISTILLATION_JSON, threshold=0.3)
        bm.distil_and_update(DISTILLED_INFO, INSTANTIATION_RESPONSE)
        bm.distil_and_update(DISTILLED_INFO, INSTANTIATION_RESPONSE)
        self.assertEqual(mb.size, 1)

    def test_counters_incremented(self):
        bm, _, llm = self._make(DISTILLATION_JSON)
        bm.distil_and_update(DISTILLED_INFO, INSTANTIATION_RESPONSE)
        self.assertEqual(bm.total_input_tokens, llm._input_tokens)
        self.assertEqual(bm.total_output_tokens, llm._output_tokens)
        self.assertEqual(bm.num_calls, 1)

    def test_counters_accumulate(self):
        bm, _, llm = self._make(DISTILLATION_JSON, threshold=0.1)
        bm.distil_and_update(DISTILLED_INFO, INSTANTIATION_RESPONSE)
        bm.distil_and_update(
            "### Distilled Task\nSolve chess checkmate problem.",
            "### Answer\nKnight to e5."
        )
        self.assertEqual(bm.num_calls, 2)
        self.assertEqual(bm.total_input_tokens, llm._input_tokens * 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. BoT initialisation (FIX-E)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoTInitialisation(unittest.TestCase):

    def test_default_values(self):
        bot = BoT(FixedMockLLM())
        self.assertEqual(bot.baseline_name, "BoT")
        self.assertAlmostEqual(bot.similarity_threshold, 0.5)
        self.assertAlmostEqual(bot.distill_temperature, 0.2)
        self.assertAlmostEqual(bot.instantiation_temperature, 0.1)
        self.assertTrue(bot.update_buffer)

    def test_custom_values(self):
        bot = BoT(FixedMockLLM(), similarity_threshold=0.7,
                  distill_temperature=0.5, instantiation_temperature=0.3,
                  update_buffer=False)
        self.assertAlmostEqual(bot.similarity_threshold, 0.7)
        self.assertFalse(bot.update_buffer)

    def test_fix_e_meta_buffer_is_public(self):
        """FIX-E: meta_buffer must be a plain public attribute."""
        bot = BoT(FixedMockLLM())
        self.assertIsInstance(bot.meta_buffer, MetaBuffer)

    def test_fix_e_buffer_manager_is_public(self):
        """FIX-E: buffer_manager must be a plain public attribute."""
        bot = BoT(FixedMockLLM())
        self.assertIsInstance(bot.buffer_manager, BufferManager)

    def test_fix_e_no_private_underscore_attrs(self):
        """FIX-E: _meta_buffer and _buffer_manager must NOT exist."""
        bot = BoT(FixedMockLLM())
        self.assertFalse(hasattr(bot, "_meta_buffer"))
        self.assertFalse(hasattr(bot, "_buffer_manager"))

    def test_default_seeds_paper_templates(self):
        # With no buffer_path and no init_templates, BoT auto-loads the paper's
        # Appendix B.1 thought-templates so retrieval has something to match from.
        self.assertEqual(BoT(FixedMockLLM()).meta_buffer.size, len(SEED_TEMPLATES))

    def test_empty_buffer_with_explicit_empty_init(self):
        # Passing an explicit empty list opts out of the paper seeds.
        self.assertEqual(BoT(FixedMockLLM(), init_templates=[]).meta_buffer.size, 0)

    def test_init_templates_seeded(self):
        templates = [_make_template(0, "prime sieve"), _make_template(1, "chess opening")]
        bot = BoT(FixedMockLLM(), init_templates=templates)
        self.assertEqual(bot.meta_buffer.size, 2)

    def test_buffer_manager_shares_meta_buffer(self):
        bot = BoT(FixedMockLLM())
        self.assertIs(bot.buffer_manager.meta_buffer, bot.meta_buffer)

    def test_repr_contains_key_fields(self):
        bot = BoT(FixedMockLLM(), similarity_threshold=0.55, update_buffer=False)
        r = repr(bot)
        self.assertIn("BoT", r)
        self.assertIn("0.55", r)
        self.assertIn("False", r)

    def test_repr_does_not_raise(self):
        try:
            _ = repr(BoT(FixedMockLLM()))
        except AttributeError as e:
            self.fail(f"repr raised AttributeError (FIX-D/E): {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. BoT public stage methods
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoTPublicStageMethods(unittest.TestCase):

    def test_distil_problem_public(self):
        bot = BoT(FixedMockLLM(response=DISTILLED_INFO), update_buffer=False)
        self.assertEqual(bot.distil_problem("x^2=4"), DISTILLED_INFO)
        self.assertFalse(hasattr(bot, "_distil_problem"))

    def test_distil_problem_question_in_prompt(self):
        question = "Find all prime factors of 360."
        llm = FixedMockLLM(response=DISTILLED_INFO)
        BoT(llm, update_buffer=False).distil_problem(question)
        self.assertIn(question, llm.prompts_received[0])

    def test_distil_problem_strips_whitespace(self):
        bot = BoT(FixedMockLLM(response="  spaced  "), update_buffer=False)
        self.assertEqual(bot.distil_problem("q"), "spaced")

    def test_retrieve_template_public(self):
        bot = BoT(FixedMockLLM(), update_buffer=False, init_templates=[])
        self.assertIsNone(bot.retrieve_template("any query"))
        self.assertFalse(hasattr(bot, "_retrieve_template"))

    def test_instantiate_with_template_public(self):
        llm = FixedMockLLM(response=INSTANTIATION_RESPONSE)
        bot = BoT(llm, update_buffer=False)
        result = bot.instantiate_with_template(DISTILLED_INFO, _make_template())
        self.assertEqual(result, INSTANTIATION_RESPONSE)
        self.assertFalse(hasattr(bot, "_instantiate_with_template"))

    def test_instantiate_with_template_embeds_template(self):
        llm = FixedMockLLM(response=INSTANTIATION_RESPONSE)
        bot = BoT(llm, update_buffer=False)
        tmpl = _make_template()
        bot.instantiate_with_template(DISTILLED_INFO, tmpl)
        self.assertIn(tmpl.template, llm.prompts_received[0])

    def test_instantiate_new_task_public(self):
        llm = FixedMockLLM(response=NEW_TASK_RESPONSE)
        bot = BoT(llm, update_buffer=False)
        self.assertEqual(bot.instantiate_new_task(DISTILLED_INFO), NEW_TASK_RESPONSE)
        self.assertFalse(hasattr(bot, "_instantiate_new_task"))

    def test_instantiate_new_task_no_template_header(self):
        llm = FixedMockLLM(response=NEW_TASK_RESPONSE)
        bot = BoT(llm, update_buffer=False)
        bot.instantiate_new_task(DISTILLED_INFO)
        self.assertNotIn("[Thought Template", llm.prompts_received[0])

    def test_extract_answer_public_static(self):
        self.assertTrue(callable(BoT.extract_answer))
        self.assertFalse(hasattr(BoT, "_extract_answer"))

    def test_extract_answer_finds_section(self):
        self.assertEqual(BoT.extract_answer("### Reasoning\nmath\n\n### Answer\nx = 3"), "x = 3")

    def test_extract_answer_stops_at_next_header(self):
        raw = "### Reasoning\nstuff\n\n### Answer\nx = 3\n\n### Notes\nextra"
        self.assertEqual(BoT.extract_answer(raw), "x = 3")

    def test_extract_answer_fallback_last_line(self):
        self.assertEqual(BoT.extract_answer("No headers\nThe result is 42"), "The result is 42")

    def test_extract_answer_case_insensitive(self):
        self.assertEqual(BoT.extract_answer("### answer\nresult = 7"), "result = 7")

    def test_extract_answer_empty_string(self):
        self.assertIsInstance(BoT.extract_answer(""), str)

    def test_extract_answer_multiline(self):
        ans = BoT.extract_answer("### Answer\nLine one\nLine two")
        self.assertIn("Line one", ans)
        self.assertIn("Line two", ans)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. BoT.run() — new-task path
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoTRunNewTask(unittest.TestCase):

    def _make(self):
        # Empty buffer (init_templates=[]) so the task is genuinely "new" and the
        # paper seeds cannot accidentally match DISTILLED_INFO's quadratic topic.
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE, DISTILLATION_JSON])
        return BoT(llm, similarity_threshold=0.5, update_buffer=True, init_templates=[]), llm

    def test_final_answer(self):
        bot, _ = self._make()
        self.assertEqual(bot.run("Q").final_answer, "42")

    def test_baseline_type(self):
        bot, _ = self._make()
        self.assertEqual(bot.run("Q").baseline_type, "BoT")

    def test_is_new_task_true(self):
        bot, _ = self._make()
        self.assertTrue(bot.run("Q").metadata["is_new_task"])

    def test_retrieved_index_none(self):
        bot, _ = self._make()
        self.assertIsNone(bot.run("Q").metadata["retrieved_template_index"])

    def test_buffer_grows(self):
        bot, _ = self._make()
        bot.run("Q")
        self.assertEqual(bot.meta_buffer.size, 1)

    def test_three_llm_calls(self):
        bot, llm = self._make()
        bot.run("Q")
        self.assertEqual(llm.call_count, 3)

    def test_all_four_stages_present(self):
        bot, _ = self._make()
        joined = "\n".join(bot.run("Q").intermediate_steps)
        for stage in ("Stage 1", "Stage 2", "Stage 3", "Stage 4"):
            self.assertIn(stage, joined)

    def test_reasoning_trace_non_empty(self):
        bot, _ = self._make()
        self.assertGreater(len(bot.run("Q").reasoning_trace), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. BoT.run() — template-hit path (FIX-D)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoTRunWithTemplate(unittest.TestCase):

    def _make(self):
        seeded = [ThoughtTemplate(
            0, "Mathematical Reasoning",
            "solve quadratic equation distilled task discriminant roots",
            "Step 1: compute D=b^2-4ac.\nStep 2: compute roots.",
        )]
        llm = SequentialMockLLM([DISTILLED_INFO, INSTANTIATION_RESPONSE, DISTILLATION_JSON])
        bot = BoT(llm, similarity_threshold=0.25, update_buffer=True, init_templates=seeded)
        return bot, llm

    def test_answer_extracted(self):
        bot, _ = self._make()
        self.assertEqual(
            bot.run("quadratic equation discriminant roots").final_answer,
            "x = 3 or x = 2",
        )

    def test_is_new_task_false(self):
        bot, _ = self._make()
        self.assertFalse(bot.run("quadratic equation discriminant roots").metadata["is_new_task"])

    def test_retrieved_index_set(self):
        bot, _ = self._make()
        idx = bot.run("quadratic equation discriminant roots").metadata["retrieved_template_index"]
        self.assertIsNotNone(idx)
        self.assertIsInstance(idx, int)

    def test_template_category_in_metadata(self):
        bot, _ = self._make()
        self.assertEqual(
            bot.run("quadratic equation discriminant roots").metadata["retrieved_template_category"],
            "Mathematical Reasoning",
        )

    def test_template_retrieved_in_steps(self):
        bot, _ = self._make()
        self.assertIn("Template Retrieved",
                      "\n".join(bot.run("quadratic equation discriminant roots").intermediate_steps))

    def test_fix_d_no_attribute_error(self):
        """FIX-D: run() must not use self._buffer_manager / self._meta_buffer."""
        bot, _ = self._make()
        try:
            bot.run("quadratic equation discriminant roots")
        except AttributeError as e:
            self.fail(f"run() raised AttributeError (FIX-D): {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 15. BoT.run() — read-only mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoTRunReadOnly(unittest.TestCase):

    def test_buffer_frozen(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE])
        bot = BoT(llm, update_buffer=False, init_templates=[])
        bot.run("Q")
        self.assertEqual(bot.meta_buffer.size, 0)

    def test_two_calls_only(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE])
        bot = BoT(llm, update_buffer=False)
        bot.run("Q")
        self.assertEqual(llm.call_count, 2)

    def test_stage_4_absent(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE])
        resp = BoT(llm, update_buffer=False).run("Q")
        self.assertNotIn("Stage 4", "\n".join(resp.intermediate_steps))

    def test_update_buffer_false_in_metadata(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE])
        self.assertFalse(BoT(llm, update_buffer=False).run("Q").metadata["update_buffer"])


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Counter accumulation
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoTCounterAccumulation(unittest.TestCase):

    def test_three_calls_with_update(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE, DISTILLATION_JSON],
                                 input_tokens=5, output_tokens=8)
        self.assertEqual(BoT(llm, update_buffer=True).run("Q").num_llm_calls, 3)

    def test_two_calls_read_only(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE],
                                 input_tokens=5, output_tokens=8)
        self.assertEqual(BoT(llm, update_buffer=False).run("Q").num_llm_calls, 2)

    def test_token_arithmetic(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE, DISTILLATION_JSON],
                                 input_tokens=5, output_tokens=8)
        resp = BoT(llm, update_buffer=True).run("Q")
        self.assertEqual(resp.total_input_tokens, 15)
        self.assertEqual(resp.total_output_tokens, 24)
        self.assertEqual(resp.total_tokens, 39)

    def test_counters_reset_between_runs(self):
        llm = SequentialMockLLM(
            [DISTILLED_INFO, NEW_TASK_RESPONSE, DISTILLATION_JSON] * 2,
            input_tokens=5, output_tokens=8,
        )
        bot = BoT(llm, update_buffer=True)
        r1 = bot.run("Q1")
        r2 = bot.run("Q2")
        self.assertEqual(r1.num_llm_calls, 3)
        self.assertEqual(r2.num_llm_calls, 3)

    def test_total_tokens_property(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE],
                                 input_tokens=7, output_tokens=13)
        resp = BoT(llm, update_buffer=False).run("Q")
        self.assertEqual(resp.total_tokens, resp.total_input_tokens + resp.total_output_tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# 17. Buffer growth across questions
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoTBufferGrowth(unittest.TestCase):

    def test_diverse_questions_grow_buffer(self):
        responses = [
            "### Distilled Task\nSolve quadratic equation algebra math.",
            NEW_TASK_RESPONSE,
            json.dumps({"category": "Mathematical Reasoning",
                        "description": "quadratic algebra equation solving",
                        "template": "Step 1: discriminant"}),
            "### Distilled Task\nWrite a sonnet poem with rhyme scheme.",
            NEW_TASK_RESPONSE,
            json.dumps({"category": "Creative Language Generation",
                        "description": "sonnet poem writing rhyme creative",
                        "template": "Step 1: pick theme"}),
        ]
        bot = BoT(SequentialMockLLM(responses), similarity_threshold=0.5,
                  update_buffer=True, init_templates=[])
        bot.run("Q1")
        bot.run("Q2")
        self.assertEqual(bot.meta_buffer.size, 2)

    def test_duplicate_topics_do_not_grow_buffer(self):
        d = json.dumps({"category": "Mathematical Reasoning",
                        "description": "quadratic equation algebra discriminant roots",
                        "template": "Step 1: discriminant"})
        bot = BoT(
            SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE, d,
                                DISTILLED_INFO, INSTANTIATION_RESPONSE, d]),
            similarity_threshold=0.3, update_buffer=True, init_templates=[],
        )
        bot.run("Q1 quadratic equation discriminant")
        bot.run("Q2 quadratic equation discriminant")
        self.assertEqual(bot.meta_buffer.size, 1)

    def test_meta_buffer_size_in_metadata(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE, DISTILLATION_JSON])
        bot = BoT(llm, update_buffer=True)
        resp = bot.run("Q")
        self.assertEqual(resp.metadata["meta_buffer_size"], bot.meta_buffer.size)


# ═══════════════════════════════════════════════════════════════════════════════
# 18. BaselineResponse shape
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoTResponseShape(unittest.TestCase):

    def setUp(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE, DISTILLATION_JSON])
        self.resp = BoT(llm, update_buffer=True).run("any question")

    def test_final_answer_str(self):
        self.assertIsInstance(self.resp.final_answer, str)
        self.assertGreater(len(self.resp.final_answer), 0)

    def test_reasoning_trace_str(self):
        self.assertIsInstance(self.resp.reasoning_trace, str)

    def test_intermediate_steps_list(self):
        self.assertIsInstance(self.resp.intermediate_steps, list)

    def test_required_metadata_keys(self):
        for key in ("is_new_task", "retrieved_template_index", "retrieved_template_category",
                    "new_template_added", "meta_buffer_size", "similarity_threshold",
                    "instantiation_temperature", "distill_temperature", "update_buffer"):
            self.assertIn(key, self.resp.metadata)

    def test_json_serialisable(self):
        json.dumps(self.resp.to_dict())  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 19. Persistence integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoTPersistenceIntegration(unittest.TestCase):

    def test_templates_survive_reinitialisation(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            llm1 = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE, DISTILLATION_JSON])
            bot1 = BoT(llm1, similarity_threshold=0.3, buffer_path=path, update_buffer=True)
            bot1.run("quadratic equation discriminant algebra")
            size = bot1.meta_buffer.size

            llm2 = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE, DISTILLATION_JSON])
            bot2 = BoT(llm2, similarity_threshold=0.3, buffer_path=path, update_buffer=True)
            self.assertEqual(bot2.meta_buffer.size, size)
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# 20. Bug regression tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBugRegressions(unittest.TestCase):

    # FIX-A ───────────────────────────────────────────────────────────────────

    def test_fix_a_counter_read_no_recursion(self):
        bm = BufferManager(MetaBuffer(), FixedMockLLM())
        try:
            _ = bm.total_input_tokens + bm.total_output_tokens + bm.num_calls
        except RecursionError:
            self.fail("FIX-A: @property shadowing instance attr → RecursionError")

    def test_fix_a_counter_increment_no_recursion(self):
        bm = BufferManager(MetaBuffer(), FixedMockLLM(DISTILLATION_JSON))
        bm.distil_and_update(DISTILLED_INFO, INSTANTIATION_RESPONSE)
        try:
            self.assertEqual(bm.num_calls, 1)
        except RecursionError:
            self.fail("FIX-A: reading num_calls after increment → RecursionError")

    # FIX-B ───────────────────────────────────────────────────────────────────

    def test_fix_b_no_name_error(self):
        llm = FixedMockLLM(DISTILLATION_JSON)
        bm = BufferManager(MetaBuffer(), llm)
        try:
            bm.distil_and_update(DISTILLED_INFO, INSTANTIATION_RESPONSE)
        except NameError as e:
            self.fail(f"FIX-B: NameError for _DISTILLATION_SYSTEM: {e}")

    def test_fix_b_prompt_has_distillation_system_text(self):
        llm = FixedMockLLM(DISTILLATION_JSON)
        bm = BufferManager(MetaBuffer(), llm)
        bm.distil_and_update(DISTILLED_INFO, INSTANTIATION_RESPONSE)
        self.assertIn(DISTILLATION_SYSTEM, llm.prompts_received[0])

    # FIX-C ───────────────────────────────────────────────────────────────────

    def test_fix_c_parse_template_accessible(self):
        try:
            result = BufferManager.parse_template(DISTILLATION_JSON)
        except AttributeError:
            self.fail("FIX-C: BufferManager.parse_template not accessible")
        self.assertIsNotNone(result)

    def test_fix_c_distil_uses_parse_template_not_private(self):
        llm = FixedMockLLM(DISTILLATION_JSON)
        bm = BufferManager(MetaBuffer(), llm)
        try:
            bm.distil_and_update(DISTILLED_INFO, INSTANTIATION_RESPONSE)
        except AttributeError as e:
            if "_parse_template" in str(e):
                self.fail("FIX-C: distil_and_update calls self._parse_template (use parse_template)")
            raise

    # FIX-D ───────────────────────────────────────────────────────────────────

    def test_fix_d_run_no_attribute_error_buffer_manager(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE, DISTILLATION_JSON])
        bot = BoT(llm, update_buffer=True)
        try:
            bot.run("Q")
        except AttributeError as e:
            if "_buffer_manager" in str(e):
                self.fail("FIX-D: run() uses self._buffer_manager; use self.buffer_manager")
            raise

    def test_fix_d_metadata_no_attribute_error_meta_buffer(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE, DISTILLATION_JSON])
        bot = BoT(llm, update_buffer=True)
        try:
            resp = bot.run("Q")
            _ = resp.metadata["meta_buffer_size"]
        except AttributeError as e:
            if "_meta_buffer" in str(e):
                self.fail("FIX-D: run() uses self._meta_buffer; use self.meta_buffer")
            raise

    # FIX-E ───────────────────────────────────────────────────────────────────

    def test_fix_e_meta_buffer_no_attribute_error(self):
        bot = BoT(FixedMockLLM())
        try:
            mb = bot.meta_buffer
        except AttributeError as e:
            self.fail(f"FIX-E: bot.meta_buffer raised AttributeError: {e}")
        self.assertIsInstance(mb, MetaBuffer)

    def test_fix_e_buffer_manager_no_attribute_error(self):
        bot = BoT(FixedMockLLM())
        try:
            bm = bot.buffer_manager
        except AttributeError as e:
            self.fail(f"FIX-E: bot.buffer_manager raised AttributeError: {e}")
        self.assertIsInstance(bm, BufferManager)

    # FIX-F ───────────────────────────────────────────────────────────────────

    def test_fix_f_empty_file_no_json_error(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            try:
                mb = MetaBuffer(buffer_path=path)
            except Exception as e:
                self.fail(f"FIX-F: MetaBuffer raised on empty file: {e}")
            self.assertEqual(mb.size, 0)
        finally:
            os.unlink(path)

    def test_fix_f_bot_with_empty_buffer_path(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            try:
                bot = BoT(FixedMockLLM(), buffer_path=path, update_buffer=False)
            except Exception as e:
                self.fail(f"FIX-F: BoT raised on empty buffer file: {e}")
            self.assertEqual(bot.meta_buffer.size, 0)
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# 21. Code extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractCode(unittest.TestCase):

    def test_extracts_fenced_python(self):
        text = "Here is the program:\n```python\nprint('hi')\n```\nDone."
        self.assertEqual(extract_code(text), "print('hi')")

    def test_language_tag_optional(self):
        self.assertEqual(extract_code("```\nx = 1\nprint(x)\n```"), "x = 1\nprint(x)")

    def test_no_fence_returns_none(self):
        self.assertIsNone(extract_code("just prose, no code at all"))

    def test_picks_longest_block(self):
        text = "```python\nprint(1)\n```\nand\n```python\na = 2\nb = 3\nprint(a + b)\n```"
        self.assertEqual(extract_code(text), "a = 2\nb = 3\nprint(a + b)")


# ═══════════════════════════════════════════════════════════════════════════════
# 22. Code execution (real subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunCode(unittest.TestCase):

    def test_successful_run_captures_stdout(self):
        r = run_code("print(6 * 7)")
        self.assertIsInstance(r, CodeExecutionResult)
        self.assertTrue(r.success)
        self.assertEqual(r.output, "42")
        self.assertEqual(r.error, "")

    def test_runtime_error_is_failure(self):
        r = run_code("raise ValueError('boom')")
        self.assertFalse(r.success)
        self.assertIn("ValueError", r.error)

    def test_no_output_is_failure(self):
        r = run_code("x = 1 + 1")  # computes but never prints
        self.assertFalse(r.success)

    def test_timeout_is_failure(self):
        r = run_code("while True:\n    pass", timeout=1.0)
        self.assertFalse(r.success)
        self.assertIn("timed out", r.error.lower())

    def test_can_import_stdlib(self):
        r = run_code("from itertools import permutations\nprint(len(list(permutations([1,2,3]))))")
        self.assertTrue(r.success)
        self.assertEqual(r.output, "6")


# ═══════════════════════════════════════════════════════════════════════════════
# 23. BoT.run() with code execution
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoTCodeExecution(unittest.TestCase):

    def test_disabled_by_default(self):
        self.assertFalse(BoT(FixedMockLLM()).execute_code)

    def test_executed_output_becomes_answer(self):
        # New-task path; instantiation returns a program that prints the answer.
        llm = SequentialMockLLM([
            DISTILLED_INFO,
            "Here is the solution:\n```python\nprint((3 + 7) + (10 + 4))\n```",
        ])
        bot = BoT(llm, update_buffer=False, init_templates=[], execute_code=True)
        resp = bot.run("3 7 10 4")
        self.assertEqual(resp.final_answer, "24")
        self.assertTrue(resp.metadata["code_executed"])
        self.assertTrue(resp.metadata["code_execution_success"])

    def test_inspector_repairs_broken_code(self):
        # First program errors; the inspector returns a corrected one.
        llm = SequentialMockLLM([
            DISTILLED_INFO,
            "```python\nraise ValueError('boom')\n```",          # instantiation (bad)
            "```python\nprint('FIXED')\n```",                    # inspector repair (good)
        ])
        bot = BoT(llm, update_buffer=False, init_templates=[],
                  execute_code=True, max_code_repairs=3)
        resp = bot.run("Q")
        self.assertEqual(resp.final_answer, "FIXED")
        self.assertEqual(llm.call_count, 3)  # distil + instantiate + 1 repair
        self.assertTrue(resp.metadata["code_execution_success"])

    def test_repair_budget_exhausted_falls_back_to_text(self):
        # Every program fails; after the repair budget the textual answer stands.
        bad = "```python\nraise RuntimeError('nope')\n```\n\n### Answer\nTEXTFALLBACK"
        llm = SequentialMockLLM([DISTILLED_INFO] + [bad] * 6)
        bot = BoT(llm, update_buffer=False, init_templates=[],
                  execute_code=True, max_code_repairs=2)
        resp = bot.run("Q")
        self.assertEqual(resp.final_answer, "TEXTFALLBACK")
        self.assertTrue(resp.metadata["code_executed"])
        self.assertFalse(resp.metadata["code_execution_success"])
        # distil + instantiate + 2 repair attempts
        self.assertEqual(llm.call_count, 4)

    def test_no_code_in_response_uses_text_answer(self):
        llm = SequentialMockLLM([DISTILLED_INFO, NEW_TASK_RESPONSE])  # no code fence
        bot = BoT(llm, update_buffer=False, init_templates=[], execute_code=True)
        resp = bot.run("Q")
        self.assertEqual(resp.final_answer, "42")
        self.assertFalse(resp.metadata["code_executed"])
        self.assertIsNone(resp.metadata["code_execution_success"])

    def test_code_block_ignored_when_execution_disabled(self):
        # execute_code=False: a code block in the response is NOT run; the
        # textual ### Answer is used instead.
        llm = SequentialMockLLM([
            DISTILLED_INFO,
            "```python\nprint('SHOULD-NOT-RUN')\n```\n\n### Answer\n99",
        ])
        bot = BoT(llm, update_buffer=False, init_templates=[], execute_code=False)
        resp = bot.run("Q")
        self.assertEqual(resp.final_answer, "99")
        self.assertFalse(resp.metadata["code_executed"])


if __name__ == "__main__":
    unittest.main()
