"""
Unit tests for BigBenchHard benchmark dataset.

Tests cover:
- Task enumeration and validation
- Answer extraction from various formats
- Task-specific answer normalization
- HuggingFace dataset loading
- Evaluation results
"""

import unittest
from benchmark.BigBenchHard.bigbenchhard import (
    BigBenchHard,
    BigBenchHardTask,
    _extract_answer_from_text,
    _normalize_whitespace,
    _parse_options,
    _resolve_choice_by_text,
    TASK_ANSWER_TYPES,
)

GEOMETRIC_SHAPES_Q = (
    'This SVG path element <path d="M 55.57,80.69 L 57.38,65.80"/> draws a\n'
    "Options:\n"
    "(A) circle\n"
    "(B) heptagon\n"
    "(C) hexagon\n"
    "(D) kite\n"
    "(E) line\n"
    "(J) triangle"
)


class TestBigBenchHardTasks(unittest.TestCase):
    """Test BigBenchHardTask enumeration and task initialization."""

    def test_enum_count(self):
        """Verify that all 27 BBH tasks are in the enum."""
        tasks = list(BigBenchHardTask)
        self.assertEqual(len(tasks), 27, "Should have exactly 27 tasks")

    def test_valid_task_initialization(self):
        """Test initialization with valid task names."""
        for task in BigBenchHardTask:
            ds = BigBenchHard(task=task.value)
            self.assertEqual(ds.task, task.value)

    def test_invalid_task_raises_error(self):
        """Test that invalid task names raise ValueError."""
        with self.assertRaises(ValueError) as context:
            BigBenchHard(task="invalid_task_name")
        self.assertIn("Invalid task", str(context.exception))

    def test_default_task(self):
        """Test default task initialization."""
        ds = BigBenchHard()
        self.assertEqual(ds.task, "boolean_expressions")

    def test_custom_split(self):
        """Test initialization with custom split."""
        ds = BigBenchHard(task="boolean_expressions", split="train")
        self.assertEqual(ds.split, "train")


class TestBigBenchHardAnswerExtraction(unittest.TestCase):
    """Test answer extraction from various model output formats."""

    def test_extract_markdown_codeblock(self):
        """Test extraction from markdown code blocks."""
        text = "```\nFalse\n```"
        result = _extract_answer_from_text(text)
        self.assertIn("False", result)

    def test_extract_latex_dollars(self):
        """Test extraction from LaTeX $...$ delimiters."""
        text = "The answer is $42$."
        result = _extract_answer_from_text(text)
        self.assertIn("42", result)

    def test_extract_answer_prefix(self):
        """Test extraction with 'Answer:' prefix."""
        text = "After careful reasoning, Answer: True"
        result = _extract_answer_from_text(text)
        self.assertIn("True", result)

    def test_extract_final_answer(self):
        """Test extraction with 'Final answer:' prefix."""
        text = "Working through the problem...\nFinal answer: C"
        result = _extract_answer_from_text(text)
        self.assertIn("C", result)

    def test_extract_fallback_last_line(self):
        """Test fallback: extract last non-empty line."""
        text = "Some reasoning here\n\nB"
        result = _extract_answer_from_text(text)
        self.assertEqual(result.strip(), "B")

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "  word1   word2   word3  "
        result = _normalize_whitespace(text)
        self.assertEqual(result, "word1 word2 word3")


class TestBigBenchHardEvaluation(unittest.TestCase):
    """Test task-specific evaluation and normalization."""

    def test_boolean_normalization(self):
        """Test boolean answer normalization."""
        ds = BigBenchHard(task="boolean_expressions")

        # Test True variants
        for pred in ["true", "True", "yes", "YES", "correct", "1"]:
            result = ds.evaluate_answer(pred, "True")
            self.assertTrue(result.is_correct, f"Failed for prediction: {pred}")

        # Test False variants
        for pred in ["false", "False", "no", "NO", "incorrect", "0"]:
            result = ds.evaluate_answer(pred, "False")
            self.assertTrue(result.is_correct, f"Failed for prediction: {pred}")

        # Test mismatch
        result = ds.evaluate_answer("True", "False")
        self.assertFalse(result.is_correct)

    def test_numeric_normalization(self):
        """Test numeric answer extraction."""
        ds = BigBenchHard(task="multistep_arithmetic_two")

        # Test number extraction
        result = ds.evaluate_answer("The answer is 42", "42")
        self.assertTrue(result.is_correct)

        # Test negative number
        result = ds.evaluate_answer("Answer: -5", "-5")
        self.assertTrue(result.is_correct)

        # Test number in sentence
        result = ds.evaluate_answer("After calculation, 100 is the result", "100")
        self.assertTrue(result.is_correct)

        # Regression: trailing sentence period must not be absorbed into the match
        result = ds.evaluate_answer("The answer is -13.", "-13")
        self.assertTrue(result.is_correct, "Trailing period caused false negative")

        result = ds.evaluate_answer("The final answer is 42.", "42")
        self.assertTrue(result.is_correct, "Trailing period caused false negative")

        # \\boxed{...} format used by math models
        result = ds.evaluate_answer(r"\[\n\boxed{24}\n\]", "24")
        self.assertTrue(result.is_correct, "\\boxed not extracted")

        result = ds.evaluate_answer(r"Step 1: \boxed{5}. Final: \boxed{-13}", "-13")
        self.assertTrue(result.is_correct, "Last \\boxed should win over intermediate")

        # Regression: "answer to the expression ((8-2+...)) is 42" — the expression
        # contains many numbers; the actual answer is the LAST number in the string.
        result = ds.evaluate_answer(
            r"the final answer to the expression \(((8 - 2 + -2 \times 6) \times (8 + -6 + -8 + -1))\) is **42**.",
            "42",
        )
        self.assertTrue(result.is_correct, "First number in embedded expression caused false negative")

        result = ds.evaluate_answer(
            r"the final numerical answer to the expression \(((1 - 7 - -8 \times 3) + (-7 - -2 + -3 \times 6))\) is \(-5\).",
            "-5",
        )
        self.assertTrue(result.is_correct, "First number in embedded expression caused false negative")

    def test_choice_normalization(self):
        """Test multiple-choice answer extraction."""
        ds = BigBenchHard(task="disambiguation_qa")

        # Test letter extraction
        result = ds.evaluate_answer("The answer is (B)", "B")
        self.assertTrue(result.is_correct)

        # Test case-insensitive
        result = ds.evaluate_answer("Answer: c", "C")
        self.assertTrue(result.is_correct)

        # Test standalone letter
        result = ds.evaluate_answer("A", "A")
        self.assertTrue(result.is_correct)

        # Regression: sentence-initial "I" must not shadow the answer letter
        result = ds.evaluate_answer("I think option B is correct", "(B)")
        self.assertTrue(result.is_correct, "Pronoun 'I' caused false negative")

        result = ds.evaluate_answer("I would say b", "(B)")
        self.assertTrue(result.is_correct, "Pronoun 'I' caused false negative")

        # Regression: half-parenthesised form "A)"
        result = ds.evaluate_answer("Answer: A)", "(A)")
        self.assertTrue(result.is_correct, "Half-parenthesised form not handled")

    def test_word_list_normalization(self):
        """Test word list normalization for word_sorting."""
        ds = BigBenchHard(task="word_sorting")

        # Test comma-separated
        result = ds.evaluate_answer("apple, banana, cherry", "apple banana cherry")
        self.assertTrue(result.is_correct)

        # Test space-separated
        result = ds.evaluate_answer("dog cat bird", "bird cat dog")
        self.assertTrue(result.is_correct)

        # Test order doesn't matter (they get sorted)
        result = ds.evaluate_answer("zebra apple mouse", "apple mouse zebra")
        self.assertTrue(result.is_correct)

    def test_answer_extraction_with_formatting(self):
        """Test answer extraction combined with formatting."""
        ds = BigBenchHard(task="boolean_expressions")

        # LaTeX + answer prefix
        result = ds.evaluate_answer(
            "The answer is $True$",
            "True"
        )
        self.assertTrue(result.is_correct)

        # Markdown code block
        result = ds.evaluate_answer(
            "```\nFalse\n```",
            "False"
        )
        self.assertTrue(result.is_correct)

    def test_evaluation_details(self):
        """Test that evaluation result contains details."""
        ds = BigBenchHard(task="boolean_expressions")
        result = ds.evaluate_answer("True", "True")

        self.assertIn("raw_prediction", result.details)
        self.assertIn("extracted_answer", result.details)
        self.assertIn("extracted_normalized", result.details)
        self.assertIn("truth_normalized", result.details)
        self.assertIn("task", result.details)
        self.assertIn("answer_type", result.details)


class TestBigBenchHardDatasetLoading(unittest.TestCase):
    """Test HuggingFace dataset loading."""

    def test_load_dataset(self):
        """Test loading dataset from HuggingFace (network dependent)."""
        ds = BigBenchHard(task="boolean_expressions", split="train")
        try:
            ds.load_dataset()
            self.assertIsNotNone(ds._data)
            self.assertGreater(len(ds._data), 0)
        except RuntimeError as e:
            if "Failed to load" in str(e):
                self.skipTest("HuggingFace dataset not accessible (network issue)")
            else:
                raise

    def test_get_problem(self):
        """Test retrieving a problem after loading."""
        ds = BigBenchHard(task="boolean_expressions", split="train")
        try:
            ds.load_dataset()
            problem = ds.get_problem(0)

            self.assertIsNotNone(problem.index)
            self.assertIsNotNone(problem.question)
            self.assertIsNotNone(problem.ground_truth)
            self.assertIsNotNone(problem.metadata)
        except RuntimeError as e:
            if "Failed to load" in str(e):
                self.skipTest("HuggingFace dataset not accessible (network issue)")
            else:
                raise

    def test_get_problem_out_of_range(self):
        """Test that out-of-range index raises IndexError."""
        ds = BigBenchHard(task="boolean_expressions", split="train")
        try:
            ds.load_dataset()
            with self.assertRaises(IndexError):
                ds.get_problem(999999)
        except RuntimeError as e:
            if "Failed to load" in str(e):
                self.skipTest("HuggingFace dataset not accessible (network issue)")
            else:
                raise

    def test_get_problem_before_load(self):
        """Test that accessing problem before load raises RuntimeError."""
        ds = BigBenchHard(task="boolean_expressions")
        with self.assertRaises(RuntimeError):
            ds.get_problem(0)

    def test_dataset_name(self):
        """Test dataset name formatting."""
        ds = BigBenchHard(task="boolean_expressions")
        self.assertIn("boolean_expressions", ds.dataset_name)
        self.assertIn("BigBenchHard", ds.dataset_name)


class TestOptionParsing(unittest.TestCase):
    """Test recovery of the option letter from an answer given as option text."""

    OPTIONS = {"a": "circle", "b": "heptagon", "d": "kite", "j": "triangle"}

    def test_parse_options(self):
        """Option lines are parsed into a lowercased letter -> body map."""
        options = _parse_options(GEOMETRIC_SHAPES_Q)
        self.assertEqual(options["b"], "heptagon")
        self.assertEqual(options["j"], "triangle")
        self.assertEqual(len(options), 6)

    def test_parse_options_absent(self):
        """A question without an option block yields an empty map."""
        self.assertEqual(_parse_options("What is 2 + 2?"), {})

    def test_resolve_bare_body(self):
        """A bare option body resolves to its letter."""
        self.assertEqual(_resolve_choice_by_text("heptagon", self.OPTIONS), "b")

    def test_resolve_body_in_sentence(self):
        """A body embedded in a sentence resolves to its letter."""
        self.assertEqual(
            _resolve_choice_by_text("the path draws a heptagon", self.OPTIONS), "b")

    def test_resolve_prefers_last_occurrence(self):
        """When several bodies occur, the last one is the stated answer."""
        self.assertEqual(
            _resolve_choice_by_text("not a triangle, but a kite", self.OPTIONS), "d")

    def test_resolve_requires_word_boundary(self):
        """Matching is word-bounded, never a loose substring."""
        self.assertIsNone(_resolve_choice_by_text("circles", {"a": "circle"}))

    def test_resolve_ignores_dollar_sign(self):
        """The prediction has "$" stripped as a LaTeX delimiter; bodies match anyway."""
        self.assertEqual(
            _resolve_choice_by_text("my 1000 dollar phone", {"a": "my $1000 dollar phone"}),
            "a")

    def test_resolve_ambiguous_options(self):
        """Identical option bodies carry no information and stay unmatched."""
        self.assertIsNone(_resolve_choice_by_text("same text", {"a": "same text",
                                                                "b": "same text"}))

    def test_resolve_no_match(self):
        """An answer naming no option returns None."""
        self.assertIsNone(_resolve_choice_by_text("dodecahedron", self.OPTIONS))


class TestChoiceGradedByOptionText(unittest.TestCase):
    """A correct multiple-choice answer must not be lost to formatting."""

    def setUp(self):
        self.ds = BigBenchHard(task="geometric_shapes")
        # Stand in for load_dataset()/get_problem() so the test stays offline.
        self.ds._last_problem = type("P", (), {
            "ground_truth": "(B)",
            "metadata": {"options": _parse_options(GEOMETRIC_SHAPES_Q)},
        })()

    def test_option_text_scored_correct(self):
        """"heptagon" is the body of "(B)" and is scored correct."""
        result = self.ds.evaluate_answer("heptagon", "(B)")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.details["extracted_normalized"], "b")

    def test_option_text_with_article(self):
        """The bare-letter fallback must not reduce "a heptagon" to "a"."""
        self.assertTrue(self.ds.evaluate_answer("The answer is a heptagon.", "(B)").is_correct)

    def test_wrong_option_text_scored_incorrect(self):
        """A different shape is still wrong."""
        self.assertFalse(self.ds.evaluate_answer("triangle", "(B)").is_correct)

    def test_letter_still_wins(self):
        """An explicit letter takes precedence over any body text."""
        self.assertTrue(self.ds.evaluate_answer("(B) heptagon", "(B)").is_correct)

    def test_stale_problem_not_reused(self):
        """Grading another question's ground truth falls back to letter matching."""
        result = self.ds.evaluate_answer("heptagon", "(C)")
        self.assertFalse(result.is_correct)
        self.assertEqual(result.details["options"], {})


class TestChoiceInstruction(unittest.TestCase):
    """Choice tasks are graded on the letter, so the prompt must request one."""

    def test_choice_tasks_request_a_letter(self):
        for task, answer_type in TASK_ANSWER_TYPES.items():
            instruction = BigBenchHard(task=task).get_instruction()
            if answer_type == "choice":
                self.assertIn("letter", instruction, f"{task} does not ask for a letter")
            else:
                self.assertNotIn("letter", instruction, f"{task} wrongly asks for a letter")

    def test_choice_instruction_not_boolean(self):
        """geometric_shapes used to ask for "True, False, or the requested property"."""
        instruction = BigBenchHard(task="geometric_shapes").get_instruction()
        self.assertNotIn("True", instruction)


class TestTaskAnswerTypes(unittest.TestCase):
    """Test task answer type classification."""

    def test_all_tasks_have_type(self):
        """Verify all tasks have an answer type classification."""
        for task in BigBenchHardTask:
            self.assertIn(task.value, TASK_ANSWER_TYPES,
                         f"Task {task.value} missing from TASK_ANSWER_TYPES")

    def test_valid_answer_types(self):
        """Verify answer types are valid."""
        valid_types = {"boolean", "numeric", "choice", "word_list", "default"}
        for task, answer_type in TASK_ANSWER_TYPES.items():
            self.assertIn(answer_type, valid_types,
                         f"Task {task} has invalid answer type: {answer_type}")


if __name__ == "__main__":
    unittest.main()
