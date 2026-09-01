"""
Tests for the Checkmate (checkmate-in-one) benchmark.

Covers:
- SAN normalisation and the notation-insensitive "loose" form
- Move extraction from every baseline's answer shape (bare move, CoT trace,
  \\boxed{}, Markdown) and the preference for moves legal in the position
- Canonicalisation against the legal-move list, including the ambiguity guard
- Dataset loading, problem retrieval and grading
- Registry wiring

Author: Egor Morozov
"""

import unittest

from benchmark import DATASET_REGISTRY
from benchmark.Checkmate.checkmate import (
    Checkmate,
    _build_legal_index,
    extract_move,
    _loose_san,
    _normalize_san,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

LEGAL = ["Qg8+", "Qf7+", "Qxe7#", "Qxd7+", "Bxe7", "Nf6+", "Kh1", "Re3", "O-O"]

MOVETEXT = (
    "1. e4 e5 2. Nf3 d6 3. d4 exd4 4. Nxd4 Nf6 5. Nc3 Qe7 6. Bd3 d5 7. O-O dxe4 "
    "8. Re1 Be6 9. Nxe6 fxe6 10. Bxe4 Nxe4 11. Nxe4 Nd7 12. Bg5 Qb4 13. Qg4 Qd4 "
    "14. Qxe6+ Be7 15."
)


def _example(input_text=MOVETEXT, target="Qxe7#", legal=None):
    """Minimal BIG-bench-style example row."""
    legal = LEGAL if legal is None else legal
    return {
        "input": input_text,
        "target": target,
        "target_scores": {m: (1 if m == target else 0) for m in legal},
    }


def _loaded(rows=None, num_samples=None):
    """A Checkmate dataset with in-memory rows, bypassing the JSON file."""
    ds = Checkmate(num_samples=num_samples)
    ds._data = [_example()] if rows is None else rows
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# SAN normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeSan(unittest.TestCase):

    def test_strips_mate_and_check_suffix(self):
        self.assertEqual(_normalize_san("Rg5#"), "Rg5")
        self.assertEqual(_normalize_san("Qg8+"), "Qg8")

    def test_strips_move_number_and_annotation(self):
        self.assertEqual(_normalize_san("32. Qxe7#"), "Qxe7")
        self.assertEqual(_normalize_san("31... Rg5"), "Rg5")
        self.assertEqual(_normalize_san("Qxe7#!!"), "Qxe7")

    def test_normalizes_castling_written_with_zeros(self):
        self.assertEqual(_normalize_san("0-0"), "O-O")
        self.assertEqual(_normalize_san("0-0-0#"), "O-O-O")

    def test_normalizes_unicode_dashes_in_castling(self):
        self.assertEqual(_normalize_san("O–O"), "O-O")

    def test_is_case_sensitive(self):
        # Bd2 is a bishop move; bd2 would be a pawn from the b-file.
        self.assertNotEqual(_normalize_san("Bd2"), _normalize_san("bd2"))

    def test_loose_form_drops_capture_and_promotion_markers(self):
        self.assertEqual(_loose_san("Qxe7#"), "Qe7")
        self.assertEqual(_loose_san("e8=Q#"), "e8Q")
        self.assertEqual(_loose_san("Qe7"), _loose_san("Qxe7#"))


# ─────────────────────────────────────────────────────────────────────────────
# Move extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractMove(unittest.TestCase):

    def test_bare_move(self):
        self.assertEqual(extract_move("Qxe7#", LEGAL), "Qxe7#")

    def test_answer_is_pattern(self):
        text = "Let's think step by step. ... Therefore, the answer is Qxe7#."
        self.assertEqual(extract_move(text, LEGAL), "Qxe7#")

    def test_answer_pattern_takes_first_move_after_the_phrase(self):
        text = "The answer is Qxe7# because Bxe7 allows the king to run."
        self.assertEqual(extract_move(text, LEGAL), "Qxe7#")

    def test_boxed_answer_wins_over_the_trace(self):
        text = "Consider Qg8+ first.\nThen \\boxed{Qxe7\\#}"
        self.assertEqual(extract_move(text, LEGAL), "Qxe7")

    def test_markdown_and_code_fences_are_stripped(self):
        self.assertEqual(extract_move("```\nQxe7#\n```", LEGAL), "Qxe7#")
        self.assertEqual(extract_move("**Qxe7#**", LEGAL), "Qxe7#")

    def test_castling_is_recognised(self):
        self.assertEqual(extract_move("The answer is O-O-O#", ["O-O-O#"]), "O-O-O#")

    def test_prefers_a_legal_move_over_stray_square_names(self):
        # "e7" is a square being discussed, "Qxe7#" is the move being proposed.
        text = "The king is stuck; the square e7 is the target. Answer: Qxe7#"
        self.assertEqual(extract_move(text, LEGAL), "Qxe7#")

    def test_last_move_wins_in_a_free_form_trace(self):
        text = "Maybe Qg8+.\nNo — that fails.\nQxe7# mates."
        self.assertEqual(extract_move(text, LEGAL), "Qxe7#")

    def test_works_without_a_legal_move_list(self):
        self.assertEqual(extract_move("Answer: Rg5#"), "Rg5#")

    def test_returns_none_when_no_move_present(self):
        self.assertIsNone(extract_move("I cannot determine the position.", LEGAL))
        self.assertIsNone(extract_move("", LEGAL))

    def test_does_not_match_inside_a_word(self):
        self.assertIsNone(extract_move("The made4 up word she4 is not a move.", []))


# ─────────────────────────────────────────────────────────────────────────────
# Legal-move index
# ─────────────────────────────────────────────────────────────────────────────

class TestLegalIndex(unittest.TestCase):

    def test_canonicalizes_exact_spelling(self):
        index = _build_legal_index(LEGAL)
        self.assertEqual(index.canonicalize("Qxe7#"), "Qxe7#")

    def test_canonicalizes_a_missing_capture_marker(self):
        index = _build_legal_index(LEGAL)
        self.assertEqual(index.canonicalize("Qe7"), "Qxe7#")

    def test_exact_spelling_wins_over_a_loose_collision(self):
        index = _build_legal_index(["Qxe7#", "Qe7"])
        self.assertEqual(index.canonicalize("Qe7"), "Qe7")
        self.assertEqual(index.canonicalize("Qxe7"), "Qxe7#")

    def test_loose_resolution_requires_a_unique_claimant(self):
        # SAN itself never yields two legal moves with the same loose form —
        # the guard is what keeps a fabricated collision from being guessed.
        index = _build_legal_index(["Qxe7#", "Qe7"])
        self.assertEqual(index.loose["Qe7"], ["Qxe7#", "Qe7"])
        self.assertIsNone(index.canonicalize("Qx=e7"))  # matches neither exactly

    def test_unknown_move_is_not_canonicalized(self):
        index = _build_legal_index(LEGAL)
        self.assertIsNone(index.canonicalize("Ra1"))
        self.assertFalse(index.is_legal("Ra1"))

    def test_empty_index_is_falsy(self):
        self.assertFalse(_build_legal_index([]))
        self.assertFalse(_build_legal_index(None))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckmateDataset(unittest.TestCase):

    def test_get_problem_hides_the_legal_moves_from_the_question(self):
        problem = _loaded().get_problem(0)
        self.assertEqual(problem.question, MOVETEXT)
        self.assertNotIn("Qxe7#", problem.question)
        self.assertEqual(problem.ground_truth["move"], "Qxe7#")
        self.assertEqual(problem.ground_truth["legal_moves"], LEGAL)

    def test_metadata_reports_side_to_move(self):
        white = _loaded().get_problem(0)
        self.assertEqual(white.metadata["side_to_move"], "white")
        self.assertEqual(white.metadata["num_legal_moves"], len(LEGAL))

        black_rows = [_example(input_text="1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6")]
        black = _loaded(black_rows).get_problem(0)
        self.assertEqual(black.metadata["side_to_move"], "black")

    def test_get_problem_requires_loading(self):
        with self.assertRaises(RuntimeError):
            Checkmate().get_problem(0)

    def test_get_problem_index_out_of_range(self):
        with self.assertRaises(IndexError):
            _loaded().get_problem(5)

    def test_evaluate_exact_match(self):
        ds = _loaded()
        problem = ds.get_problem(0)
        self.assertTrue(ds.evaluate_answer("Qxe7#", problem.ground_truth).is_correct)

    def test_evaluate_tolerates_notation_differences(self):
        ds = _loaded()
        gt = ds.get_problem(0).ground_truth
        for answer in ("Qxe7", "Qe7#", "The answer is 15. Qxe7#", "**Qxe7#**"):
            with self.subTest(answer=answer):
                self.assertTrue(ds.evaluate_answer(answer, gt).is_correct)

    def test_evaluate_rejects_a_different_legal_move(self):
        ds = _loaded()
        gt = ds.get_problem(0).ground_truth
        result = ds.evaluate_answer("The answer is Qg8+", gt)
        self.assertFalse(result.is_correct)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.details["extracted_move"], "Qg8+")

    def test_evaluate_with_no_move_in_the_answer(self):
        ds = _loaded()
        gt = ds.get_problem(0).ground_truth
        result = ds.evaluate_answer("I don't know.", gt)
        self.assertFalse(result.is_correct)
        self.assertEqual(result.details["comparison_method"], "none")

    def test_evaluate_accepts_a_plain_string_ground_truth(self):
        ds = _loaded()
        self.assertTrue(ds.evaluate_answer("Answer: Qxe7#", "Qxe7#").is_correct)
        self.assertFalse(ds.evaluate_answer("Answer: Qg8+", "Qxe7#").is_correct)

    def test_evaluate_without_ground_truth_move(self):
        result = _loaded().evaluate_answer("Qxe7#", {"move": ""})
        self.assertFalse(result.is_correct)
        self.assertIn("error", result.details)

    def test_prompts_and_demonstrations(self):
        ds = _loaded()
        self.assertIn("checkmate", ds.get_instruction().lower())
        self.assertIn("algebraic notation", ds.get_instruction())
        self.assertIn("chess", ds.get_system_prompt().lower())

        demos = ds.get_demonstrations(n_shot=1)
        self.assertIn("Output: Qxe7#", demos)

    def test_registry_entry(self):
        self.assertIn("checkmate", DATASET_REGISTRY)
        cls, extract = DATASET_REGISTRY["checkmate"]
        self.assertIs(cls, Checkmate)

        class _Args:
            checkmate_num_samples = 7

        self.assertEqual(extract(_Args()), {"num_samples": 7})
        self.assertEqual(extract(object()), {"num_samples": None})


class TestCheckmateLoading(unittest.TestCase):
    """Exercises the real task file — skipped when it is absent."""

    @classmethod
    def setUpClass(cls):
        cls.ds = Checkmate(num_samples=5)
        try:
            cls.ds.load_dataset()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc))

    def test_num_samples_truncates(self):
        self.assertEqual(len(self.ds), 5)

    def test_task_prefix_comes_from_the_file(self):
        self.assertIn("checkmate", self.ds.task_prefix.lower())
        self.assertIn(self.ds.task_prefix, self.ds.get_instruction())

    def test_reference_move_grades_correct_for_every_loaded_problem(self):
        for i in range(len(self.ds)):
            problem = self.ds.get_problem(i)
            with self.subTest(index=i):
                self.assertTrue(problem.ground_truth["move"].endswith("#"))
                result = self.ds.evaluate_answer(
                    f"The mating move is {problem.ground_truth['move']}.",
                    problem.ground_truth,
                )
                self.assertTrue(result.is_correct)

    def test_every_other_legal_move_grades_incorrect(self):
        problem = self.ds.get_problem(0)
        target = problem.ground_truth["move"]
        for move in problem.ground_truth["legal_moves"]:
            if move == target:
                continue
            with self.subTest(move=move):
                self.assertFalse(
                    self.ds.evaluate_answer(move, problem.ground_truth).is_correct
                )


if __name__ == "__main__":
    unittest.main()
