"""
Unit tests for the Falsification-of-Thought (FoT) baseline.

Test organisation
─────────────────
 1.  TestAnswerHelpers          — rho's comparison primitives
 2.  TestOptionRelations        — multiple-choice relabelling (T, g and g^-1)
 3.  TestSvgRelations           — affine / reversal transformations
 4.  TestWordProblemRelations   — scaling, backward substitution, permutation
 5.  TestCodeRelations          — identifier renaming, dead-code insertion
 6.  TestCatalogue              — HasChecker / catalogue selection, --fot_relations
 7.  TestGeneratedCatalogue     — pi_mr-gen parsing (ablation)
 8.  TestParsersAndMajority     — Extract(.) and Majority(O)
 9.  TestCheckers               — trusted checkers c_q
10.  TestExecutableRegime       — model proposes, checker decides
11.  TestMetamorphicRegime      — orbit construction, tau, orbit-majority rule
12.  TestDriver                 — fixpoint, archive order, metadata

Reference:
    "Falsification-of-Thought: Reasoning by Metamorphic Self-Refutation".
"""

import math
import re
import unittest
from typing import List, Optional

from baseline.FoT import Damage, FoT
from baseline.FoT.checkers import (
    _HAS_CHESS,
    checkmate_checker,
    gameof24_checker,
    get_checker,
    has_checker,
    multistep_arithmetic_checker,
)
from baseline.FoT.fot import _majority_keys, _parse_probe, _parse_variant
from baseline.FoT.relations import (
    answer_key,
    answers_equal,
    enumerate_slots,
    get_catalogue,
    normalize_answer,
    numeric_literals,
    option_letter,
    parse_generated_catalogue,
    parse_number,
)
from models.base import BaseLLM, LLMResponse


GEOMETRY_Q = (
    'This SVG path element <path d="M 10.00,10.00 L 20.00,10.00 M 20.00,10.00 '
    'L 20.00,20.00 M 20.00,20.00 L 10.00,10.00"/> draws a\n'
    "Options:\n(A) circle\n(B) kite\n(C) triangle"
)

CODE_Q = (
    "```python\ndef f(nums):\n    total = 0\n    for n in nums:\n"
    "        total += n\n    return total\n```\n\nWhat does `f([1, 2])` return?"
)

# Two answer-preserving relations, so a disagreement can reach tau = 2 and put the
# candidate outside the majority of a three-member orbit.
TWO_OPTION_RELATIONS = ["options_shift1", "options_reverse"]


class ScriptedLLM(BaseLLM):
    """A frozen model M driven by a caller-supplied response function."""

    def __init__(self, respond):
        self.model = "scripted"
        self.respond = respond
        self.prompts: List[str] = []

    def generate(self, prompt: str, temperature: float = 0,
                 logprobs: bool = False) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(content=self.respond(prompt, len(self.prompts) - 1),
                           model_name=self.model, input_tokens=1, output_tokens=1)


def _relation(task: str, name: str, subtask: Optional[str] = None):
    return next(r for r in get_catalogue(task, subtask) if r.name == name)


def _answer_shape(prompt: str, shape: str) -> str:
    """Answer with whichever letter labels ``shape`` in *this* prompt."""
    options = dict((body.strip(), letter)
                   for letter, body in re.findall(r"\((\w)\) (.+)", prompt))
    return f"ANSWER: ({options[shape]})"


def _is_repair(prompt: str) -> bool:
    return "Previous answer to the original problem" in prompt


# ── 1. Answer helpers ──────────────────────────────────────────────────────────
class TestAnswerHelpers(unittest.TestCase):

    def test_parse_number_handles_separators_and_units(self):
        self.assertEqual(parse_number("$1,250.50"), 1250.50)
        self.assertEqual(parse_number("The answer is 42."), 42.0)
        self.assertIsNone(parse_number("kite"))

    def test_normalize_answer_strips_boilerplate(self):
        self.assertEqual(normalize_answer("The answer is  Kite."), "kite")

    def test_option_letter_forms(self):
        self.assertEqual(option_letter("(D)"), "D")
        self.assertEqual(option_letter("D) kite"), "D")
        self.assertEqual(option_letter("d"), "D")
        self.assertIsNone(option_letter("kite"))

    def test_answers_equal_prefers_letters_then_numbers(self):
        self.assertTrue(answers_equal("(B)", "B) kite"))
        self.assertTrue(answers_equal("18", "The answer is 18"))
        self.assertFalse(answers_equal("18", "20"))
        self.assertTrue(answers_equal("kite", "Kite."))

    def test_answer_key_agrees_with_equality(self):
        """Majority(O) groups by this key, so it must not split equal answers."""
        self.assertEqual(answer_key("(B)"), answer_key("B) kite"))
        self.assertEqual(answer_key("18"), answer_key("The answer is 18.0"))
        self.assertNotEqual(answer_key("18"), answer_key("20"))

    def test_numeric_literals_are_compared_as_a_multiset(self):
        self.assertEqual(numeric_literals("5 apples and 3.0 pears"), ["3", "5"])
        self.assertEqual(numeric_literals("3 pears, 5 apples"), ["3", "5"])
        self.assertNotEqual(numeric_literals("5 apples"), numeric_literals("6 apples"))


# ── 2. Multiple-choice relabelling ─────────────────────────────────────────────
class TestOptionRelations(unittest.TestCase):

    def test_shift_moves_bodies_and_pullback_inverts_it(self):
        rel = _relation("bigbenchhard", "options_shift1")
        variant = rel.apply(GEOMETRY_Q, "(C)")
        self.assertIn("(A) triangle", variant.question)
        self.assertIn("(B) circle", variant.question)
        self.assertIn("(C) kite", variant.question)
        # 'triangle' was (C) and is now (A): the variant's (A) pulls back to (C).
        self.assertEqual(variant.pullback("(A)"), "(C)")
        self.assertTrue(variant.holds("(C)", "(A)"))
        self.assertFalse(variant.holds("(C)", "(B)"))

    def test_expected_for_is_the_forward_map_used_by_the_witness(self):
        rel = _relation("bigbenchhard", "options_shift1")
        variant = rel.apply(GEOMETRY_Q, "(C)")
        # g(a): answering (C) originally must mean answering (A) on the variant.
        self.assertEqual(variant.expected_value("(C)"), "(A)")

    def test_reverse_is_its_own_inverse(self):
        rel = _relation("bigbenchhard", "options_reverse")
        variant = rel.apply(GEOMETRY_Q, "(A)")
        self.assertIn("(A) triangle", variant.question)
        self.assertIn("(C) circle", variant.question)
        self.assertEqual(variant.pullback("(C)"), "(A)")

    def test_unparseable_variant_answer_is_not_a_violation(self):
        rel = _relation("bigbenchhard", "options_shift1")
        variant = rel.apply(GEOMETRY_Q, "(C)")
        # No letter to pull back → the relation abstains rather than refutes.
        self.assertTrue(variant.holds("(C)", "I am not sure"))

    def test_skipped_when_there_are_no_options(self):
        rel = _relation("bigbenchhard", "options_shift1")
        self.assertIsNone(rel.apply("What is 2 + 2?", "4"))


# ── 3. SVG relations ───────────────────────────────────────────────────────────
class TestSvgRelations(unittest.TestCase):

    def _coords(self, text: str):
        d = re.search(r'd="([^"]*)"', text).group(1)
        return [(float(x), float(y))
                for x, y in re.findall(r"(-?\d+\.\d+),(-?\d+\.\d+)", d)]

    def _path(self, text: str) -> str:
        return re.search(r'd="([^"]*)"', text).group(1)

    def test_canonicalisation_collapses_the_per_edge_subpaths(self):
        """The vertex count *is* the answer, so the redundant encoding is the enemy."""
        rel = _relation("bigbenchhard", "svg_canonicalise", "geometric_shapes")
        variant = rel.apply(GEOMETRY_Q, "(C)")
        self.assertEqual(self._path(variant.question),
                         "M 10.00,10.00 L 20.00,10.00 L 20.00,20.00 L 10.00,10.00")
        self.assertTrue(variant.holds("(C)", "(C)"))
        self.assertFalse(variant.holds("(C)", "(B)"))

    def test_canonicalisation_is_skipped_when_it_is_a_no_op(self):
        already = ('This SVG path element <path d="M 10.00,10.00 L 20.00,10.00 '
                   'L 20.00,20.00"/> draws a\nOptions:\n(A) circle\n(B) line')
        rel = _relation("bigbenchhard", "svg_canonicalise", "geometric_shapes")
        self.assertIsNone(rel.apply(already, "(B)"))

    def test_translate_is_exact_on_the_two_decimal_grid(self):
        rel = _relation("bigbenchhard", "svg_translate", "geometric_shapes")
        self.assertEqual(self._coords(rel.apply(GEOMETRY_Q, "(C)").question)[0],
                         (15.00, 15.00))

    def test_isometries_preserve_every_pairwise_distance(self):
        """Remark 2: only exact isometries of the grid are catalogued."""
        for name in ("svg_reflect", "svg_rotate90", "svg_translate"):
            with self.subTest(relation=name):
                rel = _relation("bigbenchhard", name, "geometric_shapes")
                src = self._coords(_relation("bigbenchhard", "svg_canonicalise",
                                             "geometric_shapes")
                                   .apply(GEOMETRY_Q, "(C)").question)
                dst = self._coords(rel.apply(GEOMETRY_Q, "(C)").question)
                self.assertEqual(len(src), len(dst))
                for i in range(len(src)):
                    for j in range(i + 1, len(src)):
                        d0 = math.dist(src[i], src[j])
                        d1 = math.dist(dst[i], dst[j])
                        self.assertAlmostEqual(d0, d1, places=6)

    def test_isometries_stay_inside_the_datasets_own_frame(self):
        """A variant that leaves the 0-100 frame is out of distribution, not neutral."""
        tall = ('This SVG path element <path d="M 5.00,2.00 L 8.00,95.00 '
                'L 11.00,2.00 L 5.00,2.00"/> draws a\nOptions:\n(A) triangle\n(B) line')
        for name in ("svg_translate", "svg_rotate90"):
            with self.subTest(relation=name):
                rel = _relation("bigbenchhard", name, "geometric_shapes")
                pts = self._coords(rel.apply(tall, "(A)").question)
                self.assertTrue(all(0 <= x <= 100 and 0 <= y <= 100 for x, y in pts),
                                pts)

    def test_reverse_emits_the_walk_backwards(self):
        rel = _relation("bigbenchhard", "svg_reverse", "geometric_shapes")
        pts = self._coords(rel.apply(GEOMETRY_Q, "(C)").question)
        self.assertEqual(pts, [(10.0, 10.0), (20.0, 20.0), (20.0, 10.0), (10.0, 10.0)])

    def test_arcs_are_transformed_not_skipped(self):
        """Sector/ellipse queries are the arc-bearing ones: they need coverage too."""
        arc = ('This SVG path element <path d="M 10.00,20.00 A 5.00,5.00 30.00 1,0 '
               '30.00,40.00"/> draws a\nOptions:\n(A) ellipse\n(B) line')
        moved = _relation("bigbenchhard", "svg_translate",
                          "geometric_shapes").apply(arc, "(A)")
        self.assertIn("A 5.00,5.00 30.00 1,0 35.00,45.00", moved.question)
        turned = _relation("bigbenchhard", "svg_rotate90",
                           "geometric_shapes").apply(arc, "(A)")
        self.assertIn("A 5.00,5.00 120.00 1,0", turned.question)

    def test_reversal_and_mirroring_skip_arcs_rather_than_corrupt_them(self):
        arc = ('This SVG path element <path d="M 10.00,20.00 A 5.00,5.00 30.00 1,0 '
               '30.00,40.00"/> draws a\nOptions:\n(A) ellipse\n(B) line')
        for name in ("svg_reverse", "svg_reflect"):
            with self.subTest(relation=name):
                self.assertIsNone(_relation("bigbenchhard", name,
                                            "geometric_shapes").apply(arc, "(A)"))

    def test_curved_paths_are_skipped_not_corrupted(self):
        rel = _relation("bigbenchhard", "svg_translate", "geometric_shapes")
        curve = ('This SVG path element <path d="M 1.00,2.00 C 5.00,5.00 7.00,7.00 '
                 '9.00,9.00"/> draws a')
        self.assertIsNone(rel.apply(curve, "(A)"))


# ── 4. Word-problem relations ──────────────────────────────────────────────────
class TestWordProblemRelations(unittest.TestCase):

    Q = "Tom has 5 apples. Ann gives him three more apples. How many apples does Tom have?"

    def test_scaling_covers_digits_and_number_words(self):
        variant = _relation("mgsm", "scale_quantities_x2").apply(self.Q, "8")
        self.assertIn("10 apples", variant.question)
        self.assertIn("6 more", variant.question)
        self.assertTrue(variant.holds("8", "16"))
        self.assertFalse(variant.holds("8", "8"))
        self.assertEqual(variant.pullback("16"), "8")
        self.assertEqual(variant.expected_value("8"), "16")      # g(a) = 2a

    def test_scaling_abstains_when_an_answer_is_not_numeric(self):
        variant = _relation("mgsm", "scale_quantities_x3").apply(self.Q, "8")
        self.assertTrue(variant.holds("8", "no idea"))

    def test_slots_are_the_stated_quantities(self):
        slots = enumerate_slots("Tom has 5 apples and 3 pears. He pays 12 dollars.")
        self.assertEqual([s.text for s in slots], ["5", "3", "12"])

    def test_backward_substitution_builds_a_self_contained_follow_up(self):
        rel = _relation("mgsm", "mask_quantity")
        variant = rel.apply(self.Q, "8", 0)
        self.assertIn("Tom has X apples", variant.question)
        # The follow-up question is generated by a fixed template and carries the
        # candidate as its *input*: it asks for X, not for a verdict on the answer.
        self.assertIn("the final answer to the problem is: 8", variant.question)
        self.assertIn("determine the value of X", variant.question)
        self.assertEqual(variant.slot, "5")
        self.assertEqual(variant.expected_value("8"), "5")
        self.assertIsNone(variant.pullback)            # answers a different question
        self.assertTrue(variant.holds("8", "5"))
        self.assertFalse(variant.holds("8", "6"))
        self.assertTrue(variant.holds("8", "cannot tell"))   # abstains, never refutes

    def test_successive_draws_mask_distinct_quantities(self):
        rel = _relation("mgsm", "mask_quantity")
        q = "Tom has 5 apples and 3 pears. How many pieces of fruit does he have?"
        self.assertEqual(rel.apply(q, "8", 0).slot, "5")
        self.assertEqual(rel.apply(q, "8", 1).slot, "3")
        self.assertEqual(rel.apply(q, "8", 2).slot, "5")   # wraps around

    def test_premise_permutation_is_guarded_against_dependent_sentences(self):
        rel = _relation("mgsm", "permute_premises")
        dependent = "Tom has 5 apples. He eats 2 of them. How many are left?"
        self.assertIsNone(rel.apply(dependent, "3"))
        independent = ("Tom has 5 apples. Ann has 3 apples. "
                       "How many apples do they have together?")
        variant = rel.apply(independent, "8")
        self.assertTrue(variant.question.startswith("Ann has 3 apples."))
        self.assertTrue(variant.question.endswith("together?"))

    def test_translation_relation_only_fires_on_non_english(self):
        rel = _relation("mgsm", "translate_to_english")
        self.assertIsNone(rel.apply)                   # needs pi_mr
        self.assertFalse(rel.applicable(self.Q))
        self.assertTrue(rel.applicable("เป็ดของเจเน็ตวางไข่วันละ 16 ฟอง"))


# ── 5. Code relations ──────────────────────────────────────────────────────────
class TestCodeRelations(unittest.TestCase):

    def test_renaming_touches_locals_but_not_the_entry_point(self):
        variant = _relation("cruxeval", "rename_identifiers").apply(CODE_Q, "3")
        self.assertIn("def f(", variant.question)
        self.assertNotIn("total", variant.question)
        self.assertIn("f([1, 2])", variant.question)   # the call site is untouched

    def test_dead_code_is_inserted_inside_the_function(self):
        variant = _relation("cruxeval", "insert_dead_code").apply(CODE_Q, "3")
        self.assertIn("_unused_fot = 0", variant.question)

    def test_unparseable_code_is_skipped(self):
        broken = "```python\ndef f(:\n```\n\nWhat does `f(1)` return?"
        self.assertIsNone(_relation("cruxeval", "rename_identifiers").apply(broken, "1"))


# ── 6. Catalogue selection ─────────────────────────────────────────────────────
class TestCatalogue(unittest.TestCase):

    def test_has_checker_is_fixed_per_benchmark(self):
        self.assertTrue(has_checker("gameof24"))
        self.assertTrue(has_checker("bigbenchhard", "multistep_arithmetic_two"))
        self.assertFalse(has_checker("bigbenchhard", "geometric_shapes"))
        self.assertFalse(has_checker("mgsm"))
        self.assertIsNone(get_checker(None))

    def test_lookup_is_most_specific_first(self):
        names = [r.name for r in get_catalogue("bigbenchhard", "geometric_shapes")]
        self.assertIn("svg_canonicalise", names)
        self.assertNotIn("svg_canonicalise",
                         [r.name for r in get_catalogue("bigbenchhard", "snarks")])

    def test_unknown_benchmark_falls_back_to_option_relations(self):
        self.assertTrue(all(r.name.startswith("options_")
                            for r in get_catalogue("no_such_benchmark")))

    def test_relations_filter_restricts_the_catalogue(self):
        cat = get_catalogue("mgsm", None, ["mask_quantity"])
        self.assertEqual([r.name for r in cat], ["mask_quantity"])

    def test_catalogue_order_puts_covariant_relations_last(self):
        """Sample(C, n) draws in order, so ordering decides which relations run."""
        mgsm = [r.name for r in get_catalogue("mgsm")]
        self.assertEqual(mgsm[0], "mask_quantity")          # backward first
        self.assertTrue(mgsm[-1].startswith("scale_"))      # covariant last

    def test_every_relation_is_non_decreasing_in_reliability(self):
        """Remark 2: a relation whose image is answered less reliably is excluded."""
        for task in ("mgsm", "bigbenchhard", "bigbenchhard:geometric_shapes",
                     "cruxeval"):
            for relation in get_catalogue(task):
                self.assertIn(relation.direction, ("symmetric", "increasing"))


# ── 7. pi_mr-gen (ablation) ────────────────────────────────────────────────────
class TestGeneratedCatalogue(unittest.TestCase):

    def test_only_answer_preserving_proposals_are_kept(self):
        text = ("MR1: TRANSFORM: rename the people | RELATION: the answer is unchanged\n"
                "MR2: TRANSFORM: add a distractor | RELATION: the answer may shift a bit\n")
        cat = parse_generated_catalogue(text)
        self.assertEqual(len(cat), 1)
        self.assertEqual(cat[0].transformation, "rename the people")
        self.assertIsNone(cat[0].apply)


# ── 8. Parsers and Majority(O) ─────────────────────────────────────────────────
class TestParsersAndMajority(unittest.TestCase):

    def test_probe_and_variant_parsing(self):
        self.assertEqual(_parse_probe("blah\nPROBE: recompute 2+2"), "recompute 2+2")
        self.assertEqual(
            _parse_variant("VARIANT: a rewritten problem\nRELATION: unchanged"),
            "a rewritten problem")

    def test_majority_is_the_set_of_most_frequent_orbit_members(self):
        self.assertEqual(_majority_keys(["4", "4", "5"]), ["4"])
        self.assertEqual(_majority_keys(["4", "The answer is 4", "5"]), ["4"])
        self.assertEqual(sorted(_majority_keys(["4", "5"])), ["4", "5"])   # a tie


# ── 9. Trusted checkers ────────────────────────────────────────────────────────
class TestCheckers(unittest.TestCase):

    def test_gameof24_verdicts(self):
        q = "4 9 10 13"                                 # the dataset's question format
        self.assertEqual(gameof24_checker(q, "(13 - 9) * 10 - 4 * 4").verdict, "fail")
        self.assertEqual(gameof24_checker(q, "(10 - 4) * (13 - 9)").verdict, "pass")

    def test_multistep_arithmetic_recomputes_the_expression(self):
        q = "((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) ="
        self.assertEqual(multistep_arithmetic_checker(q, "24").verdict, "pass")
        r = multistep_arithmetic_checker(q, "25")
        self.assertEqual(r.verdict, "fail")
        self.assertIn("24", r.detail)

    def test_non_arithmetic_question_is_undecided(self):
        self.assertEqual(
            multistep_arithmetic_checker("How many apples?", "3").verdict, "undecided")


# ── 9b. Checkmate checker (needs python-chess) ─────────────────────────────────
@unittest.skipUnless(_HAS_CHESS, "python-chess is not installed")
class TestCheckmateChecker(unittest.TestCase):
    """c_q replays the movetext and asks the board — the model never judges."""

    # Scholar's mate, White to play 4. Qxf7#.
    Q = "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4."

    def test_the_mating_move_passes(self):
        r = checkmate_checker(self.Q, "Qxf7#")
        self.assertEqual(r.verdict, "pass")
        self.assertIn("checkmated", r.detail)

    def test_it_reads_the_move_out_of_a_reasoning_trace(self):
        trace = "The knight on f6 guards h5... Therefore, the answer is Qxf7#."
        self.assertEqual(checkmate_checker(self.Q, trace).verdict, "pass")

    def test_notation_of_the_suffix_does_not_decide_the_verdict(self):
        """The board decides mate, not the '#' the model happened to type."""
        self.assertEqual(checkmate_checker(self.Q, "Qxf7+").verdict, "pass")
        self.assertEqual(checkmate_checker(self.Q, "Qf7").verdict, "pass")
        self.assertEqual(checkmate_checker(self.Q, "Bxf7#").verdict, "fail")

    def test_a_check_that_is_not_mate_fails_with_a_concrete_escape(self):
        r = checkmate_checker(self.Q, "Bxf7+")
        self.assertEqual(r.verdict, "fail")
        self.assertIn("reply", r.detail)

    def test_a_quiet_move_fails_for_not_giving_check(self):
        r = checkmate_checker(self.Q, "d3")
        self.assertEqual(r.verdict, "fail")
        self.assertIn("does not even give check", r.detail)

    def test_an_illegal_move_fails(self):
        r = checkmate_checker(self.Q, "Qxh8#")
        self.assertEqual(r.verdict, "fail")
        self.assertIn("not a legal move", r.detail)

    def test_an_answer_with_no_move_fails(self):
        self.assertEqual(checkmate_checker(self.Q, "I cannot tell.").verdict, "fail")

    def test_the_witness_never_leaks_the_mating_move(self):
        """The repair prompt must not receive the answer the benchmark withholds."""
        for wrong in ("Bxf7+", "d3", "Nf3", "O-O"):
            with self.subTest(move=wrong):
                self.assertNotIn("Qxf7", checkmate_checker(self.Q, wrong).detail)

    def test_an_unreplayable_game_is_undecided(self):
        r = checkmate_checker("1. e4 e5 2. Qzz9 Nc6", "Qxf7#")
        self.assertEqual(r.verdict, "undecided")

    def test_registered_as_the_benchmark_checker(self):
        self.assertTrue(has_checker("checkmate"))
        self.assertIs(get_checker("checkmate"), checkmate_checker)


# ── 10. Executable regime ──────────────────────────────────────────────────────
class TestExecutableRegime(unittest.TestCase):

    Q = "((1 + 1) * 3) ="

    def test_probe_prompt_never_asks_for_a_verdict(self):
        llm = ScriptedLLM(lambda p, i: "ANSWER: 6" if i == 0 else "PROBE: recompute")
        fot = FoT(llm, task="bigbenchhard", subtask="multistep_arithmetic_two")
        fot.run(self.Q)
        probe_prompt = llm.prompts[1]
        self.assertIn("is NOT to judge whether it is correct", probe_prompt)
        self.assertIn("PROBE:", probe_prompt)

    def test_sound_failure_drives_repair_and_correct_answer_survives(self):
        replies = {0: "ANSWER: 7", 2: "ANSWER: 6"}
        llm = ScriptedLLM(lambda p, i: replies.get(i, "PROBE: recompute"))
        fot = FoT(llm, task="bigbenchhard", subtask="multistep_arithmetic_two")
        response = fot.run(self.Q)
        self.assertEqual(response.final_answer, "6")
        self.assertTrue(response.metadata["accepted_fixpoint"])
        self.assertEqual(response.metadata["regime"], "executable")
        self.assertEqual(response.metadata["repairs_used"], 1)
        self.assertTrue(response.metadata["witnesses"][0]["sound"])

    def test_one_probe_settles_a_deterministic_checker(self):
        """n and tau are forced to 1: c_q's verdict does not depend on the probe."""
        llm = ScriptedLLM(lambda p, i: "ANSWER: 6" if i == 0 else "PROBE: recompute")
        fot = FoT(llm, task="bigbenchhard", subtask="multistep_arithmetic_two",
                  probes=5, tau=3)
        response = fot.run(self.Q)
        self.assertEqual(response.metadata["probes"], 1)
        self.assertEqual(response.metadata["tau"], 1)
        self.assertEqual(response.num_llm_calls, 2)     # Solve + one probe

    def test_repair_prompt_carries_the_probe_and_its_detail(self):
        """The checker's failing probe and detail d fill pi_rep's orbit slot."""
        replies = {0: "ANSWER: 7", 2: "ANSWER: 6"}
        llm = ScriptedLLM(lambda p, i: replies.get(i, "PROBE: recompute"))
        FoT(llm, task="bigbenchhard", subtask="multistep_arithmetic_two").run(self.Q)
        repair_prompt = next(p for p in llm.prompts if _is_repair(p))
        self.assertIn("recompute", repair_prompt)
        self.assertIn("It failed:", repair_prompt)

    def test_no_checker_run_when_execution_is_disabled(self):
        fot = FoT(ScriptedLLM(lambda p, i: "ANSWER: 6"), task="gameof24",
                  execute_code=False)
        self.assertEqual(fot._checker, None)
        self.assertFalse(fot._has_checker)


# ── 11. Metamorphic regime ─────────────────────────────────────────────────────
class TestMetamorphicRegime(unittest.TestCase):

    def _fot(self, respond, **kwargs):
        kwargs.setdefault("relations", TWO_OPTION_RELATIONS)
        llm = ScriptedLLM(respond)
        return FoT(llm, task="bigbenchhard", **kwargs), llm

    @staticmethod
    def _solver(candidate: str, variants: str, repaired: str = "(A)"):
        """Answer ``candidate`` to the original query and ``variants`` to the orbit."""
        state = {"n": 0}

        def respond(prompt: str, i: int) -> str:
            if _is_repair(prompt):
                return f"ANSWER: {repaired}"
            state["n"] += 1
            return _answer_shape(prompt, candidate if state["n"] == 1 else variants)

        return respond

    def test_orbit_disagreement_refutes_and_drives_a_repair(self):
        fot, _ = self._fot(self._solver("kite", "triangle", "(C)"), budget=2)
        response = fot.run(GEOMETRY_Q)
        self.assertEqual(response.metadata["repairs_used"], 1)
        self.assertEqual(response.final_answer, "(C)")
        witness = response.metadata["witnesses"][0]
        self.assertEqual(witness["regime"], "metamorphic")
        self.assertFalse(witness["sound"])              # sound only *relative to rho*
        self.assertEqual(sorted(witness["violated"]), sorted(TWO_OPTION_RELATIONS))
        # O = {a} ∪ {g^-1(a'_i)}: the variants' answers, pulled back into q's frame.
        self.assertEqual(witness["orbit"], ["(B)", "(C)", "(C)"])

    def test_repair_sees_every_formulation_not_just_the_violated_one(self):
        fot, llm = self._fot(self._solver("kite", "triangle", "(C)"), budget=2)
        fot.run(GEOMETRY_Q)
        repair_prompt = next(p for p in llm.prompts if _is_repair(p))
        self.assertIn("mutually inconsistent", repair_prompt)
        self.assertIn("options_shift1", repair_prompt)
        self.assertIn("options_reverse", repair_prompt)
        # A contradiction is presented, never a judgement on the previous answer.
        self.assertNotIn("your answer is wrong", repair_prompt.lower())

    def test_follow_up_solves_never_see_the_candidate(self):
        """Remark 1: pi_solve has no candidate slot, so a variant cannot echo it."""
        fot, llm = self._fot(self._solver("kite", "triangle", "(C)"), budget=2)
        fot.run(GEOMETRY_Q)
        variant_prompts = [p for p in llm.prompts
                           if "Solve the following problem" in p and not _is_repair(p)]
        self.assertEqual(len(variant_prompts), 3)       # the original + two variants
        for prompt in variant_prompts:
            self.assertNotIn("Candidate answer", prompt)
            self.assertNotIn("Previous answer", prompt)

    def test_a_candidate_holding_the_orbit_majority_is_not_repaired(self):
        """Remark 3: one dissenter against a 2-1 orbit does not indict a."""
        state = {"n": 0}

        def respond(prompt: str, i: int) -> str:
            state["n"] += 1
            # candidate kite, first variant triangle, second variant kite.
            return _answer_shape(prompt, "triangle" if state["n"] == 2 else "kite")

        fot, _ = self._fot(respond, budget=1, tau=1)
        response = fot.run(GEOMETRY_Q)
        self.assertTrue(response.metadata["accepted_fixpoint"])
        self.assertEqual(response.metadata["repairs_used"], 0)

    def test_tau_requires_corroboration_across_relations(self):
        """A single violation is evidence of a defect somewhere, not against a."""
        state = {"n": 0}

        def respond(prompt: str, i: int) -> str:
            state["n"] += 1
            return _answer_shape(prompt, "triangle" if state["n"] == 2 else "kite")

        fot, _ = self._fot(respond, budget=1, tau=2, orbit_majority=False)
        response = fot.run(GEOMETRY_Q)
        self.assertTrue(response.metadata["accepted_fixpoint"])

    def test_pilot_ablation_repairs_on_a_single_violation(self):
        """tau = 1 with the majority test off recovers the pilot's trigger."""
        state = {"n": 0}

        def respond(prompt: str, i: int) -> str:
            if _is_repair(prompt):
                return "ANSWER: (A)"
            state["n"] += 1
            return _answer_shape(prompt, "triangle" if state["n"] == 2 else "kite")

        fot, _ = self._fot(respond, budget=2, tau=1, orbit_majority=False)
        response = fot.run(GEOMETRY_Q)
        self.assertGreaterEqual(response.metadata["repairs_used"], 1)

    def test_backward_substitution_refutes_without_an_orbit_member(self):
        """mask_quantity answers a different question, so O = {a} and only tau rules."""
        q = ("A ticket costs 5 dollars. Tom buys 3 tickets and pays with a "
             "20-dollar bill. How much change does he get?")

        def respond(prompt: str, i: int) -> str:
            if _is_repair(prompt):
                return "ANSWER: 5"
            if "value of X" in prompt:
                # The first round masks the ticket price and re-derives it as 4,
                # contradicting the 5 the problem states; the second masks the
                # ticket count, which the repaired answer no longer pins down.
                return ("ANSWER: 4" if "costs X dollars" in prompt
                        else "ANSWER: it cannot be determined")
            return "ANSWER: 8"

        llm = ScriptedLLM(respond)
        response = FoT(llm, task="mgsm", budget=2, tau=1,
                       relations=["mask_quantity"]).run(q)
        self.assertEqual(response.final_answer, "5")
        witness = response.metadata["witnesses"][0]
        self.assertEqual(witness["violated"], ["mask_quantity"])
        self.assertEqual(witness["orbit"], ["8"])      # no g^-1: the orbit is just a

    def test_orbit_is_solved_once_per_question(self):
        """Variants do not depend on the candidate, so their answers are reused."""
        fot, llm = self._fot(self._solver("kite", "triangle"), budget=3)
        response = fot.run(GEOMETRY_Q)
        # Solve + 2 variant solves + 3 repairs; rounds 2 and 3 reuse the orbit.
        self.assertEqual(response.num_llm_calls, 6)
        self.assertEqual(response.metadata["repairs_used"], 3)

    def test_model_applied_variant_is_rejected_when_it_drops_a_quantity(self):
        """pi_mr output is mechanically validated before it is ever solved."""
        thai = "เป็ดของเจเน็ตวางไข่วันละ 16 ฟอง คำถาม"

        def respond(prompt: str, i: int) -> str:
            if "VARIANT:" in prompt:
                return "VARIANT: Janet's ducks lay eggs every day. Question\nRELATION: same"
            return "ANSWER: 18"

        llm = ScriptedLLM(respond)
        response = FoT(llm, task="mgsm", relations=["translate_to_english"],
                       budget=1, tau=1).run(thai)
        self.assertTrue(response.metadata["accepted_fixpoint"])
        # Solve + pi_mr only: the variant was discarded, so it was never solved.
        self.assertEqual(response.num_llm_calls, 2)

    def test_model_applied_relation_uses_pi_mr(self):
        thai = "เป็ดของเจเน็ตวางไข่วันละ 16 ฟอง คำถาม"

        def respond(prompt: str, i: int) -> str:
            if "VARIANT:" in prompt:
                return ("VARIANT: Janet's ducks lay 16 eggs a day. Question\n"
                        "RELATION: same")
            return "ANSWER: 18"

        llm = ScriptedLLM(respond)
        fot = FoT(llm, task="mgsm", relations=["translate_to_english"], budget=1)
        fot.run(thai)
        mr_prompts = [p for p in llm.prompts if "VARIANT:" in p]
        self.assertEqual(len(mr_prompts), 1)
        self.assertIn("Do NOT solve", mr_prompts[0])

    def test_no_applicable_relation_degenerates_to_solve(self):
        llm = ScriptedLLM(lambda p, i: "ANSWER: 4")
        fot = FoT(llm, task="mgsm", relations=["permute_premises"])
        response = fot.run("Tom has 5 apples. He eats 2. How many are left?")
        self.assertEqual(response.final_answer, "4")
        self.assertEqual(response.num_llm_calls, 1)
        self.assertTrue(response.metadata["accepted_fixpoint"])


# ── 12. Driver loop ────────────────────────────────────────────────────────────
class TestDriver(unittest.TestCase):

    Q = "((1 + 1) * 3) ="

    def test_budget_exhaustion_returns_the_best_measured_candidate(self):
        """Def. 2: ties break toward earlier entries, so a_0 is the floor."""
        state = {"n": 0}

        def respond(prompt: str, i: int) -> str:
            if "PROBE:" in prompt:
                return "PROBE: recompute"
            state["n"] += 1
            return f"ANSWER: {100 + state['n']}"        # never correct
        llm = ScriptedLLM(respond)
        response = FoT(llm, task="bigbenchhard",
                       subtask="multistep_arithmetic_two", budget=2).run(self.Q)
        self.assertFalse(response.metadata["accepted_fixpoint"])
        self.assertEqual(response.metadata["repairs_used"], 2)
        self.assertTrue(response.metadata["budget_exhausted"])
        self.assertEqual(response.metadata["returned"], "archive")
        # Every candidate violated the checker, so the earliest measured one wins.
        self.assertEqual(response.final_answer, "101")

    def test_archive_order_is_violations_then_support_then_round(self):
        """Definition 2: ≺ is lexicographic on (v, -c, k)."""
        self.assertLess(Damage(0, 1).order_key(3), Damage(1, 5).order_key(0))
        self.assertLess(Damage(1, 3).order_key(2), Damage(1, 1).order_key(0))
        self.assertLess(Damage(1, 2).order_key(0), Damage(1, 2).order_key(1))

    def test_the_final_repair_is_never_archived(self):
        """An entry is written when a candidate is measured, not when produced."""
        state = {"solves": 0}

        def respond(prompt: str, i: int) -> str:
            if _is_repair(prompt):
                return _answer_shape(GEOMETRY_Q, "triangle")
            state["solves"] += 1
            return _answer_shape(prompt,
                                 "kite" if state["solves"] == 1 else "circle")

        llm = ScriptedLLM(respond)
        response = FoT(llm, task="bigbenchhard", budget=1,
                       relations=TWO_OPTION_RELATIONS).run(GEOMETRY_Q)
        # The repair produced triangle, but no evidence about it was ever gathered.
        self.assertEqual(response.metadata["repairs_used"], 1)
        self.assertEqual(response.final_answer, "(B)")

    def test_metadata_shape(self):
        llm = ScriptedLLM(lambda p, i: "ANSWER: 6")
        response = FoT(llm, task="bigbenchhard",
                       subtask="multistep_arithmetic_two").run(self.Q)
        for key in ("task", "subtask", "regime", "has_checker", "catalogue",
                    "accepted_fixpoint", "returned", "budget", "probes", "tau",
                    "repairs_used", "budget_exhausted", "execute_code",
                    "initial_answer", "answer_changed", "witnesses",
                    "witness_history"):
            self.assertIn(key, response.metadata)
        self.assertEqual(response.baseline_type, "FoT")
        self.assertEqual(response.metadata["initial_answer"], "6")
        self.assertFalse(response.metadata["answer_changed"])

    def test_counters_and_caches_reset_between_questions(self):
        llm = ScriptedLLM(lambda p, i: "ANSWER: 6")
        fot = FoT(llm, task="bigbenchhard", subtask="multistep_arithmetic_two")
        first = fot.run(self.Q).num_llm_calls
        second = fot.run(self.Q).num_llm_calls
        self.assertEqual(first, second)

    def test_orbit_cache_does_not_leak_across_questions(self):
        def respond(prompt: str, i: int) -> str:
            return _answer_shape(prompt, "kite")

        llm = ScriptedLLM(respond)
        fot = FoT(llm, task="bigbenchhard", budget=1,
                  relations=TWO_OPTION_RELATIONS)
        first = fot.run(GEOMETRY_Q).num_llm_calls
        second = fot.run(GEOMETRY_Q).num_llm_calls
        self.assertEqual(first, 3)                      # Solve + two variants
        self.assertEqual(second, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
