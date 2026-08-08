"""
Unit tests for the Falsification-of-Thought (FoT) baseline.

Test organisation
─────────────────
 1.  TestAnswerHelpers          — rho's comparison primitives
 2.  TestOptionRelations        — multiple-choice relabelling (T and g^-1)
 3.  TestSvgRelations           — affine / reversal relations on SVG paths
 4.  TestWordProblemRelations   — scaling, masking, premise permutation
 5.  TestCodeRelations          — identifier renaming, dead-code insertion
 6.  TestCatalogue              — HasChecker / catalogue selection, --fot_relations
 7.  TestGeneratedCatalogue     — pi_mr-gen parsing (ablation)
 8.  TestCheckers               — trusted checkers c_q
 9.  TestExecutableRegime       — model proposes, checker decides
10.  TestMetamorphicRegime      — orbit, tau, majority rule, independence
11.  TestDriver                 — fixpoint, budget exhaustion, archive floor

Reference:
    "Falsification-of-Thought: Reasoning by Metamorphic Self-Refutation".
"""

import re
import unittest
from typing import List, Optional

from baseline.FoT import FoT
from baseline.FoT.checkers import (
    gameof24_checker,
    get_checker,
    has_checker,
    multistep_arithmetic_checker,
)
from baseline.FoT.relations import (
    answers_equal,
    get_catalogue,
    normalize_answer,
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

    def test_translate_scale_is_affine_and_answer_preserving(self):
        rel = _relation("bigbenchhard", "svg_translate_scale", "geometric_shapes")
        variant = rel.apply(GEOMETRY_Q, "(C)")
        self.assertEqual(self._coords(variant.question)[0], (28.00, 8.00))
        self.assertTrue(variant.holds("(C)", "(C)"))
        self.assertFalse(variant.holds("(C)", "(B)"))

    def test_rotation_preserves_pairwise_distances(self):
        rel = _relation("bigbenchhard", "svg_rotate", "geometric_shapes")
        before = self._coords(GEOMETRY_Q)
        after = self._coords(rel.apply(GEOMETRY_Q, "(C)").question)
        self.assertEqual(len(before), len(after))
        for (ax, ay), (bx, by) in zip(before[:-1], before[1:]):
            i = before.index((ax, ay))
            j = before.index((bx, by))
            d0 = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            d1 = ((after[i][0] - after[j][0]) ** 2
                  + (after[i][1] - after[j][1]) ** 2) ** 0.5
            self.assertAlmostEqual(d0, d1, delta=0.05)   # coordinates are 2-dp

    def test_reverse_emits_the_walk_backwards(self):
        rel = _relation("bigbenchhard", "svg_reverse", "geometric_shapes")
        pts = self._coords(rel.apply(GEOMETRY_Q, "(C)").question)
        self.assertEqual(pts, [(10.0, 10.0), (20.0, 20.0), (20.0, 10.0), (10.0, 10.0)])

    def test_curved_paths_are_skipped_not_corrupted(self):
        rel = _relation("bigbenchhard", "svg_rotate", "geometric_shapes")
        arc = 'This SVG path element <path d="M 1.00,2.00 A 5.00,5.00 0 0 1 9.00,9.00"/> draws a'
        self.assertIsNone(rel.apply(arc, "(A)"))


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

    def test_scaling_abstains_when_an_answer_is_not_numeric(self):
        variant = _relation("mgsm", "scale_quantities_x3").apply(self.Q, "8")
        self.assertTrue(variant.holds("8", "no idea"))

    def test_masking_is_backward_verification(self):
        variant = _relation("mgsm", "mask_quantity").apply(self.Q, "8")
        self.assertIn("Tom has X apples", variant.question)
        self.assertIn("8", variant.question)          # the candidate is the input here
        self.assertIsNone(variant.pullback)            # answers a different question
        self.assertTrue(variant.holds("8", "5"))
        self.assertFalse(variant.holds("8", "6"))

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
        self.assertIn("svg_rotate", names)
        self.assertNotIn("svg_rotate",
                         [r.name for r in get_catalogue("bigbenchhard", "snarks")])

    def test_unknown_benchmark_falls_back_to_option_relations(self):
        self.assertTrue(all(r.name.startswith("options_")
                            for r in get_catalogue("no_such_benchmark")))

    def test_relations_filter_restricts_the_catalogue(self):
        cat = get_catalogue("mgsm", None, ["mask_quantity"])
        self.assertEqual([r.name for r in cat], ["mask_quantity"])


# ── 7. pi_mr-gen (ablation) ────────────────────────────────────────────────────
class TestGeneratedCatalogue(unittest.TestCase):

    def test_only_answer_preserving_proposals_are_kept(self):
        text = ("MR1: TRANSFORM: rename the people | RELATION: the answer is unchanged\n"
                "MR2: TRANSFORM: add a distractor | RELATION: the answer may shift a bit\n")
        cat = parse_generated_catalogue(text)
        self.assertEqual(len(cat), 1)
        self.assertEqual(cat[0].transformation, "rename the people")
        self.assertIsNone(cat[0].apply)


# ── 8. Trusted checkers ────────────────────────────────────────────────────────
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


# ── 9. Executable regime ───────────────────────────────────────────────────────
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

    def test_no_checker_run_when_execution_is_disabled(self):
        fot = FoT(ScriptedLLM(lambda p, i: "ANSWER: 6"), task="gameof24",
                  execute_code=False)
        self.assertEqual(fot._checker, None)
        self.assertFalse(fot._has_checker)


# ── 10. Metamorphic regime ─────────────────────────────────────────────────────
class TestMetamorphicRegime(unittest.TestCase):

    def _fot(self, respond, **kwargs):
        llm = ScriptedLLM(respond)
        return FoT(llm, task="bigbenchhard", subtask="geometric_shapes", **kwargs), llm

    @staticmethod
    def _answer_shape(prompt: str, shape: str) -> str:
        """Answer with whichever letter labels ``shape`` in *this* prompt."""
        options = dict((body.strip(), letter)
                       for letter, body in re.findall(r"\((\w)\) (.+)", prompt))
        return f"ANSWER: ({options[shape]})"

    def test_orbit_disagreement_refutes_and_repairs(self):
        def respond(prompt: str, i: int) -> str:
            if i == 0:                                  # the original solve
                return self._answer_shape(prompt, "kite")
            return self._answer_shape(prompt, "triangle")

        fot, llm = self._fot(respond, budget=1)
        response = fot.run(GEOMETRY_Q)
        self.assertEqual(response.metadata["repairs_used"], 1)
        self.assertEqual(response.metadata["falsification_regimes"], ["metamorphic"])
        self.assertGreaterEqual(len(response.metadata["witness_history"]), 1)

    def test_follow_up_solves_never_see_the_candidate(self):
        def respond(prompt: str, i: int) -> str:
            return self._answer_shape(prompt, "kite" if i == 0 else "triangle")

        fot, llm = self._fot(respond, budget=1)
        fot.run(GEOMETRY_Q)
        variant_prompts = [p for p in llm.prompts if "Solve the following problem" in p]
        self.assertGreater(len(variant_prompts), 1)
        for prompt in variant_prompts:
            self.assertNotIn("Candidate answer", prompt)
            self.assertNotIn("Previous answer", prompt)

    def test_tau_blocks_a_single_violation(self):
        """One dissenting variant is a defect somewhere in the orbit, not against a."""
        state = {"n": 0}

        def respond(prompt: str, i: int) -> str:
            if "Solve the following problem" not in prompt:
                return "ANSWER: (B)"
            state["n"] += 1
            # Only the first variant dissents; the rest agree with the candidate.
            shape = "triangle" if state["n"] == 2 else "kite"
            return self._answer_shape(prompt, shape)

        fot, _ = self._fot(respond, budget=1)
        response = fot.run(GEOMETRY_Q)
        self.assertTrue(response.metadata["accepted_fixpoint"])
        self.assertEqual(response.metadata["repairs_used"], 0)

    def test_majority_rule_protects_a_candidate_the_orbit_agrees_with(self):
        """Two dissenters that also disagree with each other must not trigger repair."""
        state = {"n": 0}
        shapes = ["kite", "triangle", "circle", "kite"]

        def respond(prompt: str, i: int) -> str:
            if "Solve the following problem" not in prompt:
                return "ANSWER: (B)"
            shape = shapes[min(state["n"], len(shapes) - 1)]
            state["n"] += 1
            return self._answer_shape(prompt, shape)

        fot, _ = self._fot(respond, budget=1)
        response = fot.run(GEOMETRY_Q)
        self.assertTrue(response.metadata["accepted_fixpoint"])

        # The pilot's asymmetry (tau=1, no majority test) refutes the same orbit.
        state["n"] = 0
        fot, _ = self._fot(respond, budget=1, tau=1, use_majority=False)
        self.assertEqual(fot.run(GEOMETRY_Q).metadata["repairs_used"], 1)

    def test_no_applicable_relation_degenerates_to_solve(self):
        llm = ScriptedLLM(lambda p, i: "ANSWER: 4")
        fot = FoT(llm, task="mgsm", relations=["permute_premises"])
        response = fot.run("Tom has 5 apples. He eats 2. How many are left?")
        self.assertEqual(response.final_answer, "4")
        self.assertEqual(response.num_llm_calls, 1)
        self.assertTrue(response.metadata["accepted_fixpoint"])

    def test_model_applied_relation_uses_pi_mr(self):
        thai = "เป็ดของเจเน็ตวางไข่วันละ 16 ฟอง คำถาม"

        def respond(prompt: str, i: int) -> str:
            if "VARIANT:" in prompt:
                return "VARIANT: Janet's ducks lay 16 eggs a day. Question\nRELATION: same"
            return "ANSWER: 18"

        llm = ScriptedLLM(respond)
        fot = FoT(llm, task="mgsm", relations=["translate_to_english"], budget=1)
        fot.run(thai)
        mr_prompts = [p for p in llm.prompts if "VARIANT:" in p]
        self.assertEqual(len(mr_prompts), 1)
        self.assertIn("Do NOT solve", mr_prompts[0])


# ── 11. Driver loop ────────────────────────────────────────────────────────────
class TestDriver(unittest.TestCase):

    Q = "((1 + 1) * 3) ="

    def test_budget_exhaustion_stops_after_k_rounds(self):
        """A candidate that is refuted every round costs exactly K repairs."""
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
        # Every archived candidate was soundly refuted except the last, so the
        # last one is returned rather than a provably wrong earlier answer.
        self.assertEqual(response.final_answer, "103")

    def test_archive_floors_the_metamorphic_regime_at_solve(self):
        """With no candidate ever corroborated, the initial answer is the floor."""
        state = {"n": 0}
        shapes = ["kite", "triangle", "triangle", "triangle",   # round 1 refutes kite
                  "circle",                                      # repair 1
                  "kite", "kite", "kite",                        # round 2 refutes circle
                  "triangle"]                                    # repair 2 (untested)

        def respond(prompt: str, i: int) -> str:
            if "Solve the following problem" not in prompt and "Original problem" not in prompt:
                return "ANSWER: (B)"
            shape = shapes[min(state["n"], len(shapes) - 1)]
            state["n"] += 1
            options = dict((b.strip(), a) for a, b in re.findall(r"\((\w)\) (.+)", prompt))
            return f"ANSWER: ({options[shape]})"

        llm = ScriptedLLM(respond)
        response = FoT(llm, task="bigbenchhard", subtask="geometric_shapes",
                       budget=2).run(GEOMETRY_Q)
        self.assertTrue(response.metadata["budget_exhausted"])
        self.assertEqual(response.metadata["returned"], "archive")
        self.assertEqual(response.final_answer, "(B)")          # kite: the Solve answer

    def test_metadata_shape(self):
        llm = ScriptedLLM(lambda p, i: "ANSWER: 6")
        response = FoT(llm, task="bigbenchhard",
                       subtask="multistep_arithmetic_two").run(self.Q)
        for key in ("task", "subtask", "regime", "has_checker", "catalogue",
                    "accepted_fixpoint", "returned", "budget", "probes", "tau",
                    "use_majority", "repairs_used", "budget_exhausted",
                    "execute_code", "falsification_regimes", "witness_history"):
            self.assertIn(key, response.metadata)
        self.assertEqual(response.baseline_type, "FoT")

    def test_counters_reset_between_questions(self):
        llm = ScriptedLLM(lambda p, i: "ANSWER: 6")
        fot = FoT(llm, task="bigbenchhard", subtask="multistep_arithmetic_two")
        first = fot.run(self.Q).num_llm_calls
        second = fot.run(self.Q).num_llm_calls
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
