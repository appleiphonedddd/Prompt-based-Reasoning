"""
Sonnet Writing Benchmark Dataset.

This benchmark evaluates the ability of Large Language Models to write
Shakespearean sonnets following strict structural and lexical constraints.

Task Description:
    Given five specific words, write a 14-line Shakespearean sonnet
    (ABAB CDCD EFEF GG rhyme scheme) that incorporates all five words
    verbatim within the poem.

Dataset:
    154 prompts derived from Shakespeare's canonical sonnets, each specifying
    five thematically relevant words (keywords sourced from:
    github.com/iljones00/Shakespearean-Sonnets-GPT).
    Each sonnet is evaluated on:
    - Word inclusion: all required words present (case-insensitive, whole-word match)
    - Structure: exactly 14 non-empty lines
    - Rhyme scheme: valid ABAB CDCD EFEF GG pattern (detected via suffix matching)

Scoring:
    score = (words_score + structure_score + rhyme_score) / 3
    where each component is in [0, 1].
    is_correct = True only if all three criteria are fully satisfied (score ≈ 1.0).

References:
    BoT paper: https://arxiv.org/abs/2310.04687
    Meta-prompting Sonnet task: Suzgun & Kalai (2024)
    Keyword dataset: github.com/iljones00/Shakespearean-Sonnets-GPT

Author: Egor Morozov
"""

import json
import re
from pathlib import Path
from typing import Any

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


# ─────────────────────────────────────────────────────────────────────────────
# Rhyme Detection Utility
# ─────────────────────────────────────────────────────────────────────────────

def _get_last_word(line: str) -> str:
    """Extract the last word from a line, removing punctuation.

    Args:
        line: A line of poetry.

    Returns:
        The last word stripped of trailing punctuation and lowercased.
    """
    # Remove leading/trailing whitespace
    line = line.strip()
    if not line:
        return ""

    # Remove trailing punctuation
    while line and not line[-1].isalnum():
        line = line[:-1]

    # Split on whitespace and get the last word
    words = line.split()
    return words[-1].lower() if words else ""


def _words_rhyme(word1: str, word2: str, suffix_len: int = 3) -> bool:
    """Check if two words rhyme based on suffix matching.

    Uses the last N characters as a heuristic for rhyme detection.
    For example, "moon" and "soon" both end in "oon".

    Args:
        word1: First word (lowercased).
        word2: Second word (lowercased).
        suffix_len: Number of characters to compare (default 3).

    Returns:
        True if the words share the same suffix of length suffix_len.
    """
    if not word1 or not word2:
        return False

    # Get the suffix (last N chars) of each word
    suffix1 = word1[-suffix_len:] if len(word1) >= suffix_len else word1
    suffix2 = word2[-suffix_len:] if len(word2) >= suffix_len else word2

    return suffix1 == suffix2


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Implementation
# ─────────────────────────────────────────────────────────────────────────────

class SonnetWriting(DatasetBase):
    """Benchmark for evaluating Shakespearean sonnet generation.

    Constraints:
        - 14 lines total (iambic pentameter not enforced)
        - ABAB CDCD EFEF GG rhyme scheme
        - Must include five specified words verbatim

    Args:
        split: Dataset split (default: ``"test"``).
               Currently only "test" is available; split parameter
               is provided for interface consistency.

    Example::

        ds = SonnetWriting()
        ds.load_dataset()
        print(len(ds))                          # 154

        problem = ds.get_problem(0)
        print(problem.question)                 # Task description + 5 words
        print(problem.ground_truth)             # List of 5 required words

        sonnet_output = "Write a beautiful sonnet about..."  # from LLM
        result = ds.evaluate_answer(sonnet_output, problem.ground_truth)
        print(result.score, result.is_correct)
    """

    def __init__(self, split: str = "test"):
        """Initialise the SonnetWriting benchmark.

        Args:
            split: HuggingFace split name (kept for interface consistency).
                   Currently only "test" is implemented.
        """
        super().__init__(split=split, dataset_name="SonnetWriting")

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Load the Sonnet Writing dataset from local JSON file.

        Reads `benchmark/SonnetWriting/data/sonnets.json` and populates
        ``self._data`` with a list of dictionaries.

        Each dictionary contains:
            - id (int): Shakespeare sonnet index (0–153, appears twice per sonnet)
            - n_words (int): required word count (3 or 5)
            - words (list[str]): required words to incorporate verbatim

        Raises:
            RuntimeError: If the JSON file cannot be found or parsed.
        """
        # Locate the data file relative to this module
        module_dir = Path(__file__).parent
        data_file = module_dir / "data" / "sonnets.json"

        if not data_file.exists():
            raise RuntimeError(
                f"[{self.dataset_name}] Data file not found: {data_file}\n"
                "Please ensure sonnets.json exists in the data directory."
            )

        try:
            with open(data_file, "r", encoding="utf-8") as f:
                self._data = json.load(f)

        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"[{self.dataset_name}] Failed to parse '{data_file}': {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"[{self.dataset_name}] Failed to load '{data_file}': {e}"
            ) from e

        print(
            f"[{self.dataset_name}] Loaded {len(self._data)} problems "
            f"(split='{self.split}')."
        )

    def get_problem(self, index: int) -> Problem:
        """Return the sonnet writing problem at the given index.

        Args:
            index: Zero-based index into the dataset.

        Returns:
            A Problem with:
            - question: Task description with required words (3 or 5)
            - ground_truth: List of required words to incorporate
            - metadata: Dataset-specific information including n_words and sonnet_id

        Raises:
            RuntimeError: If dataset has not been loaded.
            IndexError: If index is out of range.
        """
        self._ensure_loaded()

        if index < 0 or index >= len(self._data):
            raise IndexError(
                f"[{self.dataset_name}] Index {index} out of range "
                f"[0, {len(self._data) - 1}]."
            )

        row = self._data[index]
        words = row.get("words", [])

        words_str = ", ".join(f'"{w}"' for w in words)
        n = len(words)
        question = (
            "Write a Shakespearean sonnet (ABAB CDCD EFEF GG) "
            f"using these {n} words verbatim:\n"
            f"{words_str}\n\n"
            "Your sonnet must:\n"
            "1. Have exactly 14 lines\n"
            "2. Follow the rhyme scheme ABAB CDCD EFEF GG\n"
            f"3. Include all {n} words exactly as written"
        )

        return Problem(
            index=index,
            question=question,
            ground_truth=words,
            metadata={
                "id": row.get("id", index),
                "n_words": row.get("n_words", len(words)),
                "raw_words": words,
            },
        )

    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Evaluate a sonnet against the specified criteria.

        Three components are scored independently:
        1. Word inclusion (0–1): fraction of required words present
        2. Structure (0–1): whether exactly 14 lines are present
        3. Rhyme (0–1): proportion of required rhyme pairs that match

        The final score is the average of these three components.
        A sonnet is considered correct only if all three criteria are
        fully satisfied (score ≈ 1.0).

        Args:
            prediction: The model's generated sonnet (raw text).
            ground_truth: List of three required words.

        Returns:
            EvaluationResult with is_correct, score, and diagnostic details.
        """
        details = {
            "raw_prediction": prediction,
            "required_words": ground_truth,
        }

        if not isinstance(ground_truth, list) or len(ground_truth) == 0:
            details["error"] = "ground_truth must be a non-empty list of words."
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                prediction=prediction,
                ground_truth=ground_truth,
                details=details,
            )

        # ──────── 1. Word Inclusion Score ────────────────────────────────

        required_words = [w.lower() for w in ground_truth]
        words_found = 0

        for word in required_words:
            # Check for whole-word match (case-insensitive) using \b boundaries
            pattern = rf"\b{re.escape(word)}\b"
            if re.search(pattern, prediction, re.IGNORECASE):
                words_found += 1

        words_score = words_found / len(required_words)
        details["words_found"] = words_found
        details["words_score"] = words_score

        # ──────── 2. Structure Score ─────────────────────────────────────

        lines = [line.strip() for line in prediction.split("\n")]
        non_empty_lines = [line for line in lines if line]

        # If the model appended explanation text after the sonnet (e.g.
        # "This sonnet adheres to…"), truncate to the first 14 non-empty
        # lines so the explanation block does not inflate the line count.
        if len(non_empty_lines) > 14:
            non_empty_lines = non_empty_lines[:14]

        structure_correct = len(non_empty_lines) == 14
        structure_score = 1.0 if structure_correct else 0.0

        details["line_count"] = len(non_empty_lines)
        details["structure_score"] = structure_score
        details["lines"] = non_empty_lines

        # ──────── 3. Rhyme Scheme Score ──────────────────────────────────

        rhyme_score = 0.0
        matched_pairs = 0
        rhyme_details = []

        # Take up to 14 lines and extract end words (fewer if not enough lines)
        sonnet_lines = non_empty_lines[:14] if non_empty_lines else []
        end_words = [_get_last_word(line) for line in sonnet_lines]

        # ABAB CDCD EFEF GG rhyme pairs (zero-indexed)
        rhyme_pairs = [
            (0, 2), (1, 3),      # ABAB
            (4, 6), (5, 7),      # CDCD
            (8, 10), (9, 11),    # EFEF
            (12, 13),            # GG
        ]

        for i, j in rhyme_pairs:
            if i < len(end_words) and j < len(end_words):
                rhyme_check = _words_rhyme(end_words[i], end_words[j])
                if rhyme_check:
                    matched_pairs += 1
                rhyme_details.append({
                    "pair": (i, j),
                    "word1": end_words[i],
                    "word2": end_words[j],
                    "rhyme": rhyme_check,
                })

        if len(rhyme_pairs) > 0:
            rhyme_score = matched_pairs / len(rhyme_pairs)

        details["rhyme_pairs_matched"] = matched_pairs
        details["rhyme_pairs_total"] = len(rhyme_pairs)
        details["rhyme_score"] = rhyme_score
        details["rhyme_details"] = rhyme_details

        # ──────── 4. Combined Score and Correctness ──────────────────────

        score = (words_score + structure_score + rhyme_score) / 3.0

        # A sonnet is correct only if all three criteria are fully met
        is_correct = (
            words_score == 1.0 and
            structure_score == 1.0 and
            rhyme_score == 1.0
        )

        details["combined_score"] = score
        details["is_correct"] = is_correct

        return EvaluationResult(
            is_correct=is_correct,
            score=score,
            prediction=prediction,
            ground_truth=ground_truth,
            details=details,
        )

    # ── Optional hook overrides ────────────────────────────────────────

    def get_instruction(self) -> str:
        """Return the task-specific instruction for sonnet writing.

        Provides clear guidance on the sonnet structure and constraints.
        """
        return (
            "Write a Shakespearean sonnet following these strict rules:\n"
            "1. The sonnet must be exactly 14 lines\n"
            "2. Follow the rhyme scheme ABAB CDCD EFEF GG\n"
            "3. Include the three specified words verbatim in the poem\n"
            "4. Each line should be roughly 10 syllables (iambic pentameter) "
            "if possible, but this is secondary to meeting the constraints above"
        )

    def get_system_prompt(self) -> str:
        """Return a system prompt setting a poetic persona."""
        return (
            "You are a classical poet versed in Shakespearean verse. "
            "Your task is to compose sonnets following the traditional "
            "English sonnet form with precision and artistic merit. "
            "Always honor the structural constraints given."
        )

    def get_demonstrations(self, n_shot: int = 1) -> str:
        """Hand-crafted sonnet demonstration for RoT warm-up.

        The ground truth holds only the required words, not a finished poem,
        so the demonstration is one canonical worked example: an input that
        names the verbatim words and an output that is a complete 14-line
        sonnet in ABAB CDCD EFEF GG form containing all of them. A single
        shot is used because each example is a full poem.
        """
        return (
            'Input: Write a Shakespearean sonnet (ABAB CDCD EFEF GG) using '
            'these 3 words verbatim: "moon", "river", "silent"\n'
            "Output:\n"
            "Beneath the pale and ever-watchful moon,\n"
            "The quiet waters of the river gleam,\n"
            "And night unfolds her dark and tender boon,\n"
            "While silent shadows drift as in a dream.\n"
            "The willows bend to kiss the cooling tide,\n"
            "Their branches tracing letters on the stream,\n"
            "As secrets in the rippling current hide,\n"
            "And starlight scatters many a fragile beam.\n"
            "No voice disturbs the stillness of the hour,\n"
            "The world has hushed its restless, daily strife,\n"
            "And every leaf submits to slumber's power,\n"
            "Surrendering the noise of waking life.\n"
            "So let me linger here till break of day,\n"
            "Where moon and silent river softly stay."
        )
