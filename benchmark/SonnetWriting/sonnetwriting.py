"""
Sonnet Writing Benchmark Dataset.

This benchmark evaluates the ability of Large Language Models to write
Shakespearean sonnets following strict structural and lexical constraints.

Task Description:
    Given a rhyme scheme and a handful of required words, write a sonnet
    (ABAB CDCD EFEF GG) that incorporates every word verbatim within the poem.

Dataset:
    ``data/sonnets.jsonl`` — one JSON object per line, in the Meta-prompting /
    BoT ``sonnet_writing`` format::

        {"input":  "Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG,
                    containing each of the following words verbatim:
                    \\"grass\\", \\"value\\", and \\"jail\\".",
         "target": "ABAB CDCD EFEF GG, grass value jail"}

    ``input`` is the prompt shown to the model verbatim; ``target`` carries the
    two gradeable constraints — the rhyme scheme before the comma, the required
    words after it. Each sonnet is evaluated on:
    - Word inclusion: all required words present (case-insensitive, whole-word match)
    - Structure: exactly as many non-empty lines as the scheme has letters (14)
    - Rhyme scheme: lines sharing a scheme letter rhyme (detected via suffix matching)

Scoring:
    score = (words_score + structure_score + rhyme_score) / 3
    where each component is in [0, 1].
    is_correct = True only if all three criteria are fully satisfied (score ≈ 1.0).

References:
    BoT paper: https://arxiv.org/abs/2310.04687
    Meta-prompting Sonnet task: Suzgun & Kalai (2024)

Author: Egor Morozov
"""

import json
import re
from pathlib import Path
from typing import Any

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


# The scheme used by every row of the shipped dataset; also the fallback when a
# ground truth carries only the required words (see ``_parse_target``).
DEFAULT_RHYME_SCHEME = "ABAB CDCD EFEF GG"


# ─────────────────────────────────────────────────────────────────────────────
# Target Parsing Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _parse_target(target: Any) -> tuple[str, list[str]]:
    """Split a dataset target into its rhyme scheme and its required words.

    The dataset states both constraints in one string, scheme first::

        "ABAB CDCD EFEF GG, grass value jail"

    A bare list of words is also accepted, so a caller holding only the words
    (or a hand-written test case) still grades against the default scheme.

    Args:
        target: Either the raw ``target`` string or a list/tuple of words.

    Returns:
        ``(rhyme_scheme, required_words)``. The scheme falls back to
        :data:`DEFAULT_RHYME_SCHEME` when the target names no scheme.
    """
    if isinstance(target, (list, tuple)):
        return DEFAULT_RHYME_SCHEME, [str(w) for w in target]

    if not isinstance(target, str):
        return DEFAULT_RHYME_SCHEME, []

    scheme, sep, words = target.partition(",")
    if not sep:
        # No scheme stated — the whole string is the word list.
        return DEFAULT_RHYME_SCHEME, target.split()

    scheme = scheme.strip() or DEFAULT_RHYME_SCHEME
    return scheme, words.replace(",", " ").split()


def _rhyme_pairs(scheme: str) -> list[tuple[int, int]]:
    """Derive the line pairs that must rhyme from a rhyme scheme.

    Lines sharing a scheme letter form a rhyme group; because the suffix test
    is an equality (hence transitive), consecutive members of a group are
    enough to constrain the whole group. Groups are emitted in order of first
    appearance, so ``"ABAB CDCD EFEF GG"`` yields the canonical
    ``[(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]``.

    Args:
        scheme: A rhyme scheme such as ``"ABAB CDCD EFEF GG"``. Whitespace is
                ignored; each remaining character labels one line.

    Returns:
        The list of zero-indexed line pairs required to rhyme.
    """
    groups: dict[str, list[int]] = {}
    for position, letter in enumerate("".join(scheme.split())):
        groups.setdefault(letter.upper(), []).append(position)

    return [
        (lines[k], lines[k + 1])
        for lines in groups.values()
        for k in range(len(lines) - 1)
    ]


def _scheme_length(scheme: str) -> int:
    """Return the number of lines a rhyme scheme prescribes (14 for a sonnet)."""
    return len("".join(scheme.split()))


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
        - the rhyme scheme named by the problem (ABAB CDCD EFEF GG throughout)
        - must include every specified word verbatim

    Args:
        split: Dataset split (default: ``"test"``).
               Currently only "test" is available; split parameter
               is provided for interface consistency.

    Example::

        ds = SonnetWriting()
        ds.load_dataset()
        print(len(ds))                          # 250

        problem = ds.get_problem(0)
        print(problem.question)                 # The dataset's prompt, verbatim
        print(problem.ground_truth)             # "ABAB CDCD EFEF GG, grass value jail"

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
        """Load the Sonnet Writing dataset from the local JSONL file.

        Reads `benchmark/SonnetWriting/data/sonnets.jsonl` — one JSON object
        per line — and populates ``self._data`` with a list of dictionaries.

        Each dictionary contains:
            - input (str): the prompt shown to the model verbatim
            - target (str): ``"<rhyme scheme>, <required words>"``

        Raises:
            RuntimeError: If the file cannot be found or a line cannot be parsed.
        """
        # Locate the data file relative to this module
        module_dir = Path(__file__).parent
        data_file = module_dir / "data" / "sonnets.jsonl"

        if not data_file.exists():
            raise RuntimeError(
                f"[{self.dataset_name}] Data file not found: {data_file}\n"
                "Please ensure sonnets.jsonl exists in the data directory."
            )

        rows: list[dict] = []
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise RuntimeError(
                            f"[{self.dataset_name}] Failed to parse "
                            f"'{data_file}' line {line_no}: {e}"
                        ) from e
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"[{self.dataset_name}] Failed to load '{data_file}': {e}"
            ) from e

        self._data = rows

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
            - question: the dataset's ``input`` prompt, verbatim
            - ground_truth: the dataset's ``target`` string (scheme + words)
            - metadata: the parsed constraints (rhyme_scheme, words, n_words)

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
        target = row.get("target", "")
        rhyme_scheme, words = _parse_target(target)

        return Problem(
            index=index,
            question=row.get("input", ""),
            ground_truth=target,
            metadata={
                "id": row.get("id", index),
                "rhyme_scheme": rhyme_scheme,
                "words": words,
                "n_words": len(words),
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
        2. Structure (0–1): whether the scheme's line count (14) is met
        3. Rhyme (0–1): proportion of required rhyme pairs that match

        The final score is the average of these three components.
        A sonnet is considered correct only if all three criteria are
        fully satisfied (score ≈ 1.0).

        Args:
            prediction: The model's generated sonnet (raw text).
            ground_truth: The dataset's target string
                ``"ABAB CDCD EFEF GG, grass value jail"``, or a bare list of
                required words (graded against the default scheme).

        Returns:
            EvaluationResult with is_correct, score, and diagnostic details.
        """
        rhyme_scheme, required_words = _parse_target(ground_truth)

        details = {
            "raw_prediction": prediction,
            "rhyme_scheme": rhyme_scheme,
            "required_words": required_words,
        }

        if not required_words:
            details["error"] = (
                "ground_truth must name at least one required word "
                "(e.g. 'ABAB CDCD EFEF GG, grass value jail')."
            )
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                prediction=prediction,
                ground_truth=ground_truth,
                details=details,
            )

        # ──────── 1. Word Inclusion Score ────────────────────────────────

        required_words = [w.lower() for w in required_words]
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

        expected_lines = _scheme_length(rhyme_scheme)

        lines = [line.strip() for line in prediction.split("\n")]
        non_empty_lines = [line for line in lines if line]

        # If the model appended explanation text after the sonnet (e.g.
        # "This sonnet adheres to…"), truncate to the first `expected_lines`
        # non-empty lines so the explanation block does not inflate the count.
        if len(non_empty_lines) > expected_lines:
            non_empty_lines = non_empty_lines[:expected_lines]

        structure_correct = len(non_empty_lines) == expected_lines
        structure_score = 1.0 if structure_correct else 0.0

        details["line_count"] = len(non_empty_lines)
        details["structure_score"] = structure_score
        details["lines"] = non_empty_lines

        # ──────── 3. Rhyme Scheme Score ──────────────────────────────────

        rhyme_score = 0.0
        matched_pairs = 0
        rhyme_details = []

        # Take up to `expected_lines` lines and extract end words
        # (fewer if the model produced a shorter poem)
        sonnet_lines = non_empty_lines[:expected_lines]
        end_words = [_get_last_word(line) for line in sonnet_lines]

        # Line pairs the scheme requires to rhyme (zero-indexed)
        rhyme_pairs = _rhyme_pairs(rhyme_scheme)

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
            "2. Follow the rhyme scheme stated in the prompt "
            "(ABAB CDCD EFEF GG)\n"
            "3. Include every specified word verbatim in the poem\n"
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

        The ground truth holds only the constraints, not a finished poem, so
        the demonstration is one canonical worked example: an input phrased
        exactly like the dataset's own prompts and an output that is a
        complete 14-line sonnet in ABAB CDCD EFEF GG form containing every
        required word. A single shot is used because each example is a full
        poem.
        """
        return (
            "Input: Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, "
            'containing each of the following words verbatim: "moon", '
            '"river", and "silent".\n'
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
