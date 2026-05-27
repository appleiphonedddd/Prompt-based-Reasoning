"""
Buffer of Thoughts (BoT) Implementation.

BoT is a thought-augmented reasoning framework with three core components:

    1. Problem Distiller  — extracts key variables, constraints, and a
                            high-level task description from the raw question.

    2. Meta-Buffer        — a lightweight in-memory (or file-backed) library of
                            generalised *thought-templates* organised into six
                            categories: Text Comprehension, Creative Language
                            Generation, Common Sense Reasoning, Mathematical
                            Reasoning, Code Programming, Application Scheduling.

    3. Buffer Manager     — after every solve, distils a new thought-template
                            from the (distilled problem, solution) pair and
                            conditionally inserts it into the meta-buffer if it
                            is sufficiently novel (similarity < threshold δ).

Bugs fixed from original document version
──────────────────────────────────────────
    A. BufferManager: removed @property decorators that shadowed the plain
       instance attributes total_input_tokens / total_output_tokens / num_calls,
       causing infinite RecursionError on every counter access.
    B. distil_and_update: corrected `_DISTILLATION_SYSTEM` → `DISTILLATION_SYSTEM`.
    C. distil_and_update: corrected `self._parse_template` → `self.parse_template`.
    D. BoT.run() / __repr__: corrected `self._buffer_manager` → `self.buffer_manager`
       and `self._meta_buffer` → `self.meta_buffer`.
    E. BoT: removed the @property stubs for meta_buffer / buffer_manager that
       returned non-existent private names (_meta_buffer / _buffer_manager).
    F. MetaBuffer.load(): now handles empty JSON files gracefully instead of
       raising JSONDecodeError.

Reference:
    Yang et al., "Buffer of Thoughts: Thought-Augmented Reasoning with Large
    Language Models", NeurIPS 2024.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from baseline.basebaseline import BaseBaseline, BaselineResponse
from models.base import BaseLLM

# Lazy-loaded singleton — model is downloaded once and reused across all MetaBuffer instances.
_EMBED_MODEL: Optional[SentenceTransformer] = None

def _get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL

def _embed(text: str) -> np.ndarray:
    return _get_embed_model().encode(text, normalize_embeddings=True)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are already L2-normalised, so dot product == cosine similarity.
    return float(np.dot(a, b))


# Matches the "### Distilled Task" section emitted by PROBLEM_DISTILLER_SYSTEM,
# capturing everything up to the next "###" header (or end of text).
_DISTILLED_TASK_RE = re.compile(
    r"#+[ \t]*Distilled[ \t]+Task[ \t]*:?[ \t]*\n(.*?)(?=\n#+[ \t]|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def extract_distilled_task(distilled_info: str) -> str:
    """Return only the concise '### Distilled Task' paragraph for retrieval.

    Template retrieval (Eq. 2 of the paper) compares the distilled problem
    against terse template *descriptions*. Embedding the full multi-section
    distiller output (Key Information + Constraints + Distilled Task) dilutes
    the semantic signal and depresses cosine similarity, which makes the
    meta-buffer fire far less often than intended. Isolating the short task
    paragraph keeps the query in the same register as the stored descriptions.

    Falls back to the full text when the section is absent (e.g. the model did
    not follow the expected format).
    """
    m = _DISTILLED_TASK_RE.search(distilled_info)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return distilled_info.strip()


# ── Code extraction & execution ─────────────────────────────────────────────
# BoT's headline advantage on Game-of-24 / programming / chess tasks comes from
# *executing* the instantiated code template rather than reasoning about it in
# prose (paper §4.1). This is opt-in (BoT(execute_code=True)) because it runs
# untrusted, model-generated Python — only enable it on a trusted/sandboxed host.

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> Optional[str]:
    """Return the Python source from a markdown ```python ... ``` block.

    When several fenced blocks are present the longest one is chosen (the full
    program rather than a snippet). Returns None when no fenced code is found,
    so callers can cleanly fall back to textual answer extraction.
    """
    blocks = [b.strip() for b in _CODE_BLOCK_RE.findall(text) if b.strip()]
    if not blocks:
        return None
    return max(blocks, key=len)


@dataclass
class CodeExecutionResult:
    """Outcome of running a generated program in a subprocess."""
    success: bool
    output: str   # stdout, stripped
    error: str    # stderr / exception / timeout message
    code: str     # the source that was executed


def run_code(code: str, timeout: float = 10.0) -> CodeExecutionResult:
    """Execute Python ``code`` in an isolated subprocess and capture stdout.

    Running in a separate interpreter (rather than inline ``exec``) means
    infinite loops are bounded by ``timeout`` (e.g. a brute-force Game-of-24
    search), crashes / ``sys.exit`` cannot take down the harness, and the
    namespace stays clean. ``-I`` (isolated mode) ignores PYTHON* env vars and
    the user site directory while still exposing installed packages such as
    ``chess`` or ``itertools``.

    A run counts as failed if it raises, exits non-zero, times out, or prints
    nothing (a program that computes but never prints has no usable answer).
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CodeExecutionResult(False, "", f"Execution timed out after {timeout:g}s", code)
    except Exception as exc:  # pragma: no cover - interpreter launch failure
        return CodeExecutionResult(False, "", f"Failed to launch interpreter: {exc}", code)

    stdout = proc.stdout.strip()
    if proc.returncode != 0:
        error = proc.stderr.strip() or f"Process exited with code {proc.returncode}"
        return CodeExecutionResult(False, stdout, error, code)
    if not stdout:
        return CodeExecutionResult(
            False, "", "Code ran without error but printed nothing to stdout.", code
        )
    return CodeExecutionResult(True, stdout, "", code)


THOUGHT_CATEGORIES = [
    "Text Comprehension",
    "Creative Language Generation",
    "Common Sense Reasoning",
    "Mathematical Reasoning",
    "Code Programming",
    "Application Scheduling",
]

# Seed templates from BoT paper Appendix B.1 — verbatim from the paper.
# Each entry has three sections matching the paper exactly:
#   description          = Task Description  (used for embedding retrieval)
#   solution_description = Solution Description
#   template             = Thought Template
SEED_TEMPLATES: list = [
    {
        "index": 0,
        "category": "Text Comprehension",
        "description": (
            "The task involves analyzing a table with various attributes of penguins, "
            "such as name, age, height, and weight, and answering questions about these "
            "attributes. The table may be updated with new entries, and additional "
            "context or comparisons may be provided in natural language."
        ),
        "solution_description": (
            "To accurately answer questions about the penguins' attributes, one must "
            "be able to interpret the data presented in tabular form, understand any "
            "additional information provided in natural language, and apply logical "
            "reasoning to identify the correct attribute based on the question asked."
        ),
        "template": (
            "Step 1: Parse the initial table, extracting the header information and "
            "each penguin's attributes into a structured format (e.g., a list of "
            "dictionaries).\n"
            "Step 2: Read and integrate any additional natural language information "
            "that updates or adds to the table, ensuring the data remains consistent.\n"
            "Step 3: Identify the attribute in question (e.g., oldest penguin, heaviest "
            "penguin) and the corresponding column in the table.\n"
            "Step 4: Apply logical reasoning to compare the relevant attribute across "
            "all entries to find the correct answer (e.g., the highest age for the "
            "oldest penguin).\n"
            "Step 5: Select the answer from the provided options that matches the "
            "result of the logical comparison."
        ),
    },
    {
        "index": 1,
        "category": "Creative Language Generation",
        "description": (
            "The task is to generate a sonnet that adheres to the traditional English "
            "sonnet rhyme scheme of \"ABAB CDCD EFEF GG\" and includes three specific "
            "words verbatim in the text."
        ),
        "solution_description": (
            "Writing a sonnet involves crafting 14 lines of poetry that follow a "
            "specific rhyme pattern. The lines are typically in iambic pentameter, "
            "though flexibility in rhythm can be allowed for creative reasons. The "
            "given rhyme scheme dictates the end sounds of each line, ensuring a "
            "structured poetic form. Incorporating the three provided words verbatim "
            "requires strategic placement within the lines to maintain the poem's "
            "coherence and thematic unity."
        ),
        "template": (
            "Step 1: Identify the three words that must be included in the sonnet.\n"
            "Step 2: Understand the rhyme scheme \"ABAB CDCD EFEF GG\" and prepare a "
            "list of rhyming words that could be used.\n"
            "Step 3: Develop a theme or story for the sonnet that can naturally "
            "incorporate the three provided words.\n"
            "Step 4: Begin drafting the sonnet by writing the first quatrain (four "
            "lines) following the \"ABAB\" rhyme scheme, ensuring one or more of the "
            "provided words are included.\n"
            "Step 5: Continue with the second quatrain \"CDCD,\" the third quatrain "
            "\"EFEF,\" and finally the closing couplet \"GG,\" each time incorporating "
            "the provided words as needed.\n"
            "Step 6: Review the sonnet for coherence, flow, and adherence to the "
            "rhyme scheme, making adjustments as necessary."
        ),
    },
    {
        "index": 2,
        "category": "Common Sense Reasoning",
        "description": (
            "Given a specific date and an event, such as a holiday or historical "
            "event, determine the following date."
        ),
        "solution_description": (
            "To determine the next date, we need to consider the structure of the "
            "calendar, the number of days in each month, and whether it's a leap year. "
            "Typically, the number of days in a month is fixed, except February may "
            "vary due to leap years. The next day in a year is usually the date "
            "increased by one day unless it's the end of the month, then the next day "
            "will be the first day of the following month. For the end of the year, "
            "the next day will be January 1st of the following year."
        ),
        "template": (
            "Step 1: Identify the given date's month and day number.\n"
            "Step 2: Check if it's the end of the month; if so, confirm the start "
            "date of the next month.\n"
            "Step 3: If it's not the end of the month, simply add one to the day "
            "number.\n"
            "Step 4: Pay special attention to the end of the year, ensuring the year "
            "increments."
        ),
    },
    {
        "index": 3,
        "category": "Code Programming",
        "description": (
            "When given a list of numbers, try to utilize 4 basic mathematical "
            "operations (+-*/) to get a target number."
        ),
        "solution_description": "",
        "template": (
            "from itertools import permutations, product\n\n"
            "def perform_operation(a, b, operation):\n"
            "    # Define the operation logic (e.g., addition, subtraction, etc.).\n"
            "    pass\n\n"
            "def evaluate_sequence(sequence, operations):\n"
            "    # Apply operations to the sequence and check if the result\n"
            "    # meets the criteria.\n"
            "    pass\n\n"
            "def generate_combinations(elements, operations):\n"
            "    # Generate all possible combinations of elements and operations.\n"
            "    pass\n\n"
            "def format_solution(sequence, operations):\n"
            "    # Format the sequence and operations into a human-readable string.\n"
            "    pass\n\n"
            "def find_solution(input_elements, target_result):\n"
            "    # Data Input Handling\n"
            "    # Validate and preprocess input data if necessary.\n\n"
            "    # Core Algorithm Logic\n"
            "    for sequence in permutations(input_elements):\n"
            "        for operation_combination in generate_combinations(\n"
            "                sequence, operations):\n"
            "            try:\n"
            "                if evaluate_sequence(sequence,\n"
            "                        operation_combination) == target_result:\n"
            "                    # Data Output Formatting\n"
            "                    return format_solution(sequence,\n"
            "                            operation_combination)\n"
            "            except Exception as e:\n"
            "                # Error Handling\n"
            "                # Handle specific exceptions that may occur\n"
            "                # during evaluation.\n"
            "                continue\n\n"
            "    # If no solution is found after all iterations, return a\n"
            "    # default message.\n"
            "    # return No solution found message\n"
            "    return\n\n"
            "# Example usage:\n"
            "input_elements = [1, 7, 10, 3]\n"
            "target_result = 24\n"
            "print(find_solution(input_elements, target_result))"
        ),
    },
    {
        "index": 4,
        "category": "Application Scheduling",
        "description": (
            "Given some Chess moves in SAN, update the chess board state."
        ),
        "solution_description": "",
        "template": (
            "import chess\n\n"
            "def find_checkmate_move(moves_san):\n"
            "    # Initialize a new chess board\n"
            "    board = chess.Board()\n\n"
            "    # Apply the moves to the board\n"
            "    for move_san in moves_san:\n"
            "        # Remove move numbers and periods (e.g., \"1.\" or \"2.\")\n"
            "        if len(move_san.split('. ')) > 1:\n"
            "            move_san = move_san.split('. ')[1]\n"
            "        # Skip empty strings resulting from the removal\n"
            "        if move_san:\n"
            "            # Apply each move in SAN format to the board\n"
            "            move = board.parse_san(move_san)\n"
            "            board.push(move)\n\n"
            "    # Generate all possible legal moves from the current position\n"
            "    for move in board.legal_moves:\n"
            "        # Make the move on a copy of the board to test the result\n"
            "        board_copy = board.copy()\n"
            "        board_copy.push(move)\n\n"
            "        # Check if the move results in a checkmate\n"
            "        if board_copy.is_checkmate():\n"
            "            # Return the move that results in checkmate in SAN format\n"
            "            return board.san(move)\n\n"
            "    # return No solution found message\n"
            "    return\n\n"
            "#Example usage:\n"
            "input = '......'\n"
            "# Check input format and transform the input into legal format\n"
            "# Remove move numbers and periods (e.g., \"1.\" or \"2.\")\n"
            "checkmate_move = find_checkmate_move(moves_san)\n"
            "print(checkmate_move)"
        ),
    },
    {
        "index": 5,
        "category": "Mathematical Reasoning",
        "description": (
            "Solve a quadratic equation of the form ax^2 + bx + c = 0 considering "
            "any situations."
        ),
        "solution_description": (
            "To solve any quadratic equation of the form ax^2 + bx + c = 0, we can "
            "follow a general approach based on the method described. Here is the "
            "structured template for solving such equations:"
        ),
        "template": (
            "Step 1: Calculate the Discriminant\n"
            "- Compute the discriminant D using the formula D = b^2 - 4ac.\n\n"
            "Step 2: Determine the Nature of the Roots\n"
            "- If D > 0, the equation has two distinct real roots.\n"
            "- If D = 0, the equation has exactly one real root (also known as a "
            "repeated or double root).\n"
            "- If D < 0, the equation has two complex roots.\n\n"
            "Step 3: Compute the Roots\n"
            "- For D >= 0, calculate the roots using the formula "
            "x = (-b +/- sqrt(D)) / (2a).\n"
            "- For D < 0, calculate the real and imaginary parts of the complex roots "
            "using the formula x = -b/(2a) +/- sqrt(-D)/(2a) * i, "
            "where i is the imaginary unit."
        ),
    },
]


@dataclass
class ThoughtTemplate:
    index: int
    category: str
    description: str        # Task Description — used for embedding retrieval
    template: str           # Thought Template steps / code
    solution_description: str = ""  # Solution Description (empty for code templates)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThoughtTemplate":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})



class MetaBuffer:
    def __init__(
        self,
        buffer_path: Optional[str] = None,
        init_templates: Optional[List[ThoughtTemplate]] = None,
    ) -> None:
        self.buffer: List[ThoughtTemplate] = []
        self._embeddings: List[np.ndarray] = []  # parallel to self.buffer
        self.buffer_path = buffer_path
        if buffer_path and os.path.isfile(buffer_path):
            self.load(buffer_path)
        elif init_templates:
            self.buffer = list(init_templates)
            self._embeddings = [_embed(t.description) for t in self.buffer]

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        if not content:
            return  # FIX-F: empty file handled gracefully
        raw: List[Dict[str, Any]] = json.loads(content)
        self.buffer = [ThoughtTemplate.from_dict(d) for d in raw]
        self._embeddings = [_embed(t.description) for t in self.buffer]

    def save(self) -> None:
        if self.buffer_path:
            with open(self.buffer_path, "w", encoding="utf-8") as fh:
                json.dump([t.to_dict() for t in self.buffer], fh, indent=2, ensure_ascii=False)

    @property
    def size(self) -> int:
        return len(self.buffer)

    def retrieve(self, distilled_info: str, threshold: float = 0.5) -> Optional[ThoughtTemplate]:
        if not self.buffer:
            return None
        query_emb = _embed(distilled_info)
        scores = [_cosine_sim(query_emb, e) for e in self._embeddings]
        best_idx = int(np.argmax(scores))
        return self.buffer[best_idx] if scores[best_idx] >= threshold else None

    def add(self, template: ThoughtTemplate, threshold: float = 0.5) -> bool:
        new_emb = _embed(template.description)
        if self.buffer:
            max_sim = max(_cosine_sim(new_emb, e) for e in self._embeddings)
            if max_sim >= threshold:
                return False
        template.index = len(self.buffer)
        self.buffer.append(template)
        self._embeddings.append(new_emb)
        self.save()
        return True

    def all_templates(self) -> List[ThoughtTemplate]:
        return list(self.buffer)


PROBLEM_DISTILLER_SYSTEM = """\
You are a highly professional expert in information distillation.
Your goal is to extract the essential information needed to solve a problem \
and translate it into a structured, high-level description.

Given the user's question, produce a distilled task description that includes:

1. Key information
   - Identify all essential variables, parameters, and given values.
   - Clarify what is known and what must be found.

2. Restrictions / Constraints
   - State the objective clearly.
   - List all explicit and implicit constraints.

3. Distilled task
   - Generalise the specific problem into a *meta-problem* description.
   - Keep this section concise (≤ 5 sentences).

Return your response in the following format:
### Key Information
<bullet list>

### Constraints
<bullet list>

### Distilled Task
<short paragraph>
"""

DISTILLATION_SYSTEM = """\
You are an expert in extracting high-level reasoning paradigms from problem solutions.
Given a problem description and its solution, extract a generalised thought-template \
that can be reused for similar problems.

Return your response in the following JSON format (and ONLY this JSON):
{
  "category": "<one of: Text Comprehension, Creative Language Generation, Common Sense Reasoning, Mathematical Reasoning, Code Programming, Application Scheduling>",
  "description": "<Task Description: one or two sentences describing the task type for retrieval>",
  "solution_description": "<Solution Description: explain the general approach and key considerations for solving this type of problem; leave empty string for code-only templates>",
  "template": "<Thought Template: the full step-by-step thought template or code skeleton>"
}
"""

INSTANTIATION_SYSTEM = """\
You are a Meta Reasoner with deep expertise across Computer Science, Mathematics, \
Physics, Literature, History, Chemistry, Logic, and Language.

You will be given:
  (a) a distilled problem description, and
  (b) a thought-template that provides a high-level reasoning structure.

Format your final answer as:
### Reasoning
<step-by-step reasoning following the template>

### Answer
<concise final answer>
"""

NEW_TASK_SYSTEM = """\
You are a Meta Reasoner with deep expertise across Computer Science, Mathematics, \
Physics, Literature, History, Chemistry, Logic, and Language.

You will be given a distilled problem description for a task that has no prior \
reasoning template. First choose ONE of the following three general reasoning \
structures that best fits the problem:

  i)   Prompt-based structure   — best for Common Sense Reasoning and Application
       Scheduling tasks.
  ii)  Procedure-based structure — best for creative tasks such as Creative
       Language Generation and Text Comprehension.
  iii) Programming-based structure — best for Mathematical Reasoning and Code
       Programming; it can also transform a real-world problem into a program
       that is solved efficiently.

Format your final answer as:
### Reasoning Structure Chosen
<one of: Prompt-based, Procedure-based, Programming-based>

### Reasoning
<step-by-step reasoning following the chosen structure>

### Answer
<concise final answer>
"""

# Categories whose templates are code-bearing; when execute_code is on we ask the
# instantiation step to emit a runnable program for these.
CODE_CATEGORIES = {"Code Programming", "Application Scheduling", "Mathematical Reasoning"}

# Appended to the instantiation prompt (code-category template hit) when
# execute_code is enabled, so the model emits a program we can actually run.
CODE_EXECUTION_DIRECTIVE = """

IMPORTANT — your solution will be executed automatically. Implement it as a \
COMPLETE, self-contained Python program that computes the answer and prints ONLY \
the final answer to stdout. Put the entire program in a single ```python ... ``` \
code block. Do not print explanations."""

# Softer variant for the new-task path, conditioned on the model's own choice of
# the Programming-based structure.
NEW_TASK_CODE_DIRECTIVE = """

If you choose the Programming-based structure, implement the solution as a \
COMPLETE, self-contained Python program that prints ONLY the final answer to \
stdout, wrapped in a single ```python ... ``` code block — it will be executed."""

INSPECTOR_SYSTEM = """\
You are a meticulous Python debugger. The program below was written to solve a \
problem but failed when executed. Using the code and the execution error, return a \
corrected, COMPLETE Python program that runs without errors and prints ONLY the \
final answer to stdout.

Return ONLY the corrected program inside a single ```python ... ``` code block."""


class BufferManager:
    def __init__(
        self,
        meta_buffer: MetaBuffer,
        llm: BaseLLM,
        similarity_threshold: float = 0.5,
        distill_temperature: float = 0.2,
    ) -> None:
        self.meta_buffer = meta_buffer
        self.llm = llm
        self.threshold = similarity_threshold
        self.distill_temperature = distill_temperature
        # Plain instance attributes — NOT @property (FIX-A: avoids RecursionError)
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.num_calls: int = 0

    def reset_counters(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.num_calls = 0

    def distil_and_update(self, distilled_info: str, solution: str) -> Optional[ThoughtTemplate]:
        prompt = (
            f"{DISTILLATION_SYSTEM}\n\n"   # FIX-B: was _DISTILLATION_SYSTEM
            f"[Problem Description]\n{distilled_info}\n\n"
            f"[Solution Steps or Code]\n{solution}"
        )
        response = self.llm.generate(prompt, temperature=self.distill_temperature)
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.num_calls += 1
        template = self.parse_template(response.content.strip())  # FIX-C
        if template is None:
            return None
        return template if self.meta_buffer.add(template, threshold=self.threshold) else None

    @staticmethod
    def parse_template(raw: str) -> Optional[ThoughtTemplate]:
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]+\}", raw)
            if not m:
                return None
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return None
        description = data.get("description", "")
        template_text = data.get("template", "")
        if not description or not template_text:
            return None
        return ThoughtTemplate(
            index=-1,
            category=data.get("category", "Common Sense Reasoning"),
            description=description,
            template=template_text,
        )


class BoT(BaseBaseline):
    """Buffer of Thoughts (BoT) prompting baseline."""

    def __init__(
        self,
        llm: BaseLLM,
        similarity_threshold: float = 0.5,
        buffer_path: Optional[str] = None,
        distill_temperature: float = 0.2,
        instantiation_temperature: float = 0.1,
        update_buffer: bool = True,
        init_templates: Optional[List[ThoughtTemplate]] = None,
        execute_code: bool = False,
        max_code_repairs: int = 3,
        code_timeout: float = 10.0,
    ) -> None:
        super().__init__(llm, baseline_name="BoT")
        self.similarity_threshold = similarity_threshold
        self.distill_temperature = distill_temperature
        self.instantiation_temperature = instantiation_temperature
        self.update_buffer = update_buffer
        # Code execution (paper §4.1) — runs untrusted model-generated Python, so
        # it is OFF by default; enable only on a trusted/sandboxed host.
        self.execute_code = execute_code
        self.max_code_repairs = max_code_repairs
        self.code_timeout = code_timeout
        # Fall back to paper's seed templates when the buffer would otherwise be empty.
        effective_init = init_templates
        if effective_init is None and not (buffer_path and os.path.isfile(buffer_path)):
            effective_init = [ThoughtTemplate.from_dict(d) for d in SEED_TEMPLATES]
        # Public attributes — no @property aliases needed (FIX-E)
        self.meta_buffer = MetaBuffer(buffer_path=buffer_path, init_templates=effective_init)
        self.buffer_manager = BufferManager(
            meta_buffer=self.meta_buffer,
            llm=llm,
            similarity_threshold=similarity_threshold,
            distill_temperature=distill_temperature,
        )

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def distil_problem(self, question: str, temperature: float = 0.0) -> str:
        prompt = f"{PROBLEM_DISTILLER_SYSTEM}\n\nQuestion:\n{question}"
        return self.call_llm(prompt, temperature=temperature).content.strip()

    # ── Stage 2 ───────────────────────────────────────────────────────────────

    def retrieve_template(self, distilled_info: str) -> Optional[ThoughtTemplate]:
        # Query on the concise distilled-task paragraph, not the whole verbose
        # distiller dump — the latter dilutes embedding similarity and suppresses
        # legitimate template hits.
        query = extract_distilled_task(distilled_info)
        return self.meta_buffer.retrieve(query, threshold=self.similarity_threshold)

    # ── Stage 3 ───────────────────────────────────────────────────────────────

    def instantiate_with_template(
        self, distilled_info: str, template: ThoughtTemplate, system_prompt: Optional[str] = None
    ) -> str:
        role_line = f"[Role]\n{system_prompt}\n\n" if system_prompt else ""
        solution_desc_line = (
            f"\nSolution Description:\n{template.solution_description}\n"
            if template.solution_description else ""
        )
        code_directive = (
            CODE_EXECUTION_DIRECTIVE
            if self.execute_code and template.category in CODE_CATEGORIES
            else ""
        )
        prompt = (
            f"{INSTANTIATION_SYSTEM}{code_directive}\n\n"
            f"{role_line}"
            f"[Distilled Problem]\n{distilled_info}\n\n"
            f"[Thought Template — {template.category}]\n"
            f"Task Description:\n{template.description}\n"
            f"{solution_desc_line}"
            f"\nThought Template:\n{template.template}"
        )
        return self.call_llm(prompt, temperature=self.instantiation_temperature).content.strip()

    def instantiate_new_task(self, distilled_info: str, system_prompt: Optional[str] = None) -> str:
        role_line = f"[Role]\n{system_prompt}\n\n" if system_prompt else ""
        code_directive = NEW_TASK_CODE_DIRECTIVE if self.execute_code else ""
        prompt = f"{NEW_TASK_SYSTEM}{code_directive}\n\n{role_line}[Distilled Problem]\n{distilled_info}"
        return self.call_llm(prompt, temperature=self.instantiation_temperature).content.strip()

    # ── Answer extraction ─────────────────────────────────────────────────────

    @staticmethod
    def extract_answer(raw_response: str) -> str:
        m = re.compile(r"###\s*Answer\s*\n(.*?)(?=###|\Z)", re.DOTALL | re.IGNORECASE).search(raw_response)
        if m:
            return m.group(1).strip()
        lines = [ln.strip() for ln in raw_response.split("\n") if ln.strip()]
        return lines[-1] if lines else raw_response.strip()

    # ── Code execution + inspector repair loop ──────────────────────────────────

    def execute_solution(
        self, raw_solution: str, system_prompt: Optional[str] = None
    ) -> Optional[CodeExecutionResult]:
        """Extract and run the program in ``raw_solution``, repairing on failure.

        Returns None when the solution contains no fenced code (so the caller
        falls back to textual answer extraction). On a failed run the inspector
        prompt feeds the code + error back to the LLM, up to ``max_code_repairs``
        times (paper §4.1). All LLM calls go through ``call_llm`` for accounting.
        """
        code = extract_code(raw_solution)
        if code is None:
            return None

        result = run_code(code, timeout=self.code_timeout)
        attempts = 0
        while not result.success and attempts < self.max_code_repairs:
            attempts += 1
            role_line = f"[Role]\n{system_prompt}\n\n" if system_prompt else ""
            prompt = (
                f"{INSPECTOR_SYSTEM}\n\n"
                f"{role_line}"
                f"[Code]\n```python\n{result.code}\n```\n\n"
                f"[Execution Error]\n{result.error}"
            )
            fixed = self.call_llm(prompt, temperature=self.instantiation_temperature).content
            new_code = extract_code(fixed) or fixed.strip()
            result = run_code(new_code, timeout=self.code_timeout)
        return result

    # ── run() ─────────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> BaselineResponse:
        self.reset_counters()
        self.buffer_manager.reset_counters()
        intermediate_steps: List[str] = []

        # Combine instruction with the raw question so the distiller has full context.
        full_question = f"{instruction}\n\n{question}" if instruction else question
        distilled_info = self.distil_problem(full_question, temperature=temperature)
        intermediate_steps.append(f"[Stage 1: Problem Distiller]\n{distilled_info}")

        template = self.retrieve_template(distilled_info)
        is_new_task = template is None
        if template is not None:
            intermediate_steps.append(
                f"[Stage 2: Template Retrieved]\nCategory: {template.category}\n"
                f"Description: {template.description}\nIndex: {template.index}"
            )
        else:
            intermediate_steps.append(
                "[Stage 2: Template Retrieval] No suitable template found — new task."
            )

        raw_solution = (
            self.instantiate_with_template(distilled_info, template, system_prompt=system_prompt)
            if template is not None
            else self.instantiate_new_task(distilled_info, system_prompt=system_prompt)
        )
        intermediate_steps.append(f"[Stage 3: Instantiated Reasoning]\n{raw_solution}")
        final_answer = self.extract_answer(raw_solution)

        # ── Stage 3b: optional code execution (paper §4.1) ──────────────────
        code_attempted = False
        code_success: Optional[bool] = None
        if self.execute_code:
            exec_result = self.execute_solution(raw_solution, system_prompt=system_prompt)
            if exec_result is not None:
                code_attempted = True
                code_success = exec_result.success
                if exec_result.success:
                    final_answer = exec_result.output
                intermediate_steps.append(
                    f"[Stage 3b: Code Execution] success={exec_result.success}\n"
                    f"output={exec_result.output!r}\nerror={exec_result.error!r}"
                )

        new_template: Optional[ThoughtTemplate] = None
        if self.update_buffer:
            new_template = self.buffer_manager.distil_and_update(
                distilled_info=distilled_info, solution=raw_solution
            )
            self.total_input_tokens += self.buffer_manager.total_input_tokens
            self.total_output_tokens += self.buffer_manager.total_output_tokens
            self.num_llm_calls += self.buffer_manager.num_calls
            intermediate_steps.append(
                f"[Stage 4: Buffer Manager] New template added "
                f"(index={new_template.index}, category='{new_template.category}')."
                if new_template is not None
                else "[Stage 4: Buffer Manager] Template not added "
                     "(too similar to an existing one or distillation failed)."
            )

        if template:
            sol_desc = (
                f"\nSolution Description:\n{template.solution_description}\n"
                if template.solution_description else ""
            )
            template_block = (
                f"Category: {template.category}\n"
                f"Task Description:\n{template.description}\n"
                f"{sol_desc}"
                f"\nThought Template:\n{template.template}"
            )
        else:
            template_block = "(none — new task, used general reasoning structure)"
        reasoning_trace = (
            f"[Distilled Problem]\n{distilled_info}\n\n"
            f"[Retrieved Template]\n{template_block}\n\n"
            f"[Instantiated Reasoning]\n{raw_solution}"
        )

        return self.create_response(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            intermediate_steps=intermediate_steps,
            metadata={
                "is_new_task": is_new_task,
                "retrieved_template_index": template.index if template else None,
                "retrieved_template_category": template.category if template else None,
                "new_template_added": new_template is not None,
                "meta_buffer_size": self.meta_buffer.size,
                "similarity_threshold": self.similarity_threshold,
                "instantiation_temperature": self.instantiation_temperature,
                "distill_temperature": self.distill_temperature,
                "update_buffer": self.update_buffer,
                "execute_code": self.execute_code,
                "code_executed": code_attempted,
                "code_execution_success": code_success,
            },
        )

    def __repr__(self) -> str:
        return (
            f"BoT(baseline_name='{self.baseline_name}', "
            f"llm={self.llm.__class__.__name__}, "
            f"similarity_threshold={self.similarity_threshold}, "
            f"meta_buffer_size={self.meta_buffer.size}, "
            f"update_buffer={self.update_buffer}, "
            f"execute_code={self.execute_code})"
        )