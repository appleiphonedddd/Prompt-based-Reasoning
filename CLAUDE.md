# Claude.md: Prompt-based Reasoning Project Guide

## Project Overview

**Prompt-based Reasoning** is a comprehensive framework for evaluating and benchmarking Large Language Models (LLMs) on complex reasoning tasks using various prompting strategies. The project enables systematic comparison of:

- **Multiple LLM Providers**: OpenAI (GPT), DeepSeek, Meta (Llama), Alibaba (Qwen), Google (Gemma), IBM (Granite)
- **Prompting Baselines**: Standard input, Zero-Shot CoT (two-step & single-pass), Reversal-of-Thought (RoT), Tree-of-Thought (ToT), Buffer-of-Thought (BoT), Graph-of-Thought (GoT), and Falsification-of-Thought (FoT)
- **Reasoning Benchmarks**: Game of 24, MGSM, Sonnet Writing, BigBenchHard (27 tasks), Programming Puzzles, HumanEval, MBPP, APPS, ClassEval, CRUXEval

The framework provides standardized evaluation metrics (accuracy, efficiency) and supports both local models (via Ollama) and cloud-based APIs.

---

## Technology Stack & Versions

### Core Dependencies
- **Python**: 3.11.14 (via Conda)
- **PyTorch**: 2.11.0 (with CUDA 13.0 support for GPU acceleration)
- **NumPy**: 2.4.1
- **Pandas**: 3.0.0
- **CUDA Toolkit**: 13.0.2 (for GPU support)

### LLM & API Integration
- **OpenAI SDK**: 2.15.0 (for GPT models)
- **Requests/HTTPX**: 2.32.5 / 0.28.1 (HTTP client libraries)
- **YAML Configuration**: PyYAML 6.0.3

### Data & ML Libraries
- **Pydantic**: 2.12.5 (data validation and settings management)
- **Datasets**: 4.5.0 (HuggingFace benchmark datasets)
- **Huggingface Hub**: 1.3.4
- **PyArrow**: 23.0.0 (columnar data format)
- **scikit-learn**: similarity computations (used in BoT for template retrieval)
- **scipy**: statistical analysis
- **sentence-transformers**: semantic embedding for BoT similarity threshold

### Environment & Deployment
- **Conda**: Environment management (virtual environment in `/home/infor/miniconda3/envs/Prompt`)
- **Ollama**: Local LLM inference engine for GPU-accelerated model serving
- **NVIDIA Container Toolkit**: For GPU support (if using Docker containers)

### Development Tools
- **Typer / Click**: CLI framework (argparse used directly in `main.py`)
- **Logging**: Python's built-in logging module

---

## Directory Structure & Architecture

```
Prompt-based-Reasoning/
├── main.py                    # Entry point with Model/Baseline/Dataset registries
├── config.yaml               # LLM API endpoints & model configurations
├── env.yaml                  # Conda environment specification
├── setup_ollama_gpu.sh       # GPU/Docker environment setup script
│
├── models/                   # LLM Provider Implementations
│   ├── base.py              # Abstract BaseLLM class & LLMResponse dataclass
│   ├── gpt.py               # OpenAI GPT implementation
│   ├── deepseek.py          # DeepSeek LLM implementation
│   ├── llama.py             # Meta Llama (via Ollama) implementation
│   ├── qwen.py              # Alibaba Qwen (via Ollama) implementation
│   ├── gemma.py             # Google Gemma (via Ollama) implementation
│   └── granite.py           # IBM Granite (via Ollama) implementation
│
├── baseline/                # Prompting Strategy Implementations
│   ├── basebaseline.py      # Abstract BaseBaseline class & BaselineResponse
│   ├── Standard/
│   │   └── io.py            # Standard input-output baseline
│   ├── CoT/
│   │   └── zero_shot_cot.py  # ZeroShotCoT (2-step) & ZeroShotCoTSinglePass
│   ├── RoT/
│   │   └── rot.py            # Reversal-of-Thought (with caching & parallelism)
│   ├── ToT/
│   │   └── tot.py            # Tree-of-Thought (BFS/DFS)
│   ├── BoT/
│   │   └── bot.py            # Buffer-of-Thought (+ shared run_code executor)
│   ├── GoT/
│   │   └── got.py            # Graph-of-Thought
│   └── FoT/
│       ├── fot.py            # Falsification-of-Thought (Solve→Falsify→Repair)
│       ├── relations.py      # Relation library R_q: bwd / mr / inv (audit this file)
│       └── checkers.py       # Trusted per-benchmark checkers c_q
│
├── eval/                    # Per-baseline evaluation sweep scripts (bash)
│   ├── eval_standard.sh     # One script per baseline; each cd's to repo root
│   ├── eval_cot.sh          # and runs main.py across the benchmark suite
│   ├── eval_rot.sh
│   ├── eval_tot.sh
│   ├── eval_bot.sh
│   ├── eval_got.sh
│   └── eval_fot.sh
│
├── results/                 # Auto-created JSON run results (gitignored)
│
├── benchmark/               # Reasoning Task Datasets
│   ├── __init__.py          # DATASET_REGISTRY (10 benchmarks)
│   ├── datasetbase.py       # Abstract DatasetBase, Problem, EvaluationResult
│   ├── GameOf24/            # Arithmetic puzzle: combine 4 numbers to reach 24
│   ├── MGSM/                # Multilingual Grade School Math (10 languages)
│   ├── SonnetWriting/       # Shakespearean sonnet generation
│   ├── BigBenchHard/        # All 27 BIG-Bench Hard tasks
│   ├── ProgrammingPuzzles/  # Python programming puzzles (sat-function verification)
│   ├── HumanEval/           # 164 Python function completion tasks (OpenAI)
│   ├── MBPP/                # 974 Python function generation tasks (Google)
│   ├── APPS/                # 5000 competitive-programming problems
│   ├── ClassEval/           # 100 class-level Python implementation tasks
│   └── CRUXEval/            # 799 code output-prediction tasks (CRUXEval-O)
│
├── utils/                   # Utility Modules
│   ├── config.py            # Configuration loading & validation
│   ├── metrics.py           # Accuracy & Efficiency metric classes
│   └── get_mean_std.py      # Statistical analysis (mean, std dev)
│
└── tests/                   # Test Suite
```

### Key Architectural Patterns

1. **Registry Pattern** (`main.py`)
   - `MODEL_REGISTRY`: Maps model name prefixes to LLM client classes
   - `BASELINE_REGISTRY`: Maps baseline names to strategy classes + argument extractors
   - `DATASET_REGISTRY` (`benchmark/__init__.py`): Maps benchmark names to dataset classes

2. **Abstract Base Classes**
   - `BaseLLM` in `models/base.py`: Enforces `generate(prompt, temperature=0, logprobs=False)`
   - `BaseBaseline` in `baseline/basebaseline.py`: Enforces `run()` method interface
   - `DatasetBase` in `benchmark/datasetbase.py`: Enforces `load_dataset()`, `get_problem()`, `evaluate_answer()`; optional hooks `get_instruction()`, `get_system_prompt()`, `get_demonstrations()`

3. **Dataclass-based Response Objects**
   - `LLMResponse`: Standardized LLM output (content, model_name, input_tokens, output_tokens, `avg_logprob`, `raw_response`)
   - `BaselineResponse`: Unified benchmark evaluation result (final_answer, reasoning_trace)
   - `Problem`: Benchmark problem (index, question, ground_truth, metadata)
   - `EvaluationResult`: Evaluation outcome (is_correct, score, prediction, ground_truth, details)

4. **`get_demonstrations(n_shot)` hook** (`benchmark/datasetbase.py:167`)
   - Supplies the input-output examples `D` that RoT reverse-engineers the task definition from; only RoT consumes it (wired in `Evaluator.build_baseline`).
   - The default derives demos from the first `n_shot` problems via `_demo_output()`, which works when ground truth *is* the literal expected output (MGSM, BBH).
   - Datasets whose ground truth is not the expected output override it with hand-crafted demos: GameOf24, SonnetWriting, HumanEval, MBPP, APPS, ClassEval, ProgrammingPuzzles.

5. **Shared code executor** (`run_code` in `baseline/BoT/bot.py:130`)
   - Runs Python in an isolated subprocess (`python -I -c`) with a timeout; a run fails if it raises, exits non-zero, times out, or prints nothing.
   - Reused by BoT, RoT, and FoT's checkers rather than re-implemented — change it in one place.

---

## Baselines

| Key | Class | Description |
|-----|-------|-------------|
| `standard` | `Input` | Direct prompt → answer |
| `zerocot` | `ZeroShotCoT` | Two-step: reasoning + answer extraction |
| `zerocot_single` | `ZeroShotCoTSinglePass` | Single-pass: "Let's think step by step" inline |
| `rot` | `RoT` | Reversal-of-Thought: generate K reverse-reasoning candidates, cache Stage 1+2, parallelize LLM calls |
| `tot` | `ToT` | Tree-of-Thought: BFS or DFS over thought tree |
| `bot` | `BoT` | Buffer-of-Thought: meta-buffer with semantic template retrieval |
| `got` | `GoT` | Graph-of-Thought: branch-score-aggregate refinement loops |
| `fot` | `FoT` | Falsification-of-Thought: Solve → Falsify (executable checker or relation library) → Repair until a candidate survives |

### ZeroShotCoT vs ZeroShotCoTSinglePass
- **ZeroShotCoT** (`zerocot`): Two LLM calls — (1) elicit reasoning with "Let's think step by step", (2) extract final answer
- **ZeroShotCoTSinglePass** (`zerocot_single`): One LLM call — append the phrase and parse the answer directly from the response
- For generative tasks (code generation, sonnets), both detect `is_generative_task` and skip the second extraction pass

### BoT Buffer
- **In-memory by default** (`--buffer_path` defaults to `None`): the meta-buffer is seeded fresh from the paper's templates on every run, so runs stay independent and reproducible and no state leaks across models / benchmarks / repeats. Pass an explicit path *only* to accumulate templates across runs.
- Similarity matching via sentence-transformers embeddings + cosine similarity (δ = `--bot_threshold`, default `0.5`)
- `--no_update_buffer` disables automatic buffer updates after each solve

### FoT (Falsification-of-Thought) — self-refutation with relational falsification
Implements the paper's driver loop (Algorithm 4): `a ← Solve(q)`, then up to `K` (`--fot_budget`) rounds. Each round makes up to `m` (`--fot_survival`) falsification attempts, **breaking on the first witness**; if all `m` attempts come back ⊥ the candidate is accepted as a fixpoint, otherwise it is repaired on that witness. On budget exhaustion the **last** candidate is returned (Alg. 4 line 12) — `--fot_archive` opts into the older best-corroborated selection instead. Cost is `O(K·(1+m·s))` model calls, all short completions, far below ToT/GoT's recursive expansion.

The design invariant is **the model constructs; a deterministic procedure decides**. No prompt in FoT asks "is this correct?" — the model only ever produces objects (an answer, a probe, a re-derived quantity, a solution to a transformed problem) and every accept/refute verdict is issued by a checker or a comparator. Two regimes, selected by `HasChecker(q)` — **fixed per benchmark, never decided at run time by the model**:

- **Executable** — for benchmarks with a registered trusted checker `c_q` in `CHECKERS` (`baseline/FoT/checkers.py`): `gameof24` (arithmetic evaluation), `cruxeval` (program execution), `programmingpuzzles` (`sat()` predicate), `bigbenchhard:multistep_arithmetic_two` (expression recomputation). The model only *proposes a probe*; the external checker returns the verdict, so witnesses are **sound** and a correct candidate is never discarded. Checkers return `fail` / `pass` / `undecided`, and `undecided` counts as survival. Keys are `benchmark` or `benchmark:subtask`, resolved most-specific-first. `m` is forced to **1** here: every `c_q` is deterministic and probe-independent, so a second probe could only repeat the first verdict.
- **Relational** — every other benchmark. The candidate is attacked with a fixed, human-audited library `R_q` of relation schemas (`baseline/FoT/relations.py`) from the paper's three families (see below). `Next(R_q)` / `NextSlot(q)` cycle through **distinct** relations and masked slots, so surviving `m` attempts means surviving `m` different checks rather than `m` re-rolls of one. The witness is `⟨ρ, instance, (expected, observed)⟩` — a mechanically exhibited contradiction, never an opinion.

The relational regime's one model-produced value per check is a **voted derivation**: the majority over `s` (`--fot_votes`, default 5) independent completions of the same instance, abstaining on a tie or an `UNDETERMINED` majority. That is what converts "one bad sample can refute a correct answer" into "a false refutation needs a systematic majority error on a short, tightly constrained derivation". Since the votes must be independent, they are sampled at `--fot_vote_temp` (0.7) — at temperature 0 a deterministic backend would just return `s` copies of one completion.

This replaces the pilot's *semantic* regime (`pi_nec` + `pi_chk` + the `ReCheckable` guard), where the model asserted that the candidate violated a self-generated necessary condition — the source of the paper's MGSM regression. There is no judgement template anywhere in FoT: `φ` is evaluated by the driver.

`Evaluator.build_baseline` threads the benchmark in as `task` (plus `subtask` for BigBenchHard), which selects both the checker and the library. `--no_fot_execute_code` forces the relational regime everywhere.

#### The four prompt templates (§4)
`pi_solve` (Solve, and every follow-up solve — it has **no candidate slot**, which is what makes `ρ_mr`'s follow-up evidence rather than an echo) → `ANSWER:`; `pi_probe` (executable Falsify) → `PROBE:`; `pi_mask` (backward substitution) → `DERIVED:` with an `UNDETERMINED` escape; `pi_rep` (Repair) → `ANSWER:`. `pi_mr` is an implementation extra for the paraphrase relations whose `T` cannot be run in code. Solve/Repair receive the full task framing so their output stays parseable; the falsifier's own calls receive only the one-line task description, so the answer-format persona never conflicts with "propose a probe" / "re-derive X".

#### The relation library (`baseline/FoT/relations.py`)
`get_catalogue(task, subtask, names)` resolves `task:subtask` → `task` → default. **Order is load-bearing**: Falsify returns on the first violation, so each library is sorted by soundness rung — invariants, then backward substitution, then answer-preserving transforms, then covariant ones. Every entry must be *valid* and *non-decreasing in reliability*; distractor insertion is deliberately excluded. A relation that does not fit a given query returns `None` and is skipped — never misapplied — so an unregistered benchmark degrades to FoT ≡ Solve.

| Family | `Relation.family` | Realised by | Model calls per check |
|--------|-------------------|-------------|----------------------|
| Backward substitution `ρ_bwd` | `bwd` | `apply_slot(q, a, slot)` + `pi_mask` | `s` |
| Metamorphic transformation `ρ_mr` | `mr` | `apply(q, a)` (or `pi_mr`) + `pi_solve` | `s` (+1 if model-instantiated) |
| Mechanical invariant `ρ_inv` | `inv` | `invariant(q, a)` → `(expected, observed, detail)` | **0** |

| Task | Library (in draw order) |
|------|-------------------------|
| `mgsm` | `mask_quantity` (bwd, NextSlot over the numeric literals), `permute_premises`, `translate_to_english` (`pi_mr`, non-English only), `scale_quantities_x2/x3` (φ: `a' = c·a`) |
| `bigbenchhard:geometric_shapes` | `shape_vertex_count` (**inv** — vertex count and path closure, no model call), `svg_rotate`, `svg_translate_scale`, `svg_reverse`, `options_shift1` |
| `bigbenchhard` (and default) | `options_shift1`, `options_reverse`, `options_shift2` — relabelling options is answer-transforming with a known `g⁻¹` |
| code benchmarks | `rename_identifiers`, `insert_dead_code` (AST-based) |

Model-instantiated transforms (`apply is None`) are **mechanically validated** after generation: the variant must carry exactly the numeric literals of the original (`preserve_numbers`), or it is discarded rather than checked.

Two audit notes are recorded in the module docstring: `scale_quantities` is valid only when the answer is homogeneous of degree one in the scaled quantities — with τ gone its guards are now library ordering (it sits last), the voted derivation, and `pi_rep`'s explicit licence to defend the candidate against a pair-implicating witness — and `mask_quantity` is the one relation whose check instance intentionally contains the candidate.

#### Mechanism-level metadata
Each `BaselineResponse.metadata` carries `initial_answer`, `answer_changed`, `accepted_fixpoint` and a `witnesses` list (`regime`, `family`, `relation`, `expected`, `observed`, `votes`, `survived`, `sound`), which is what the paper's §3.6 metrics need — flip matrix, witness precision per family, false-fixpoint rate — once joined with ground truth. Note that `main.py` does **not** currently write per-question metadata into `results/`, so consuming these means logging the responses yourself.

### Code Execution in BoT / RoT / FoT
BoT and RoT **execute the model-generated program by default** and use its stdout as the answer (this is what drives the Game-of-24 / programming results); FoT's executable regime runs the *benchmark's own* reference code. All three go through `run_code`, which uses a timeout-bounded isolated subprocess — but this is still **untrusted model-generated code**. Disable on an untrusted host with `--no_bot_execute_code` / `--no_rot_execute_code` / `--no_fot_execute_code`.

When execution errors, BoT and RoT attempt inspector-guided repair up to `--{bot,rot}_max_repairs` times (default 3).

---

## Benchmarks

### Math & Reasoning
| Key | Class | Size | Description |
|-----|-------|------|-------------|
| `gameof24` | `GameOf24` | ~1362 | Arithmetic puzzle: combine 4 numbers to reach 24 using +−×÷ |
| `mgsm` | `MGSM` | 250×lang | Multilingual Grade School Math (10 languages: en, de, fr, es, ru, zh, ja, th, sw, bn) |
| `bigbenchhard` | `BigBenchHard` | 250/task | All 27 BIG-Bench Hard tasks (use `--bigbenchhard_task`) |
| `sonnetwriting` | `SonnetWriting` | 20 | Shakespearean sonnet generation with constraints |

### Programming
| Key | Class | Size | Description |
|-----|-------|------|-------------|
| `humaneval` | `HumanEval` | 164 | Python function completion from docstring (OpenAI HumanEval) |
| `mbpp` | `MBPP` | 974 | Python function generation from problem description (Google MBPP) |
| `apps` | `APPS` | 5000 | Competitive programming (stdin/stdout + fn_name formats) |
| `classeval` | `ClassEval` | 100 | Class-level Python implementation (methods + class structure) |
| `cruxeval` | `CRUXEval` | 799 | Code output prediction: given `f` and input, predict return value |
| `programmingpuzzles` | `ProgrammingPuzzles` | 1715 | Python puzzles with sat-function verification |

### APPS Evaluation Details
APPS has two problem formats:
- **stdin/stdout** (~99%): Model writes a program reading from `input()`/`sys.stdin`, mocked at eval time
- **fn_name** (~1%): LeetCode-style — model implements a named function or `Solution` class method

Answer extraction always takes the **last** fenced code block so CoT baselines (which may emit wrong code before the corrected solution) are handled correctly.

### CRUXEval Evaluation Details
Model must predict the exact Python literal returned by a given function `f(*args)`. Evaluation uses `ast.literal_eval` when possible, falling back to normalized string comparison.

### BigBenchHard Task Categories
- **Boolean/Yes-No** (6): `boolean_expressions`, `causal_judgement`, `formal_fallacies`, `navigate`, `sports_understanding`, `web_of_lies`
- **Multiple-Choice letter** (17): `date_understanding`, `disambiguation_qa`, `geometric_shapes`, `hyperbaton`, `logical_deduction_{3,5,7}_objects`, `movie_recommendation`, `penguins_in_a_table`, `reasoning_about_colored_objects`, `ruin_names`, `salient_translation_error_detection`, `snarks`, `temporal_sequences`, `tracking_shuffled_objects_{3,5,7}_objects`
- **Numeric** (2): `multistep_arithmetic_two`, `object_counting`
- **Free-form** (2): `dyck_languages`, `word_sorting`

---

## CLI Interface

```bash
python main.py --model <model> --baseline <baseline> --benchmark <benchmark> [options]
```

### General Options
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen2.5:32b` | Model name (prefix = provider, e.g. `gpt:gpt-4o`) |
| `--baseline` | `zerocot` | `standard \| zerocot \| zerocot_single \| rot \| tot \| bot \| got \| fot` |
| `--benchmark` | `gameof24` | See benchmark keys in table above |
| `--num_runs` | `1` | Independent experiment runs |
| `--language` | `all` | MGSM language (`en, de, fr, es, ru, zh, ja, th, sw, bn` or `all`) |
| `--languages` | None | MGSM: space-separated list, overrides `--language` |
| `--bigbenchhard_task` | `boolean_expressions` | BigBenchHard task name (validated against the `BigBenchHardTask` enum) |
| `--split` | `test` | Dataset split (used by BigBenchHard) |

### RoT Options
| Flag | Default | Description |
|------|---------|-------------|
| `--warmup` | `5` | Number of reverse-reasoning candidates K |
| `--rot_n_shot` | `2` | Input-output demonstrations D for the warm-up (paper uses 1–2) |
| `--candidate_temperature` | `0.7` | Temperature for candidate generation |
| `--instantiation_temperature` | `0.1` | Temperature for instantiation reasoning |
| `--rot_embedding_model` | `BAAI/bge-large-en-v1.5` | SentenceTransformer model for CPM similarity |
| `--rot_similarity_threshold` | `0.7` | CPM similarity threshold δ (paper: 0.6–0.8) |
| `--no_rot_execute_code` | — | Disable executing the instantiated program (on by default) |
| `--rot_max_repairs` | `3` | Max inspector repair attempts on execution error |
| `--rot_code_timeout` | `10.0` | Per-execution timeout (seconds) |

### ToT Options
| Flag | Default | Description |
|------|---------|-------------|
| `--tot_algorithm` | `bfs` | Search: `bfs` or `dfs` |
| `--tot_n_generate` | `5` | Candidate thoughts per node |
| `--tot_n_evaluate` | `3` | Value-prompt samples per thought |
| `--tot_breadth` | `5` | BFS frontier width |
| `--tot_max_steps` | `3` | Max tree depth |
| `--tot_value_threshold` | `1.0` | DFS pruning threshold |
| `--tot_propose_temperature` | `0.7` | Temperature for thought generation |
| `--tot_value_temperature` | `0.0` | Temperature for state evaluation |

### BoT Options
| Flag | Default | Description |
|------|---------|-------------|
| `--bot_threshold` | `0.5` | Similarity threshold δ for retrieval and novelty updates (paper: 0.5–0.7) |
| `--buffer_path` | `None` | JSON file for persisting templates; default = in-memory, reseeded each run |
| `--bot_distill_temp` | `0.2` | Temperature for problem distillation |
| `--bot_instantiate_temp` | `0.1` | Temperature for final reasoning instantiation |
| `--no_update_buffer` | — | Disable automatic buffer updates |
| `--no_bot_execute_code` | — | Disable executing the instantiated program (on by default) |
| `--bot_max_repairs` | `3` | Max inspector repair attempts on execution error |
| `--bot_code_timeout` | `10.0` | Per-execution timeout (seconds) |

### GoT Options
| Flag | Default | Description |
|------|---------|-------------|
| `--got_branches` | `3` | Number of branches to explore |
| `--got_keep` | `2` | Top-scored branches kept for aggregation (**must be ≥ 2** or Aggregate never runs) |
| `--got_refine` | `2` | Refinement rounds |
| `--got_gen_temp` | `0.7` | Temperature for branch generation |
| `--got_score_temp` | `0.0` | Temperature for branch scoring |
| `--got_agg_temp` | `0.0` | Temperature for final aggregation |

### FoT Options
| Flag | Default | Description |
|------|---------|-------------|
| `--fot_budget` | `3` | Budget K: max Falsify→Repair iterations before the last candidate is returned |
| `--fot_survival` | `3` | Survival threshold m: **distinct** checks a candidate must survive to be accepted as a fixpoint (executable regime always uses 1) |
| `--fot_votes` | `5` | Vote size s: independent completions a relational check takes the majority over; ties and `UNDETERMINED` majorities abstain |
| `--fot_relations` | None (all) | Restrict `R_q` to these relation names (e.g. `--fot_relations mask_quantity`) |
| `--fot_generate_relations` | — | Ablation: build `R_q` with `pi_mr-gen` from the run's first question instead of the audited library |
| `--fot_archive` | — | Deviation from Alg. 4: on budget exhaustion return the best-corroborated candidate instead of the last |
| `--fot_solve_temp` | `0.0` | Temperature for the initial Solve |
| `--fot_vote_temp` | `0.7` | Temperature for the s completions of a relational check (`pi_mask`, follow-up `pi_solve`) — must be > 0 |
| `--fot_falsify_temp` | `0.7` | Temperature for the falsifier's own construction calls (`pi_probe`, `pi_mr`) |
| `--fot_repair_temp` | `0.0` | Temperature for witness-guided Repair |
| `--no_fot_execute_code` | — | Force relational falsification on every benchmark |
| `--fot_code_timeout` | `10.0` | Per-execution timeout for the trusted checker c_q |
| `--no_fot_witness_history` | — | Stop carrying past witnesses into Repair (history is on by default) |

### ProgrammingPuzzles Options
| Flag | Default | Description |
|------|---------|-------------|
| `--pp_num_samples` | None (all) | Number of puzzles to evaluate |
| `--pp_module` | None (all) | Filter by module (e.g. `study.py`, `basic.py`, `IMO.py`) |

---

## Usage Examples

```bash
# Activate environment
conda activate Prompt

# Local model via Ollama
python main.py --model qwen2.5:32b --baseline zerocot --benchmark gameof24

# FoT (executable regime — gameof24 has a trusted checker)
python main.py --model qwen2.5:32b --baseline fot --benchmark gameof24 \
  --fot_budget 4

# FoT (relational regime — no checker for geometric_shapes, so R_q attacks it)
python main.py --model qwen2.5:32b --baseline fot --benchmark bigbenchhard \
  --bigbenchhard_task geometric_shapes --fot_survival 3 --fot_votes 5

# BoT with a persistent buffer (default is in-memory)
python main.py --model qwen2.5:32b --baseline bot --benchmark gameof24 \
  --buffer_path my_buffer.json --bot_threshold 0.7

# Run a full per-baseline sweep
./eval/eval_fot.sh

# MGSM: all 10 languages
python main.py --model gemma3:27b --baseline zerocot_single --benchmark mgsm

# MGSM: specific languages
python main.py --model gpt:gpt-4o --baseline zerocot --benchmark mgsm \
  --languages en zh ja

# BigBenchHard: specific task
python main.py --benchmark bigbenchhard --bigbenchhard_task boolean_expressions \
  --baseline zerocot --model qwen2.5:32b --num_runs 5

# Programming benchmarks
python main.py --model gpt:gpt-4o --baseline zerocot_single --benchmark humaneval
python main.py --model gpt:gpt-4o --baseline zerocot_single --benchmark mbpp
python main.py --model gpt:gpt-4o --baseline zerocot_single --benchmark apps
python main.py --model qwen2.5:32b --baseline zerocot_single --benchmark cruxeval

# ToT with DFS
python main.py --model qwen2.5:32b --baseline tot --benchmark gameof24 \
  --tot_algorithm dfs --tot_max_steps 4

# Setup GPU/Ollama (first time)
./setup_ollama_gpu.sh
docker exec -it ollama ollama run qwen2.5:32b
```

### Run Output
Every run writes a JSON file to `results/` (auto-created, gitignored):

```
results/{benchmark}_{baseline}_{model-slug}_{YYYYmmdd_HHMMSS}.json
```

Single-language runs record `per_run_accuracies`, `mean_accuracy`, and `avg_time_per_question_s`. Multi-language MGSM runs record `per_language` plus a `summary` (mean/std/min/max accuracy and success rate). A question that raises is caught, scored incorrect, and the run continues — so a `results` file can contain silent failures; check the `✗ ERROR` lines in stdout.

---

## Code Style & Conventions

### Python Style
- **Version**: Python 3.11+ (modern syntax with type hints)
- **Type Hints**: Full type annotations on all function signatures
- **Formatting**: 4-space indentation (PEP 8 compliant)
- **Docstrings**: Google-style with Args, Returns, and Example sections

### Registry Pattern Usage
- **New model**: Create `models/my_model.py` extending `BaseLLM`, register in `MODEL_REGISTRY` in `main.py`
- **New baseline**: Create `baseline/MyMethod/my_method.py` extending `BaseBaseline`, register in `BASELINE_REGISTRY` in `main.py`
- **New benchmark**: Create `benchmark/MyBench/mybench.py` extending `DatasetBase`, register in `DATASET_REGISTRY` in `benchmark/__init__.py` — no `main.py` changes needed

### Example: Adding a New Model

```python
# models/my_model.py
from models.base import BaseLLM, LLMResponse

class MyModelClient(BaseLLM):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)

    def generate(self, prompt: str, temperature: float = 0,
                 logprobs: bool = False) -> LLMResponse:
        # Set avg_logprob when logprobs=True and the backend supports it;
        # leave it None otherwise (RoT degrades gracefully).
        return LLMResponse(content="...", model_name=self.model,
                           input_tokens=0, output_tokens=0,
                           avg_logprob=None)

# main.py MODEL_REGISTRY — key is the --model prefix (everything before ':'):
"mymodel": MyModelClient,
```

### Example: Adding a New Baseline

```python
# baseline/MyMethod/my_method.py
from baseline.basebaseline import BaseBaseline, BaselineResponse

class MyMethodBaseline(BaseBaseline):
    def __init__(self, llm: BaseLLM, param1: int = 10):
        super().__init__(llm, baseline_name="MyMethod")
        self.param1 = param1

    def run(self, question: str, **kwargs) -> BaselineResponse:
        return self.create_response(final_answer="...", reasoning_trace="...")

# main.py BASELINE_REGISTRY:
"mymethod": (MyMethodBaseline, lambda a: dict(param1=a.my_param)),
```

### Example: Adding a New Benchmark

```python
# benchmark/MyBench/mybench.py
from benchmark.datasetbase import DatasetBase, Problem, EvaluationResult

class MyBenchmark(DatasetBase):
    def load_dataset(self) -> None:
        self._data = [...]  # list or dict

    def get_problem(self, index: int) -> Problem:
        self._ensure_loaded()
        return Problem(index=index, question="...", ground_truth=..., metadata={})

    def evaluate_answer(self, prediction: str, ground_truth: Any) -> EvaluationResult:
        is_correct = ...
        return EvaluationResult(is_correct=is_correct, score=float(is_correct),
                                prediction=prediction, ground_truth=ground_truth, details={})

    def get_instruction(self) -> Optional[str]:
        return "..."

    def get_system_prompt(self) -> Optional[str]:
        return "..."

# benchmark/__init__.py DATASET_REGISTRY:
"mybench": (MyBenchmark, lambda _: {}),
```

---

## Development Notes

### Configuration Management
- API keys in environment variables (never hardcoded), exported before running. Ollama-backed clients fall back to the literal key `"local"` when `API_KEY` is unset.
- `config.yaml`: centralized LLM endpoints and default model selections
  - `llm.local.base_url` is the OpenAI-compatible Ollama endpoint shared by the Qwen / Llama / Gemma / Granite clients. It is **not necessarily localhost** — it currently points at a remote inference host, so check it before assuming a local server.
  - `models:` now only defines defaults for `granite`, `qwen`, and `gemma` (`gpt`, `deepseek`, `llama`, `gemini` were dropped). In practice this map is **dead config**: every client signature already hardcodes a default (`QwenClient(model="qwen2:7b")`, `LlamaClient(model="llama3.1:8b")`, …), so the `model or config["models"][...]` fallback never fires. Don't expect editing `models:` to change which model runs — pass `--model` instead.
- `env.yaml`: frozen dependencies for reproducibility

### Local Development Setup

```bash
conda env create -f env.yaml
conda activate Prompt
./setup_ollama_gpu.sh          # first time only
export API_KEY="ollama"
docker exec -it ollama ollama run qwen2.5:32b
python main.py --model qwen2.5:32b --baseline standard --benchmark mgsm
```

### Tests
`tests/` holds `unittest` suites (pytest is **not** installed in the `Prompt` env). There is no `tests/__init__.py`, so `unittest discover` does not work — it either errors (`Start directory is not importable`) or silently collects 0 tests. Run modules explicitly from the repo root:

```bash
python -m unittest tests.test_metrics                 # single module
python -m unittest tests.test_metrics tests.test_bot  # several
```

**Current suite state (verified 2026-08-06): 559 tests, `FAILED (failures=11, errors=65, skipped=3)`.** The failures are pre-existing and concentrated in tests that have not been updated to match refactored code — do not assume you caused them:

- `test_got` (39) and `test_tot` (23) — stale relative to the GoT/ToT fixes
- `test_llm` (9) — all in the deepseek / llama / qwen clients, from two stale assumptions:
  - `setUp` hardcodes `local_llm_url = "http://localhost:11434/v1/"`, which must be kept in sync by hand with `llm.local.base_url` in `config.yaml` (currently a remote host). Better fixed by reading the value from `get_config()`.
  - `*_missing_key` expects `ValueError` when no key is set, but these clients now intentionally fall back to the key `"local"` (Ollama needs no key). Only `GPTClient` still raises.
- `test_bot` (4), `test_benchmark` (1)

`tests/test_fot.py` (57 tests, all passing) covers the FoT relation library, the voted derivation, both falsification regimes, and the driver.

### Logging
- Level set to `ERROR` in `main.py` line 26 by default
- Set to `logging.DEBUG` for verbose development output

### Performance Considerations
1. **GPU Memory**: Monitor with `nvidia-smi` for large models
2. **API Rate Limiting**: Space out requests to cloud APIs
3. **RoT**: Stage 1+2 warmup is cached across questions; LLM calls within a stage are parallelized
4. **BoT**: `--no_update_buffer` skips buffer writes to speed up pure evaluation runs
5. **FoT**: `O(K·(1+m·s))` calls — lower `--fot_budget` / `--fot_survival` / `--fot_votes` to cut cost. The executable regime costs `O(K·2)` (one probe settles the verdict); a relational check's `s` completions run in parallel, and `rho_inv` checks cost nothing at all
6. **APPS**: Execution timeout applies per test case to prevent infinite loops
7. **Code execution**: `--{bot,rot}_code_timeout` bounds each generated-program run; repair attempts multiply LLM calls

### Token Tracking & Metrics
- `LLMResponse`: tracks `input_tokens` / `output_tokens` per call, plus optional `avg_logprob` (average token probability, RoT paper Eq. 2) when `generate(..., logprobs=True)` — populated by the Ollama-backed clients, `None` where unsupported
- `BaselineResponse`: aggregates totals across all LLM calls
- `Efficiency` metric: average time per question over M runs
- `Accuracy` metric: correct/incorrect evaluation per benchmark

### Git Workflow
- Main branch: `main`
- Commit message convention: `type(scope): description` (e.g. `feat(bot): ...`, `fix(rot): ...`)

---

## Quick Reference

### Files to Modify for Common Tasks
| Task | File(s) |
|------|---------|
| Add LLM provider | `models/new_provider.py`, `main.py` MODEL_REGISTRY |
| Add prompting method | `baseline/NewMethod/`, `main.py` BASELINE_REGISTRY |
| Add benchmark | `benchmark/NewBench/`, `benchmark/__init__.py` |
| Add CLI flag | `main.py` argument group functions |
| Update metrics | `utils/metrics.py`, `utils/get_mean_std.py` |
| Add an FoT checker | `baseline/FoT/checkers.py` `CHECKERS` dict (keyed by `benchmark` or `benchmark:subtask`) |
| Add an FoT relation | `baseline/FoT/relations.py` `CATALOGUES` dict (set `family` to `bwd` / `mr` / `inv`) |
| Add an eval sweep | `eval/eval_<baseline>.sh` |

### Supported Models (MODEL_REGISTRY prefixes)
`gpt`, `deepseek`, `llama` / `llama3.1` / `llama3.3` / `llama2`, `qwen` / `qwen2` / `qwen2.5` / `qwen3`, `gemma` / `gemma3`, `granite4.1`

The prefix is everything before the first `:` in `--model`, lowercased. All Llama variants map to `LlamaClient`, all Qwen variants to `QwenClient`, both Gemma keys to `GemmaClient`. Granite is registered **only** as `granite4.1` — `--model granite:...` raises "not supported"; use `--model granite4.1:30b`.

**Gemini has been removed**: `models/gemini.py` no longer exists and there is no registry entry, though a stale `llm.gemini.base_url` remains in `config.yaml`. Any `--model gemini:...` invocation will fail validation.
