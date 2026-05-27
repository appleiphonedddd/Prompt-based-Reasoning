import argparse
import json
import logging
import os
import time
from datetime import datetime
from models.gpt import GPTClient
from models.deepseek import DeepSeekClient
from models.llama import LlamaClient
from models.qwen import QwenClient
from models.gemma import GemmaClient
from models.granite import GraniteClient
from baseline.CoT import ZeroShotCoT, ZeroShotCoTSinglePass
from baseline.RoT import RoT, SentenceTransformerEmbedding
from baseline.ToT import ToT
from baseline.BoT import BoT
from baseline.GoT import GoT
from baseline.Standard import Input
from utils.metrics import Efficiency, Accuracy
from utils.get_mean_std import AccuracyStatistics
from benchmark import DATASET_REGISTRY
from benchmark.BigBenchHard.bigbenchhard import BigBenchHardTask


logging.getLogger().setLevel(logging.ERROR)

# ── Registries ────────────────────────────────────────────────────────────────
# To add a new model: insert one entry here.
MODEL_REGISTRY: dict[str, type] = {
    "gpt":      GPTClient,
    "deepseek": DeepSeekClient,
    "llama":    LlamaClient,
    "llama3.1":    LlamaClient,
    "llama3.3":    LlamaClient,
    "llama2":    LlamaClient,
    "qwen":     QwenClient,
    "qwen2":    QwenClient,
    "qwen2.5":  QwenClient,
    "qwen3":    QwenClient,
    "gemma":    GemmaClient,
    "gemma3":   GemmaClient,
    "granite4.1": GraniteClient
}

# To add a new baseline: insert one entry here (class, kwargs-extractor).
BASELINE_REGISTRY: dict[str, tuple] = {
    "standard":       (Input,                  lambda _: {}),
    "zerocot":        (ZeroShotCoT,            lambda _: {}),
    "zerocot_single": (ZeroShotCoTSinglePass,  lambda _: {}),
    "rot": (RoT, lambda a: dict(
        warmup=a.warmup,
        candidate_temperature=a.candidate_temperature,
        instantiation_temperature=a.instantiation_temperature,
        embedding_model=SentenceTransformerEmbedding(a.rot_embedding_model),
        similarity_threshold=a.rot_similarity_threshold,
    )),
    "tot": (ToT, lambda a: dict(
        n_generate_sample=a.tot_n_generate,
        n_evaluate_sample=a.tot_n_evaluate,
        breadth_limit=a.tot_breadth,
        max_steps=a.tot_max_steps,
        search_algorithm=a.tot_algorithm,
        value_threshold=a.tot_value_threshold,
        propose_temperature=a.tot_propose_temperature,
        value_temperature=a.tot_value_temperature,
    )),
    "bot": (BoT, lambda a: dict(
        similarity_threshold=a.bot_threshold,
        buffer_path=a.buffer_path,
        distill_temperature=a.bot_distill_temp,
        instantiation_temperature=a.bot_instantiate_temp,
        update_buffer=a.update_buffer,
        execute_code=a.bot_execute_code,
        max_code_repairs=a.bot_max_repairs,
        code_timeout=a.bot_code_timeout,
    )),
    "got": (GoT, lambda a: dict(
        num_branches=a.got_branches,
        keep_best=a.got_keep,
        refine_rounds=a.got_refine,
        gen_temperature=a.got_gen_temp,
        score_temperature=a.got_score_temp,
        agg_temperature=a.got_agg_temp,
    )),
}


# ── Evaluator ─────────────────────────────────────────────────────────────────
class Evaluator:
    """Orchestrates a single benchmark evaluation session."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_family = args.model.split(":")[0].lower()
        self.validate()

    def validate(self) -> None:
        if self.model_family not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{self.model_family}' not supported. "
                f"Supported providers: {list(MODEL_REGISTRY)}"
            )
        if self.args.baseline.lower() not in BASELINE_REGISTRY:
            raise ValueError(
                f"Baseline '{self.args.baseline}' not supported. "
                f"Supported: {list(BASELINE_REGISTRY)}"
            )
        if self.args.benchmark.lower() not in DATASET_REGISTRY:
            raise ValueError(
                f"Benchmark '{self.args.benchmark}' not supported. "
                f"Supported: {list(DATASET_REGISTRY)}"
            )

    def build_client(self):
        return MODEL_REGISTRY[self.model_family](model=self.args.model)

    def build_baseline(self, client, dataset=None):
        cls, extract_kwargs = BASELINE_REGISTRY[self.args.baseline.lower()]
        kwargs = extract_kwargs(self.args)
        if self.args.baseline.lower() == "rot" and dataset is not None:
            parts = [p for p in [dataset.get_system_prompt(), dataset.get_instruction()] if p]
            kwargs["task_prompt"] = "\n".join(parts)
            # Input-output demonstrations D for the reverse-reasoning warm-up.
            kwargs["demos"] = dataset.get_demonstrations(n_shot=self.args.rot_n_shot)
        return cls(llm=client, **kwargs)

    def build_dataset(self):
        benchmark_key = self.args.benchmark.lower()
        cls, extract_kwargs = DATASET_REGISTRY[benchmark_key]
        dataset = cls(**extract_kwargs(self.args))
        dataset.load_dataset()
        return dataset

    def run_once(self, run_index: int, dataset, efficiency: Efficiency) -> float:
        print(f"\n[Run {run_index}/{self.args.num_runs}]")
        client   = self.build_client()
        baseline = self.build_baseline(client, dataset)
        accuracy = Accuracy()

        n = len(dataset)

        task_times: list[float] = []
        for i in range(n):
            problem = dataset.get_problem(i)
            t0 = time.perf_counter()
            try:
                response = baseline.run(
                    problem.question,
                    system_prompt=dataset.get_system_prompt(),
                    instruction=dataset.get_instruction(),
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                task_times.append(elapsed)
                accuracy.record(False)
                print(f"  [{i + 1}/{n}] ✗  ERROR ({elapsed:.1f}s): {exc!r}")
                continue
            elapsed = time.perf_counter() - t0
            task_times.append(elapsed)

            result = dataset.evaluate_answer(response.final_answer, problem.ground_truth)
            accuracy.record(result.is_correct)
            mark = "✓" if result.is_correct else "✗"
            print(f"  [{i + 1}/{n}] {mark}  ({elapsed:.1f}s)  answer={response.final_answer!r}")

        efficiency.record_sample(task_times)
        acc = accuracy.get_accuracy()
        print(f"  Accuracy: {acc:.2f}%")
        return acc

    def _get_languages_to_test(self) -> list[str]:
        """Determine which languages to test.

        Priority:
        1. --languages arg (if provided)
        2. --language arg (if single language specified)
        3. 'en' (default)
        """
        args = self.args

        # If --languages is provided, use it
        if hasattr(args, 'languages') and args.languages:
            return args.languages

        # If --language is 'all', test all 10 languages
        if args.language.lower() == 'all':
            return ["en", "de", "fr", "es", "ru", "zh", "ja", "th", "sw", "bn"]

        # Otherwise use single language
        return [args.language]

    def run(self) -> None:
        args = self.args

        # Only apply multilingual logic if the benchmark is MGSM
        if args.benchmark.lower() != "mgsm":
            # For non-MGSM benchmarks, use 'en' as default and ignore --language arg
            self._run_single_language("en")
            return

        # For MGSM: determine languages to test
        languages = self._get_languages_to_test()

        # If only one language, run normally
        if len(languages) == 1:
            self._run_single_language(languages[0])
        else:
            # Multiple languages: run each and aggregate
            self._run_multiple_languages(languages)

    def _save_results(self, data: dict) -> None:
        os.makedirs("results", exist_ok=True)
        model_slug = self.args.model.replace(":", "-").replace("/", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{self.args.benchmark}_{self.args.baseline}_{model_slug}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results saved → {filename}")

    def _run_single_language(self, language: str) -> None:
        """Run evaluation for a single language."""
        args = self.args
        print(f"\n--- Starting Evaluation ---")
        print(f"Model:          {args.model}")
        print(f"Benchmark:      {args.benchmark}")
        print(f"Baseline:       {args.baseline}")
        print(f"Language:       {language}")
        print(f"Number of Runs: {args.num_runs}")

        # Temporarily set the language for this evaluation
        original_language = args.language
        args.language = language

        dataset = self.build_dataset()
        n = len(dataset)
        print(f"Questions:      {n}")
        efficiency = Efficiency(num_tasks=n)

        stats = AccuracyStatistics()
        for i in range(1, args.num_runs + 1):
            stats.add_result(self.run_once(i, dataset, efficiency))

        stats.print_summary(baseline_name=args.baseline)
        avg_t = efficiency.get_T()
        print(f"Avg time/question: {avg_t:.2f}s  (over {efficiency.get_M()} run(s))")
        print("\nAll done!")

        self._save_results({
            "metadata": {
                "model": args.model,
                "benchmark": args.benchmark,
                "baseline": args.baseline,
                "language": language,
                "num_runs": args.num_runs,
                "num_questions": n,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            "per_run_accuracies": stats.accuracy_list,
            "mean_accuracy": round(sum(stats.accuracy_list) / len(stats.accuracy_list), 4) if stats.accuracy_list else 0.0,
            "avg_time_per_question_s": round(avg_t, 4),
        })

        # Restore original language
        args.language = original_language

    def _run_multiple_languages(self, languages: list[str]) -> None:
        """Run evaluation for multiple languages and aggregate results."""
        args = self.args
        print(f"\n--- Starting Multilingual MGSM Evaluation ---")
        print(f"Model:          {args.model}")
        print(f"Benchmark:      {args.benchmark}")
        print(f"Baseline:       {args.baseline}")
        print(f"Languages:      {', '.join(languages)}")
        print(f"Number of Runs: {args.num_runs}")
        print("=" * 70)

        results = {}  # language -> accuracy

        # Run evaluation for each language
        for lang_idx, language in enumerate(languages, 1):
            print(f"\n[{lang_idx}/{len(languages)}] Testing {language.upper()}...")
            print("-" * 70)

            # Temporarily set the language
            original_language = args.language
            args.language = language

            try:
                dataset = self.build_dataset()
                n = len(dataset)
                efficiency = Efficiency(num_tasks=n)

                stats = AccuracyStatistics()
                for run_idx in range(1, args.num_runs + 1):
                    accuracy = self.run_once(run_idx, dataset, efficiency)
                    stats.add_result(accuracy)

                results[language] = {
                    'accuracy': stats.accuracy_list[-1] if stats.accuracy_list else 0.0,
                    'all_runs': stats.accuracy_list
                }

            except Exception as e:
                print(f"✗ Error testing {language}: {e}")
                results[language] = {'accuracy': 0.0, 'error': str(e)}
            finally:
                args.language = original_language

        # Print aggregated results
        print("\n" + "=" * 70)
        print("MULTILINGUAL RESULTS SUMMARY")
        print("=" * 70)

        valid_results = {lang: res for lang, res in results.items() if 'error' not in res}

        for language in languages:
            if language in valid_results:
                acc = valid_results[language]['accuracy']
                print(f"  ✓ {language.upper():3s}: {acc:6.2f}%")
            else:
                error = results[language].get('error', 'Unknown error')
                print(f"  ✗ {language.upper():3s}: FAILED ({error})")

        if valid_results:
            accuracies = [res['accuracy'] for res in valid_results.values()]
            mean_acc = sum(accuracies) / len(accuracies)

            if len(accuracies) > 1:
                variance = sum((x - mean_acc) ** 2 for x in accuracies) / (len(accuracies) - 1)
                std_acc = variance ** 0.5
            else:
                std_acc = 0.0

            print("-" * 70)
            print(f"Benchmark:      {args.benchmark}")
            print(f"Baseline:       {args.baseline}")
            print(f"  Mean Accuracy:  {mean_acc:.2f}%  (std: {std_acc:.2f}%)")
            print(f"  Min/Max:        {min(accuracies):.2f}% / {max(accuracies):.2f}%")
            print(f"  Success Rate:   {len(valid_results)}/{len(languages)} languages")
            print("-" * 70)

            self._save_results({
                "metadata": {
                    "model": args.model,
                    "benchmark": args.benchmark,
                    "baseline": args.baseline,
                    "languages": languages,
                    "num_runs": args.num_runs,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
                "per_language": results,
                "summary": {
                    "mean_accuracy": round(mean_acc, 4),
                    "std_accuracy": round(std_acc, 4),
                    "min_accuracy": round(min(accuracies), 4),
                    "max_accuracy": round(max(accuracies), 4),
                    "success_rate": f"{len(valid_results)}/{len(languages)}",
                },
            })

        print("\nAll done!")


# ── CLI argument groups ───────────────────────────────────────────────────────
# To add a news baseline: add one _add_<name>_args function and call it below.

def general_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model",        default="qwen2.5:14b",
                        help="Model name (prefix = provider, e.g. 'gemini:...')")
    parser.add_argument("--benchmark",    default="gameof24",
                        help="Benchmark / dataset name (gameof24, mgsm, sonnetwriting, bigbenchhard, programmingpuzzles, humaneval, mbpp, apps, classeval, cruxeval)")
    parser.add_argument("--baseline",     default="zerocot",
                        help="Baseline: standard | zerocot | zerocot_single | rot | tot | bot | got")
    parser.add_argument("--num_runs",     type=int, default=1,
                        help="Independent experiment runs")
    parser.add_argument("--language",     default="all",
                        help="Language for MGSM benchmark (en, de, fr, es, ru, zh, ja, th, sw, bn, or 'all' for all languages)")
    parser.add_argument("--languages",    nargs="+", default=None,
                        help="MGSM: test multiple languages (e.g. --languages en zh ja), overrides --language")
    parser.add_argument("--bigbenchhard_task", default="boolean_expressions",
                        choices=[t.value for t in BigBenchHardTask],
                        help="BigBenchHard task (only used when --benchmark=bigbenchhard)")
    parser.add_argument("--split", default="test",
                        help="Dataset split (default: test for BigBenchHard)")

def rot_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("RoT")
    g.add_argument("--warmup", type=int, default=5,
                   help="Number of reverse-reasoning candidates K")
    g.add_argument("--rot_n_shot", type=int, default=2,
                   help="Number of input-output demonstrations D for the "
                        "reverse-reasoning warm-up (paper uses 1 or 2)")
    g.add_argument("--candidate_temperature", type=float, default=0.7,
                   help="Sampling temperature for candidate generation")
    g.add_argument("--instantiation_temperature", type=float, default=0.1,
                   help="Sampling temperature for instantiation reasoning")
    g.add_argument("--rot_embedding_model", default="BAAI/bge-large-en-v1.5",
                   help="SentenceTransformer model name for CPM similarity")
    g.add_argument("--rot_similarity_threshold", type=float, default=0.7,
                   help="CPM similarity threshold δ (paper recommends 0.6–0.8)")


def tot_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("ToT")
    g.add_argument("--tot_algorithm", default="bfs", choices=["bfs", "dfs"],
                   help="Search algorithm: breadth-first (bfs) or depth-first (dfs)")
    g.add_argument("--tot_n_generate", type=int, default=5,
                   help="Candidate thoughts to propose at each node (k in paper)")
    g.add_argument("--tot_n_evaluate", type=int, default=3,
                   help="Value-prompt samples per thought for majority-vote scoring")
    g.add_argument("--tot_breadth", type=int, default=5,
                   help="[BFS] Frontier width b — states kept per BFS level")
    g.add_argument("--tot_max_steps", type=int, default=3,
                   help="Maximum tree depth T (3 for Game of 24)")
    g.add_argument("--tot_value_threshold", type=float, default=1.0,
                   help="[DFS] Prune states whose value score falls below this threshold")
    g.add_argument("--tot_propose_temperature", type=float, default=0.7,
                   help="Sampling temperature for thought generation (higher → more diverse)")
    g.add_argument("--tot_value_temperature", type=float, default=0.0,
                   help="Sampling temperature for state evaluation (0 = deterministic)")


def bot_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("BoT")
    g.add_argument("--bot_threshold", type=float, default=0.5,
                   help="Similarity threshold (δ) for template retrieval and novelty "
                        "updates (paper recommends 0.5–0.7). Lower = retrieve seeds "
                        "more readily on the first question; higher = grow the buffer faster.")
    g.add_argument("--buffer_path", default=None,
                   help="Path to a JSON file for persisting thought-templates on disk. "
                        "Default: None (in-memory only) — the meta-buffer is seeded fresh "
                        "from the paper's templates at the start of every run, so runs stay "
                        "independent and reproducible and no state leaks across models / "
                        "benchmarks / repeats. Pass an explicit path ONLY if you want the "
                        "buffer to accumulate and persist across runs.")
    g.add_argument("--bot_distill_temp", type=float, default=0.2,
                   help="Temperature for problem distillation and template extraction")
    g.add_argument("--bot_instantiate_temp", type=float, default=0.1,
                   help="Temperature for final reasoning instantiation")
    g.add_argument("--no_update_buffer", action="store_false", dest="update_buffer",
                   help="Disable automatic buffer updating after solving")
    g.add_argument("--bot_execute_code", action="store_true", dest="bot_execute_code",
                   help="Execute the instantiated Python program and use its stdout as the "
                        "answer (paper §4.1; drives the Game-of-24 / programming / chess "
                        "results). WARNING: runs untrusted model-generated code in a "
                        "timeout-bounded subprocess — enable only on a trusted host. Default: off.")
    g.add_argument("--bot_max_repairs", type=int, default=3,
                   help="Max inspector repair attempts when executed code errors (default: 3)")
    g.add_argument("--bot_code_timeout", type=float, default=10.0,
                   help="Per-execution timeout in seconds for generated code (default: 10)")


def got_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("GoT")
    g.add_argument("--got_branches", type=int, default=3,
                   help="Number of branches to explore at each step")
    g.add_argument("--got_keep", type=int, default=2,
                   help="Number of top-scored branches to keep for aggregation "
                        "(must be >= 2 for the Aggregate step to run)")
    g.add_argument("--got_refine", type=int, default=2,
                   help="Number of refinement rounds to perform")
    g.add_argument("--got_gen_temp", type=float, default=0.7,
                   help="Sampling temperature for branch generation")
    g.add_argument("--got_score_temp", type=float, default=0.0,
                   help="Sampling temperature for branch scoring (0 = deterministic)")
    g.add_argument("--got_agg_temp", type=float, default=0.0,
                   help="Sampling temperature for final answer aggregation (0 = deterministic)")


def pp_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("ProgrammingPuzzles")
    g.add_argument("--pp_num_samples", type=int, default=None,
                   help="ProgrammingPuzzles: number of puzzles to evaluate (default: all 1715)")
    g.add_argument("--pp_module", default=None,
                   help="ProgrammingPuzzles: filter by module category "
                        "(e.g. study.py, basic.py, IMO.py; default: all modules)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prompt-Based Reasoning Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    general_args(parser)
    rot_args(parser)
    tot_args(parser)
    bot_args(parser)
    got_args(parser)
    pp_args(parser)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    for key, val in vars(args).items():
        print(f"{key} = {val}")

    Evaluator(args).run()
