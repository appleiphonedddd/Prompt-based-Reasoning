"""
Benchmark Module.

Provides dataset wrappers for evaluating LLMs on reasoning tasks.

Available datasets:
    - GameOf24:            Arithmetic puzzle — combine 4 numbers to reach 24.
    - MGSM:                Multilingual Grade School Math word problems.
    - SonnetWriting:       Shakespearean sonnet generation with constraints.
    - BigBenchHard:        All 27 BIG-Bench Hard reasoning tasks.
    - ProgrammingPuzzles:  Python programming puzzles with sat-function verification.
    - HumanEval:           164 Python function completion tasks (OpenAI HumanEval).
    - MBPP:                974 Python function generation tasks (Google MBPP).
    - APPS:                5000 competitive-programming problems (stdin/stdout + fn_name).
    - ClassEval:           100 class-level Python implementation tasks.
    - CRUXEval:            799 code output-prediction tasks (CRUXEval-O).
    - Checkmate:           3500 checkmate-in-one chess positions (BIG-bench).

Usage::

    from benchmark import GameOf24, MGSM, SonnetWriting, BigBenchHard, ProgrammingPuzzles

    ds = GameOf24()
    ds.load_dataset()
    problem = ds.get_problem(0)

    # BigBenchHard example
    bbh = BigBenchHard(task="boolean_expressions")
    bbh.load_dataset()

    # ProgrammingPuzzles example
    pp = ProgrammingPuzzles(num_samples=50, module="study.py")
    pp.load_dataset()

    # HumanEval example
    he = HumanEval()
    he.load_dataset()
    problem = he.get_problem(0)

    # MBPP example
    mbpp = MBPP()
    mbpp.load_dataset()
    problem = mbpp.get_problem(0)

    # APPS example
    apps = APPS()
    apps.load_dataset()
    problem = apps.get_problem(0)

    # ClassEval example
    ce = ClassEval()
    ce.load_dataset()
    problem = ce.get_problem(0)

    # CRUXEval example
    crux = CRUXEval()
    crux.load_dataset()
    problem = crux.get_problem(0)

    # Checkmate example
    cm = Checkmate(num_samples=100)
    cm.load_dataset()
    problem = cm.get_problem(0)

Author: Egor Morozov
"""

from benchmark.datasetbase import DatasetBase, Problem, EvaluationResult
from benchmark.GameOf24.gameof24 import GameOf24
from benchmark.MGSM.mgsm import MGSM
from benchmark.SonnetWriting.sonnetwriting import SonnetWriting
from benchmark.BigBenchHard.bigbenchhard import BigBenchHard
from benchmark.ProgrammingPuzzles.programpuzzles import ProgrammingPuzzles
from benchmark.HumanEval.humaneval import HumanEval
from benchmark.MBPP.mbpp import MBPP
from benchmark.APPS.apps import APPS
from benchmark.ClassEval.classeval import ClassEval
from benchmark.CRUXEval.cruxeval import CRUXEval
from benchmark.Checkmate.checkmate import Checkmate

# Registry used by main.py for dynamic instantiation.
# Format: { key: (DatasetClass, kwargs_extractor) }
# kwargs_extractor receives the parsed argparse.Namespace and returns a dict
# of keyword arguments to pass to the dataset constructor.
# To add a new dataset: insert one entry here — no changes to main.py needed.
DATASET_REGISTRY: dict[str, tuple] = {
    "gameof24":            (GameOf24,            lambda _: {}),
    "mgsm":                (MGSM,                lambda a: dict(language=a.language)),
    "sonnetwriting":       (SonnetWriting,        lambda _: {}),
    "bigbenchhard":        (BigBenchHard,         lambda a: dict(
        task=a.bigbenchhard_task,
        split=a.split if hasattr(a, "split") else "test",
    )),
    "programmingpuzzles":  (ProgrammingPuzzles,   lambda a: dict(
        num_samples=a.pp_num_samples,
        module=a.pp_module if hasattr(a, "pp_module") else None,
    )),
    "humaneval":           (HumanEval,            lambda _: {}),
    "mbpp":                (MBPP,                 lambda _: {}),
    "apps":                (APPS,                 lambda _: {}),
    "classeval":           (ClassEval,            lambda _: {}),
    "cruxeval":            (CRUXEval,             lambda _: {}),
    "checkmate":           (Checkmate,            lambda a: dict(
        num_samples=getattr(a, "checkmate_num_samples", None),
    )),
}

__all__ = [
    "DatasetBase",
    "Problem",
    "EvaluationResult",
    "GameOf24",
    "MGSM",
    "SonnetWriting",
    "BigBenchHard",
    "ProgrammingPuzzles",
    "HumanEval",
    "MBPP",
    "APPS",
    "ClassEval",
    "CRUXEval",
    "Checkmate",
    "DATASET_REGISTRY",
]
