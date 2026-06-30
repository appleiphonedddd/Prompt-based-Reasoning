"""
Falsification-of-Thought (FoT) Prompting Baseline.

FoT makes self-refutation the engine of reasoning: given a candidate answer, the
model constructs a falsifying *witness* (a counterexample, a failing input, or a
violated necessary condition) and, when one is found, repairs the candidate
conditioned on that witness, iterating until a candidate survives falsification
(a fixpoint) or a budget is exhausted. The guiding principle is to falsify, not
critique. It unifies executable falsification (a cheap checker decides) and
semantic falsification (a model-derived necessary condition) under one operator.

Reference:
- "Falsification-of-Thought: Reasoning by Self-Refutation" (2026).
"""

from .fot import FoT

__all__ = ["FoT"]
