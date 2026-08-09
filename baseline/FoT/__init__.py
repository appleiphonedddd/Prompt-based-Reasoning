"""
Falsification-of-Thought (FoT) Prompting Baseline.

FoT makes self-refutation the engine of reasoning: given a candidate answer, the
model constructs a falsifying *witness* and, when one is found, repairs the
candidate conditioned on that witness, iterating until a candidate survives
falsification (a fixpoint) or a budget is exhausted.

The witness is a constructive artifact, never a verdict. Where a cheap decisive
checker c_q exists, the model only proposes a probe and the checker decides
(executable regime). Where it does not, the query is transformed by a fixed,
human-audited catalogue of semantics-preserving relations, each variant is solved
independently, and the witness is a concrete disagreement inside the resulting
orbit of answers (metamorphic regime). In both regimes the model only constructs
artifacts; a deterministic comparator issues every verdict.

Reference:
- "Falsification-of-Thought: Reasoning by Metamorphic Self-Refutation".
"""

from .fot import Damage, FoT, Witness
from .relations import Relation, Slot, Variant, enumerate_slots, get_catalogue

__all__ = ["FoT", "Witness", "Damage", "Relation", "Slot", "Variant",
           "enumerate_slots", "get_catalogue"]
