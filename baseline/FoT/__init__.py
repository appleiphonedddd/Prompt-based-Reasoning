"""
Falsification-of-Thought (FoT) Prompting Baseline.

FoT makes self-refutation the engine of reasoning: given a candidate answer, the
model constructs a falsifying *witness* and, when one is found, repairs the
candidate conditioned on that witness, iterating until a candidate survives
falsification (a fixpoint) or a budget is exhausted.

The witness is a constructive artifact, never a verdict. Where a cheap decisive
checker c_q exists, the model only proposes a probe and the checker decides
(executable regime). Where it does not, the candidate is attacked with a fixed,
human-audited library of relation schemas — backward substitution, metamorphic
transformations and mechanical invariants — whose single model-produced value is
a majority over independent completions, and the witness is the mechanically
exhibited contradiction (relational regime). In both regimes the model only
constructs artifacts; a deterministic comparator issues every verdict.

Reference:
- "Falsification-of-Thought: Reasoning by Self-Refutation with Relational
  Falsification".
"""

from .fot import FoT, Witness
from .relations import Relation, Slot, Variant, enumerate_slots, get_catalogue

__all__ = ["FoT", "Witness", "Relation", "Slot", "Variant", "enumerate_slots",
           "get_catalogue"]
