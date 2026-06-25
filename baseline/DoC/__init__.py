"""
Dynamic Operator Composition (DOC) Prompting Baseline.

DOC synthesizes a per-instance reasoning program over a small algebra of typed
operators {Decompose, Branch, Aggregate, Abstract, Ground, Repair}, driven by a
verifier that types the residual gap rather than choosing a topology up front.
It adds two mechanisms absent from the other baselines: external execution as a
first-class operator (Ground) and a confidence gate for model robustness.

Reference:
- Egor (2026). "DOC: Dynamic Operator Composition for Verification-Guided
  Reasoning."
"""

from .doc import DoC

__all__ = ["DoC"]
