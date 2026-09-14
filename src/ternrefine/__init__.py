"""Core utilities for TernRefine.

Representative experiment runners live in ``examples``. This package keeps the
paper-level CPSR/QGP/APG primitives plus the reusable affine/PT2 adapters used
by those examples.
"""

from .core import (
    APGPoint,
    Relocation,
    apply_relocations,
    apg_first_non_improvement,
    changed_coordinates,
    enumerate_cpsr_moves,
    qgp_score,
)

__all__ = [
    "APGPoint",
    "Relocation",
    "apply_relocations",
    "apg_first_non_improvement",
    "changed_coordinates",
    "enumerate_cpsr_moves",
    "qgp_score",
]
