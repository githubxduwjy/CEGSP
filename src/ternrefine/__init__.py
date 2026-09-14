"""Core utilities for TernRefine.

The executable representative examples live in ``examples``. This package keeps
the paper-level CPSR/QGP/APG primitives in a small importable form.
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
