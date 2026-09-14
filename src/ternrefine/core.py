"""Minimal CPSR/QGP/APG primitives.

The functions here are intentionally tensor-light and independent of a specific
model architecture. The representative examples adapt these primitives to
OPT/Llama affine ternary states and PT2 sidecars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence


@dataclass(frozen=True)
class Relocation:
    """A donor-receiver CPSR move inside one row/group."""

    layer: int
    module: str
    row: int
    group: int
    donor: int
    receiver: int
    donor_sign: int
    receiver_sign: int
    score: float = 0.0


@dataclass(frozen=True)
class APGPoint:
    """One validation point on an adaptive patch-growth curve."""

    patch_size: int
    validation_loss: float
    num_relocations: int
    changed_coordinates: int


def qgp_score(
    donor_gradient: float,
    receiver_gradient: float,
    mu: float,
    alpha: float,
    donor_sign: int,
    receiver_sign: int,
) -> float:
    """Return the first-order QGP score ``-<G, Delta Q>`` for one move."""

    donor_before = mu + alpha * donor_sign
    receiver_before = mu
    donor_after = mu
    receiver_after = mu + alpha * receiver_sign
    delta_donor = donor_after - donor_before
    delta_receiver = receiver_after - receiver_before
    return -float(donor_gradient * delta_donor + receiver_gradient * delta_receiver)


def enumerate_cpsr_moves(
    *,
    layer: int,
    module: str,
    row: int,
    group: int,
    ternary_group: Sequence[int],
    score_fn: Callable[[int, int, int, int], float],
) -> List[Relocation]:
    """Enumerate legal same-group donor-receiver relocations.

    Donors are side-state entries, receivers are center-state entries. A move
    transfers the donor side sign to a center entry and turns the donor into a
    center entry, preserving side-state cardinality exactly.
    """

    donors = [(i, int(v)) for i, v in enumerate(ternary_group) if int(v) in (-1, 1)]
    receivers = [i for i, v in enumerate(ternary_group) if int(v) == 0]
    moves: List[Relocation] = []
    for donor, donor_sign in donors:
        for receiver in receivers:
            for receiver_sign in (-1, 1):
                moves.append(
                    Relocation(
                        layer=layer,
                        module=module,
                        row=row,
                        group=group,
                        donor=donor,
                        receiver=receiver,
                        donor_sign=donor_sign,
                        receiver_sign=receiver_sign,
                        score=float(score_fn(donor, receiver, donor_sign, receiver_sign)),
                    )
                )
    return moves


def apply_relocations(state: List[int], moves: Iterable[Relocation]) -> List[int]:
    """Apply compatible relocations to a one-dimensional ternary state copy."""

    out = [int(x) for x in state]
    touched: set[int] = set()
    for move in moves:
        if move.donor in touched or move.receiver in touched:
            raise ValueError("incompatible CPSR moves touch the same coordinate")
        if out[move.donor] != move.donor_sign:
            raise ValueError("donor state does not match the registered move")
        if out[move.receiver] != 0:
            raise ValueError("receiver is not a center-state coordinate")
        out[move.donor] = 0
        out[move.receiver] = move.receiver_sign
        touched.add(move.donor)
        touched.add(move.receiver)
    return out


def changed_coordinates(before: Sequence[int], after: Sequence[int]) -> int:
    """Count coordinates whose ternary code changed."""

    if len(before) != len(after):
        raise ValueError("states must have equal length")
    return sum(int(a) != int(b) for a, b in zip(before, after))


def apg_first_non_improvement(curve: Sequence[APGPoint]) -> APGPoint:
    """Select APG's prefix using the first strict validation non-improvement.

    If every tested prefix strictly improves over the preceding point, the last
    tested prefix is returned. This implements the paper's practical bounded
    growth rule.
    """

    if not curve:
        raise ValueError("APG curve is empty")
    best = curve[0]
    previous = curve[0].validation_loss
    for point in curve[1:]:
        if not point.validation_loss < previous:
            return best
        best = point
        previous = point.validation_loss
    return best
