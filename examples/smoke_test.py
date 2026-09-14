#!/usr/bin/env python3
"""CPU-only miniature TernRefine example."""

from __future__ import annotations

from ternrefine import APGPoint, apg_first_non_improvement, apply_relocations, changed_coordinates, enumerate_cpsr_moves, qgp_score


def main() -> None:
    state = [-1, 0, 0, 1, -1, 1]
    gradient = [0.9, -0.2, 0.1, -0.8, 0.6, -0.3]
    mu = 0.0
    alpha = 0.5

    moves = enumerate_cpsr_moves(
        layer=0,
        module="q_proj",
        row=0,
        group=0,
        ternary_group=state,
        score_fn=lambda donor, receiver, donor_sign, receiver_sign: qgp_score(
            gradient[donor], gradient[receiver], mu, alpha, donor_sign, receiver_sign
        ),
    )
    scored = sorted(((move.score, move) for move in moves), key=lambda item: item[0], reverse=True)
    patch = []
    touched = set()
    for _, move in scored:
        if move.donor in touched or move.receiver in touched:
            continue
        patch.append(move)
        touched.update([move.donor, move.receiver])
        if len(patch) == 2:
            break
    refined = apply_relocations(state, patch)

    assert sorted(state) == sorted(refined)
    assert len(patch) == 2
    assert changed_coordinates(state, refined) == 4

    selected = apg_first_non_improvement(
        [
            APGPoint(1, 1.00, 1, 2),
            APGPoint(2, 0.92, 2, 4),
            APGPoint(4, 0.90, 4, 8),
            APGPoint(8, 0.91, 8, 16),
        ]
    )
    assert selected.patch_size == 4
    assert selected.validation_loss == 0.90

    print("smoke test passed")
    print(f"candidate_moves={len(moves)} selected_relocations={len(patch)} apg_k={selected.patch_size}")


if __name__ == "__main__":
    main()
