import unittest

from ternrefine import (
    APGPoint,
    apply_relocations,
    apg_first_non_improvement,
    changed_coordinates,
    enumerate_cpsr_moves,
    qgp_score,
)


class CoreTest(unittest.TestCase):
    def test_cpsr_preserves_side_cardinality(self):
        state = [1, 0, -1, 0]
        moves = enumerate_cpsr_moves(
            layer=0,
            module="q",
            row=0,
            group=0,
            ternary_group=state,
            receiver_signs=[1, -1, 1, -1],
            score_fn=lambda donor, receiver, donor_sign, receiver_sign: 0.0,
        )
        self.assertEqual(len(moves), 4)
        after = apply_relocations(state, [moves[0]])
        self.assertEqual(sum(abs(x) for x in state), sum(abs(x) for x in after))
        self.assertEqual(changed_coordinates(state, after), 2)

    def test_apg_first_non_improvement(self):
        curve = [
            APGPoint(1, 3.0, 1, 2),
            APGPoint(2, 2.8, 2, 4),
            APGPoint(4, 2.85, 4, 8),
        ]
        self.assertEqual(apg_first_non_improvement(curve).patch_size, 2)

    def test_qgp_score_sign(self):
        score = qgp_score(
            donor_gradient=1.0,
            receiver_gradient=-1.0,
            mu=0.0,
            alpha=0.5,
            donor_sign=1,
            receiver_sign=1,
        )
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
