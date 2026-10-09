import unittest
import numpy as np

from gate_zo_bdnn.shd_gatezo_support import balanced_sample, smoke_training_indices
from gate_zo_bdnn.shd_radlif_residual_proposal_core import proposal_partitions
from gate_zo_bdnn.train_shd_individual_pool_structural_zo import internal_positions


class GateZOSmokeSamplingTests(unittest.TestCase):
    def test_original_four_samples_reproduce_empty_fit_and_clear_error(self):
        labels = np.tile(np.arange(4), 2)
        _, fit, cal = proposal_partitions(labels, .5, .2, 57)
        self.assertEqual(len(fit), 0)
        self.assertEqual(balanced_sample(fit, labels, 2, np.random.RandomState(57)), [])
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            internal_positions(labels, fit, cal, 4, np.random.RandomState(57))

    def test_smoke_has_disjoint_nonempty_partitions_with_or_without_jitter(self):
        labels = np.repeat(np.arange(20), 8)
        indices = np.arange(len(labels), dtype=np.int64)
        chosen = smoke_training_indices(labels, indices, 4)
        self.assertEqual(len(chosen), 80)
        np.testing.assert_array_equal(np.bincount(labels[chosen]), np.full(20, 4))
        for copies in (1, 2):
            selected_labels = np.tile(labels[chosen], copies)
            proposal, fit, cal = proposal_partitions(selected_labels, .5, .2, 57)
            for pool in (proposal, fit, cal):
                self.assertTrue(pool)
                self.assertEqual(len(np.unique(selected_labels[list(pool)])), 20)
            self.assertFalse(set(proposal) & set(fit))
            self.assertFalse(set(proposal) & set(cal))
            self.assertFalse(set(fit) & set(cal))
            sampled_fit, sampled_cal = internal_positions(selected_labels, fit, cal, 4, np.random.RandomState(57))
            self.assertEqual((len(sampled_fit), len(sampled_cal)), (2, 2))

    def test_insufficient_originals_and_invalid_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'four training originals'):
            smoke_training_indices([0, 0, 1, 1], [0, 1, 2, 3], 4)
        with self.assertRaisesRegex(ValueError, 'integer'):
            balanced_sample([0.5], [0, 1], 1, np.random.RandomState(57))


if __name__ == '__main__':
    unittest.main()
