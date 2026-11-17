"""Behavioural checks for the numerical core."""

import math
import unittest

from synapse.network import DenseNetwork


XOR = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


class DenseNetworkTests(unittest.TestCase):
    def test_seed_produces_reproducible_predictions(self):
        first = DenseNetwork(2, 6, seed=7)
        second = DenseNetwork(2, 6, seed=7)
        self.assertEqual(first.predict([0, 1]), second.predict([0, 1]))

    def test_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            DenseNetwork(0, 6)
        with self.assertRaises(ValueError):
            DenseNetwork(2, -1)

    def test_rejects_invalid_inputs(self):
        network = DenseNetwork(2, 6)
        with self.assertRaises(ValueError):
            network.predict([1])
        with self.assertRaises(ValueError):
            network.predict([math.nan, 1])
        with self.assertRaises(ValueError):
            network.loss([])

    def test_training_reduces_xor_loss(self):
        network = DenseNetwork(2, 6, seed=7)
        before = network.loss(XOR)
        for _ in range(6000):
            network.train_step(XOR)
        self.assertLess(network.loss(XOR), before * 0.2)
        for inputs, target in XOR:
            prediction = network.predict(inputs)
            self.assertLess(abs(prediction - target), 0.2)


if __name__ == "__main__":
    unittest.main()
