"""Finite-difference checks for the backpropagation implementation."""

import unittest

from synapse.network import DenseNetwork


SAMPLES = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


class GradientChecks(unittest.TestCase):
    def test_backpropagation_matches_finite_differences(self):
        network = DenseNetwork(2, 3, learning_rate=0.01, seed=19)
        epsilon = 1e-5

        # Record each scalar parameter and its location before the update.
        parameters = [
            (row, index)
            for row in network.input_weights
            for index in range(len(row))
        ]
        parameters += [
            (network.hidden_biases, index)
            for index in range(network.hidden_size)
        ]
        parameters += [
            (network.output_weights, index)
            for index in range(network.hidden_size)
        ]

        numerical = []
        for values, index in parameters:
            original = values[index]
            values[index] = original + epsilon
            loss_plus = network.loss(SAMPLES)
            values[index] = original - epsilon
            loss_minus = network.loss(SAMPLES)
            values[index] = original
            numerical.append((loss_plus - loss_minus) / (2 * epsilon))

        original_output_bias = network.output_bias
        network.output_bias = original_output_bias + epsilon
        loss_plus = network.loss(SAMPLES)
        network.output_bias = original_output_bias - epsilon
        loss_minus = network.loss(SAMPLES)
        network.output_bias = original_output_bias
        numerical.append((loss_plus - loss_minus) / (2 * epsilon))

        before = [values[index] for values, index in parameters]
        network.train_step(SAMPLES)
        after = [values[index] for values, index in parameters]

        measured = [
            (old - new) / network.learning_rate
            for old, new in zip(before, after)
        ]
        measured.append(
            (original_output_bias - network.output_bias)
            / network.learning_rate
        )

        for position, (expected, actual) in enumerate(
            zip(numerical, measured)
        ):
            with self.subTest(parameter=position):
                self.assertAlmostEqual(expected, actual, delta=1e-7)


if __name__ == "__main__":
    unittest.main()
