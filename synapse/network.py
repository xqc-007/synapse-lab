"""A compact, inspectable neural network for binary classification."""

from __future__ import annotations

import math
import random
from typing import Sequence


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


class DenseNetwork:
    """A single-hidden-layer network trained with batch gradient descent."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        learning_rate: float = 0.8,
        seed: int = 7,
    ) -> None:
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("Layer sizes must be positive.")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("The learning rate must be finite and positive.")

        generator = random.Random(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.input_weights = [
            [generator.uniform(-1.0, 1.0) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.hidden_biases = [0.0] * hidden_size
        self.output_weights = [
            generator.uniform(-1.0, 1.0) for _ in range(hidden_size)
        ]
        self.output_bias = 0.0

    def _forward(self, inputs: Sequence[float]) -> tuple[list[float], float]:
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs.")
        if not all(math.isfinite(value) for value in inputs):
            raise ValueError("Inputs must be finite.")

        hidden = [
            math.tanh(
                sum(weight * value for weight, value in zip(row, inputs))
                + bias
            )
            for row, bias in zip(self.input_weights, self.hidden_biases)
        ]
        output = _sigmoid(
            sum(weight * value for weight, value in zip(self.output_weights, hidden))
            + self.output_bias
        )
        return hidden, output

    def predict(self, inputs: Sequence[float]) -> float:
        """Return the estimated probability of the positive class."""
        return self._forward(inputs)[1]

    def loss(self, samples: Sequence[tuple[Sequence[float], float]]) -> float:
        """Mean squared error across a non-empty dataset."""
        if not samples:
            raise ValueError("The dataset cannot be empty.")
        total = 0.0
        for inputs, target in samples:
            if not math.isfinite(target) or not 0 <= target <= 1:
                raise ValueError("Targets must be finite and between 0 and 1.")
            total += (self.predict(inputs) - target) ** 2
        return total / len(samples)

    def train_step(
        self, samples: Sequence[tuple[Sequence[float], float]]
    ) -> float:
        """Apply one full-batch gradient update and return the new loss."""
        if not samples:
            raise ValueError("The dataset cannot be empty.")

        gradient_input = [
            [0.0] * self.input_size for _ in range(self.hidden_size)
        ]
        gradient_hidden = [0.0] * self.hidden_size
        gradient_output = [0.0] * self.hidden_size
        gradient_bias = 0.0
        count = len(samples)

        for inputs, target in samples:
            if not math.isfinite(target) or not 0 <= target <= 1:
                raise ValueError("Targets must be finite and between 0 and 1.")

            hidden, output = self._forward(inputs)
            output_delta = (
                2.0 * (output - target) * output * (1.0 - output) / count
            )
            gradient_bias += output_delta

            for neuron in range(self.hidden_size):
                gradient_output[neuron] += output_delta * hidden[neuron]
                hidden_delta = (
                    output_delta
                    * self.output_weights[neuron]
                    * (1.0 - hidden[neuron] ** 2)
                )
                gradient_hidden[neuron] += hidden_delta
                for feature in range(self.input_size):
                    gradient_input[neuron][feature] += (
                        hidden_delta * inputs[feature]
                    )

        for neuron in range(self.hidden_size):
            for feature in range(self.input_size):
                self.input_weights[neuron][feature] -= (
                    self.learning_rate * gradient_input[neuron][feature]
                )
            self.hidden_biases[neuron] -= (
                self.learning_rate * gradient_hidden[neuron]
            )
            self.output_weights[neuron] -= (
                self.learning_rate * gradient_output[neuron]
            )

        self.output_bias -= self.learning_rate * gradient_bias
        return self.loss(samples)
