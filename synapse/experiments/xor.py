"""Train a small network on the XOR truth table."""

from synapse.network import DenseNetwork

SAMPLES = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


def main() -> None:
    network = DenseNetwork(input_size=2, hidden_size=6, seed=7)
    print("Synapse Lab | XOR classification")
    print(f"Initial mean squared error: {network.loss(SAMPLES):.6f}")

    for epoch in range(1, 6001):
        current_loss = network.train_step(SAMPLES)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:4d} | MSE {current_loss:.6f}")

    print("\nInput       Target   Prediction")
    for inputs, target in SAMPLES:
        print(f"{str([int(v) for v in inputs]):<11} {target:.0f}        {network.predict(inputs):.4f}")


if __name__ == "__main__":
    main()
