# Synapse Lab

A small computational workspace for exploring how neural networks learn.

The first module implements a fully connected neural network in plain Python. It is intentionally small enough to read, test and extend without relying on a machine-learning framework. The aim is to make each step of the training process inspectable.

## Run an experiment

Python 3.10 or later is recommended. No third-party packages are required for the initial module.

```bash
python3 -m synapse.experiments.xor
```

The experiment prints the loss at regular intervals and the predictions for the four XOR inputs. Results are reproducible because the model uses a fixed random seed.

## Record an experiment

The experiment recorder captures the initial loss, losses at fixed intervals and the final loss, including when the last epoch is not on a recording boundary. The result includes the network configuration and predictions in dataset order.

```python
from synapse.experiments.history import ExperimentConfig, run_experiment
from synapse.experiments.xor import SAMPLES

config = ExperimentConfig(epochs=6000, record_every=100, seed=7)
result = run_experiment(SAMPLES, config)
result.save("results/xor")
```

This writes `experiment.json` (configuration and predictions) and `metrics.csv` (epoch and mean squared error). The files are suitable for subsequent analysis and plotting. Results are deterministic for a fixed Python version and configuration.

## Run tests

```bash
python3 -m unittest discover -s tests -v
```

## Scope

- Fully connected layers, forward propagation and backpropagation
- Training curves and model diagnostics
- Interactive architecture and weight visualisation
- Repeatable experiments and comparative analysis

The initial implementation is deliberately limited to a two-layer network trained with batch gradient descent. More elaborate experiments and a restrained scientific interface will be developed in subsequent increments.

## Development principles

The underlying calculations must be verifiable. Experiments should be reproducible, code should remain approachable, and visualisations should explain observed behaviour rather than merely decorate the interface.

## Licence

No licence has been selected yet. External code will only be incorporated once its licence and attribution requirements have been checked.
