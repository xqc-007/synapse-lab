# Synapse Lab

A small computational workspace for exploring how neural networks learn.

The first module implements a fully connected neural network in plain Python. It is intentionally small enough to read, test and extend without relying on a machine-learning framework. The aim is to make each step of the training process inspectable.

## Run an experiment

Python 3.10 or later is recommended. No third-party packages are required for the initial module.

```bash
python3 -m synapse.experiments.xor
```

The experiment prints the loss at regular intervals and the predictions for the four XOR inputs. Results are reproducible because the model uses a fixed random seed.

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
