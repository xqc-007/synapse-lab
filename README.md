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

## Plot training loss

Generate a standalone SVG chart from recorded experiment metrics. The plot uses
a linear scale, labelled axes and no third-party packages.

```python
from synapse.experiments.history import ExperimentConfig, run_experiment
from synapse.experiments.xor import SAMPLES
from synapse.visualisation.loss_plot import save_loss_svg

result = run_experiment(SAMPLES, ExperimentConfig(epochs=6000, record_every=100))
save_loss_svg(result.metrics, "results/xor/loss.svg")
```

Open `results/xor/loss.svg` in a browser or image viewer. The chart is
generated from actual recorded losses, not illustrative data.


## Compare training runs

Plot multiple experiments on the same axes. This is useful for checking how a
change in the hidden-layer width affects convergence. Each curve uses its
recorded epoch positions, even when sampling intervals differ.

```python
from synapse.experiments.history import ExperimentConfig, run_experiment
from synapse.experiments.xor import SAMPLES
from synapse.visualisation.comparison_plot import save_comparison_svg

small = run_experiment(SAMPLES, ExperimentConfig(hidden_size=3, epochs=2000))
large = run_experiment(SAMPLES, ExperimentConfig(hidden_size=8, epochs=2000))
save_comparison_svg(
    {"3 hidden units": small.metrics, "8 hidden units": large.metrics},
    "results/xor/comparison.svg",
)
```

The output is a standalone SVG with a shared linear loss scale, a legend and
labelled axes. The comparison is descriptive; a single seed is not enough to
establish that one architecture consistently outperforms another.

## Compare random initialisations

A single training run can give a misleading impression of model performance.
Run the same configuration with several random seeds and report the spread in
final mean squared error:

```python
from synapse.experiments.seed_comparison import compare_seeds
from synapse.experiments.xor import SAMPLES

summary = compare_seeds(
    SAMPLES, hidden_size=6, seeds=[3, 7, 11, 19, 23], epochs=2000
)
print(f"Mean final MSE: {summary.mean_loss:.6f}")
print(f"Standard deviation: {summary.standard_deviation:.6f}")
print(f"Range: {summary.minimum_loss:.6f}–{summary.maximum_loss:.6f}")
```

The standard deviation describes the observed runs; it is not a confidence
interval or proof that one architecture is better than another. Compare
architectures using the same seed set and training budget.

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
