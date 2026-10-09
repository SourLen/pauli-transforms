# Pauli transforms

Code and data for reproducing the results in my (Lennart Sauer) bachelor's thesis
*Efficient Coordinate Transformations for Permutation-Invariant Quantum Systems*.

## Setup

Python 3.10 or newer. Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[benchmarks]'
```

## Reproduce the figures

```bash
python -m pauli_transforms.plot --data data/thesis --output figures
```

This redraws the eight thesis data figures as PDF, SVG and PNG from the saved
measurements. Add `--supplementary` for the additional plots.

## Run new measurements

```bash
python -m pauli_transforms.benchmark --comparison anschuetz --output results/new/direct
python -m pauli_transforms.benchmark --comparison chang --output results/new/fixed_locality
python -m pauli_transforms.run_permqit_comparison --methods factorial_float64 hahn_float64 --output results/new/matrix_units
python -m pauli_transforms.run_application_baselines spectral --output results/new/spectral
python -m pauli_transforms.run_random_dynamics_comparison --output results/new/dynamics
python -m pauli_transforms.ising_example --output results/new/ising
python -m pauli_transforms.plot --data results/new --output figures/new
```

Run experiments sequentially with fresh output directories. Add `--smoke` to
an experiment command for a small check; `--help` lists its options.
These commands use the current implementation and its default grids. To repeat
the original configurations with the measured sources, use the
[archived runners](data/historical_sources/README.md).

The Ising timing comparison additionally needs QuTiP:

```bash
python -m pip install '.[benchmarks,piqs]'
python -m pauli_transforms.run_application_baselines ising --output results/new/ising_runtime
python -m pauli_transforms.plot --data results/new --output figures/new
```

Setup for the optional public Anschuetz and native permqit comparisons is also
in the [archive instructions](data/historical_sources/README.md#external-comparisons).

## Tests and data

```bash
python -m unittest discover -s tests -v
```

- [Thesis data](data/thesis/README.md): saved measurements and configurations.
- [Accuracy checks](data/accuracy/README.md): checks, results and known numerical limitations.
- [Figure index](manuscript/figures.json): thesis labels and source files; paths are relative to the repository root.

The implementation, tests and documentation were developed with OpenAI Codex
assistance. Released under the [MIT license](LICENSE).
