# Reproducing the thesis figures

Run these commands from the repository root after installing `.[benchmarks]`.

## Redraw the saved measurements

```bash
python -m pauli_transforms.plot --data data/thesis --output figures
```

This exports the eight figures currently included in the thesis as PDF, SVG and
PNG. The input data and their provenance are described in
[`data/thesis/README.md`](data/thesis/README.md). Plotting does not run benchmarks.

## Run new measurements

```bash
python -m pauli_transforms.benchmark --comparison anschuetz --output results/new/direct
python -m pauli_transforms.benchmark --comparison chang --output results/new/fixed_locality
python -m pauli_transforms.run_spectral_comparison --output results/new/spectral
python -m pauli_transforms.run_random_dynamics_comparison --output results/new/dynamics
python -m pauli_transforms.ising_example --output results/new/ising
python -m pauli_transforms.plot --data results/new --output figures/new
```

Run the commands sequentially, on an otherwise idle machine. Each runner fixes
numerical libraries to one thread. Output directories must be empty, so new
measurements cannot overwrite previous ones. Add `--smoke` to each of the five
experiment commands for a small functional check. Smoke timings are not thesis
measurements.

The commands above run the two local conversion methods. To include the original
public Anschuetz routine in the general and spectral comparisons, obtain its
pinned source once:

```bash
git clone https://github.com/bkiani/symmetric_hamiltonians.git _external/symmetric_hamiltonians
git -C _external/symmetric_hamiltonians checkout 24ce1a5fe4f3234f5f9a2f1ad65c2909423d7dc8
```

Then replace the general and spectral commands with:

```bash
python -m pauli_transforms.benchmark --comparison anschuetz \
  --anschuetz-source _external/symmetric_hamiltonians --output results/new/direct
python -m pauli_transforms.run_spectral_comparison \
  --methods thesis anschuetz_optimized anschuetz_public_original \
  --anschuetz-source _external/symmetric_hamiltonians --output results/new/spectral
```

The adapter verifies the SHA-256 of `utils.py`. It applies two import-compatibility
changes in memory: `np.math.factorial` becomes `math.factorial`, and an unused
SymPy helper is replaced by an error stub. It calls the original
`construct_matrix_blocks` afresh. Public measurements stop at five qubits and
have no cached mode. External code is not included in this repository.

To reuse exact saved random inputs, add `--input-directory` to a runner:

```bash
python -m pauli_transforms.benchmark --comparison anschuetz \
  --input-directory data/thesis/direct/inputs --output results/replayed/direct
python -m pauli_transforms.run_spectral_comparison \
  --input-directory data/thesis/spectral/inputs --output results/replayed/spectral
python -m pauli_transforms.run_random_dynamics_comparison \
  --input-directory data/thesis/dynamics/inputs --output results/replayed/dynamics
```

The fixed-locality runner accepts the same option with
`data/thesis/fixed_locality/inputs`. Saved dynamics inputs include the original
state blocks. Replaying inputs preserves the problem instances; runtimes depend
on the machine and software versions.

## Protocol

| Experiment | Sizes | Inputs × repetitions | Seed |
| --- | --- | --- | --- |
| General conversion | 2, 3, 4, 5, 6, 8, 10, 12, 16, 20 | 3 × 7 | 20260917 |
| Fixed locality, weight 2 or 4 | 4, 8, 12, 16, 20, 24, 32, 40 | 3 × 7 | 20260914 |
| Complete eigensystems | 2, 3, 4, 5, 8, 12, 16, 20 | 3 × 5 | 20260917 |
| Random dynamics | 2, 3, 4, 5, 8, 12, 16, 20 | 3 × 5 | 20260918 |

Each method/input has one warmup. Plots show medians and interquartile ranges.
General coefficients are independent real Gaussians with variance
`1 / (binom(n+3,3) * orbit_size)`. Fixed-weight inputs are random real linear
combinations of all orbit averages of that weight. Conversion input seeds are
`seed + 1009*n + 104729*instance + 37*weight`; application seeds are
`seed + 1009*n + instance`.

First-use conversion timings include fresh tables; cached timings reuse them
and record their preparation separately. Both include input formatting and
output construction. General conversion returns all dense Schur blocks;
fixed-locality conversion returns all blocks as CSR matrices.

Spectral trials include fresh tables, conversion and all eigenvectors and
eigenvalues using `scipy.linalg.eigh(driver="evr")`. Dynamics trials share fresh
tables across the conversions of `H`, `rho` and `M`, then use
`numpy.linalg.eigh` and evaluate the complete expectation curve. The curve stage
includes changes to the energy basis. Random states have blocks
`rho_k = (d_k/2**n) G_k G_k† / Tr(G_k G_k†)`, where `G_k` is a complex Gaussian
matrix of shape `(d_k, 2*d_k)`.

Dynamics uses 241 times in `[0,12]`, plus a sweep at `n=12` over
`L = 1, 8, 32, 128, 512, 2048`. A single time means `t=0`.
The Ising example uses `g=1`, `h=0.5`, `p=0.6`, spectrum size 12 and curve sizes
8, 20 and 40. Its Hamiltonian is
`H = -(g sum_{i<j} Zi Zj + h sum_i Xi)/n`.

Input generation, reference calculations, validation and file writing are
outside timers. Small cases use full Pauli matrices and singlet/Dicke
projections; larger cases use the documented round-trip or independent
conversion checks. Fixed-locality and Ising checks also use collective-spin
constructions. Applications check state normalization, positivity, eigenpair
residuals and expectation curves at tolerance `1e-8`. Direct conversion uses
absolute tolerance `1e-9` and relative tolerance `1e-8`.

Each run saves its inputs, raw trials, configuration, environment and source
hashes. Invalid trials are retained and cause failure; plots reject incomplete
groups. Direct-conversion worker timeouts are recorded with the interrupted
phase. Fresh timings from this simplified runner form a new campaign; they
are never pooled with the historical observations.
