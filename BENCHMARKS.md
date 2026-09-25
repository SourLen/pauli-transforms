# Reproducing the thesis figures

Run these commands from the repository root after installing `.[benchmarks]`.

## Redraw the saved measurements

```bash
python -m pauli_transforms.plot --data data/thesis --output figures
```

This exports the seven benchmark figures currently included in the thesis as
PDF, SVG and PNG. The input data and their provenance are described in
[`data/thesis/README.md`](data/thesis/README.md). Plotting does not run benchmarks.
The default selection matches the revised manuscript on 25 September 2026. Figure labels, datasets and source provenance are recorded in `FIGURE_MANIFEST.json`:

| Export | Manuscript location |
| --- | --- |
| `general_conversion_public` | `fig:appendix-general-conversion`, through n=40 |
| `matrix_unit_schur` | `fig:permqit-matrix-unit-conversion` |
| `fixed_locality` | `fig:comparison-fixed-locality`, first use, weights 2,4,6,8 |
| `fixed_weight_cache` | `fig:comparison-fixed-weight-cache`, common orbit-image preparation |
| `spectral_comparison` | `fig:spectral_comparison`, complete eigensystems |
| `random_dynamics_comparison` | `fig:random_dynamics_comparison` |
| `ising_example` | `fig:ising_example` |

Add `--supplementary` to also export `general_conversion` (without the public
Anschuetz curve) and `conversion_accuracy`, as well as `random_dynamics_diagnostics`. These plots are no longer
displayed in the manuscript. Their observations are retained. The experimental
general-eigensystem PIQS comparison is excluded, as in the current thesis.

## Accuracy checks

The [accuracy report](data/accuracy/README.md) documents quick regressions and
independent wide sweeps, including failed shared-factor cases. Accuracy data
are separate from historical timing observations.

## Historical measurement sources

[HISTORICAL_SOURCES.md](HISTORICAL_SOURCES.md) records recovered measured-source
files and hash verification. Retained timing rows were not rerun or relabelled
as measurements of this release. The September 21–23 extensions are preserved
under `data/thesis/current`, with per-file original and included hashes.

## Run new measurements

```bash
python -m pauli_transforms.benchmark --comparison anschuetz --output results/new/direct
python -m pauli_transforms.benchmark --comparison chang --output results/new/fixed_locality
python -m pauli_transforms.run_permqit_comparison --methods factorial_float64 hahn_float64 --output results/new/matrix_units
python -m pauli_transforms.run_spectral_comparison --output results/new/spectral
python -m pauli_transforms.run_random_dynamics_comparison --output results/new/dynamics
python -m pauli_transforms.ising_example --output results/new/ising
python -m pauli_transforms.plot --data results/new --output figures/new
```

Run the commands sequentially, on an otherwise idle machine. Each runner fixes
numerical libraries to one thread. Output directories must be empty, so new
measurements cannot overwrite previous ones. Add `--smoke` to each of the six
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

### Native permqit comparison

To include native permqit in the matrix-unit comparison, use Python >=3.14.
The pinned upstream revision declares Python >=3.12, but its `typing.Generator[T]`
annotations require Python >=3.13, and its dependency `qics` 1.1.3 pins an
incompatible Numba version on Python 3.13. CI uses Python 3.14.

```bash
python -m pip install '.[benchmarks,permqit]'
git clone https://github.com/bbbergh/permqit.git _external/permqit
git -C _external/permqit checkout 22af3cd245bd0e950df49f6ce16ccd422b6eb2c5
python -m pauli_transforms.run_permqit_comparison --output results/new/matrix_units
```

This replaces the local-only matrix-unit command above. The runner verifies
that the installed Python sources match the clean pinned checkout; use
`--permqit-source` to supply another checkout location. It selects the CPU
backend and one numerical-library thread before importing permqit.
`--smoke` runs a small functional check. Optional native correctness tests run
with the ordinary unittest command when permqit is installed.

Each method, size and input runs in a fresh process, because permqit caches
prepared objects. First use includes preparation of the normalized native map
(or our shared factors/Hahn kernels), input packing and complete block output.
Seven further applications reuse this preparation; their within-trial median
is the cached observation. The plot reports quartiles across five independent
inputs. Imports, input generation, validation and file writes are excluded.
Preparation is recorded separately; its median and the first-application median
need not add to the median total. No eigensolver enters this comparison.

Inputs are dense, generally non-Hermitian literal entries `a[r,s,t]`, sampled
as standard complex Gaussians divided by `sqrt(orbit_size(n,r,s,t))`, then jointly
normalized to Hilbert--Schmidt norm one. The seed is `20260919 + 1000*n + trial`.
The native map uses `EndSnBlockDiagonalizationGijswijt` and
`EndSnAlgebraIsomorphism`, including Gram/Cholesky normalization and reordering
into increasing row/column weight. No Pauli conversion is included.

Validation uses exact-integer binomial kernel sums with extended-precision
normalization and accumulation, plus independent computational-basis tests
through `n=8`. All 120 saved trials passed relative multiplicity-weighted
Hilbert--Schmidt error `1e-8` and entrywise `atol=1e-9`, `rtol=1e-8` gates.
The three largest relative errors are `3.38e-13` (shared factors), `1.12e-15`
(Hahn) and `1.54e-16` (permqit). The recorded environment and exact inputs are
in [`data/thesis/matrix_units/`](data/thesis/matrix_units/).

### Reuse saved inputs

To reuse exact saved random inputs, add `--input-directory` to a runner:

```bash
python -m pauli_transforms.benchmark --comparison anschuetz \
  --input-directory data/thesis/direct/inputs --output results/replayed/direct
python -m pauli_transforms.run_spectral_comparison \
  --input-directory data/thesis/spectral/inputs --output results/replayed/spectral
python -m pauli_transforms.run_random_dynamics_comparison \
  --input-directory data/thesis/dynamics/inputs --output results/replayed/dynamics
python -m pauli_transforms.run_permqit_comparison \
  --input-directory data/thesis/matrix_units/inputs --output results/replayed/matrix_units
```

The fixed-locality runner accepts the same option with
`data/thesis/fixed_locality/inputs`. Saved dynamics inputs include the original
state blocks. Replaying inputs preserves the problem instances; runtimes depend
on the machine and software versions.

## Protocol

| Experiment | Sizes | Inputs × repetitions | Seed |
| --- | --- | --- | --- |
| General conversion | 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40 | 3 × 7 | 20260917 |
| Fixed locality and common cache, weight 2 or 4 | 4, 8, 12, 16, 20, 24, 32, 40 | 3 × 7 | 20260914 |
| Fixed locality and common cache, weight 6 or 8 | 8, 12, 16, 20, 24, 32, 40 | 3 × 7 | 20260914 |
| Matrix-unit conversion | 2, 4, 6, 8, 10, 12, 16, 20 | 5 fresh-process trials; 7 cached applications each | 20260919 |
| Complete eigensystems | 2, 3, 4, 5, 8, 12, 16, 20 | 3 × 5 | 20260917 |
| Random dynamics | 2, 3, 4, 5, 8, 12, 16, 20 | 3 × 5 | 20260918 |

Except for the matrix-unit protocol above, each method/input has one warmup.
Plots show medians and interquartile ranges.
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
