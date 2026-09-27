# Application comparisons, 26 September 2026

These are the measurements used in Figures 6.1 and 6.4 and Section C.3.1 of
the thesis PDF dated 27 September 2026. They contain 405 spectral trials and
140 Ising trials, all passing the recorded `1e-8` gates. All plotted timing
series come from this campaign; older timings and pilot runs are excluded.

## Inputs and timing

| Comparison | Sizes | Repetitions | Output |
| --- | --- | --- | --- |
| Random spectra | 2,3,4,5,6,8,10,12,16,20; full space through 10 | Three inputs, one warmup and five timed repetitions per method/input | Complete eigensystems |
| Ising | 2,3,4,5,8,12,16,20,30,40 | One warmup and seven timed repetitions per method | 241 magnetization values in `[0,12]` |

Each trial includes fresh construction and the complete requested result.
Imports, random input generation, validation and I/O are outside the timers.
The runs used one numerical-library thread on an AMD Ryzen 7 7735U, Python
3.14.7, NumPy 2.4.6, SciPy 1.17.1 and QuTiP 5.3.1. Method order is shuffled
within each repetition using seed `20260926`. Plots show medians and quartiles
of complete groups; all raw trials and their diagnostics are retained.

The spectral inputs are real Gaussian Pauli-orbit coefficients, with variance
`1 / (binom(n+3,3) * orbit_size)` and seed `20260917 + 1009*n + instance`.
The earlier saved inputs are reused byte-for-byte; sizes 6 and 10 follow the
same rule. All exact inputs are in `spectral/inputs/`. The illustrative `n=12`
spectrum is unchanged from the earlier campaign.

The Schur routes prepare fresh Hahn or optimized Anschuetz data and construct
all distinct blocks. The full-space baseline constructs the computational-basis
matrix directly with unnormalized Walsh transforms:

```text
H[b xor x,b] = sum_z c(x,z) i^popcount(x & z) (-1)^popcount(z & b).
```

Construction takes `O(n 4^n)` arithmetic and `O(4^n)` storage. Every route then
uses `scipy.linalg.eigh(driver="evr")` and returns all eigenvalues and vectors.
The full-space vectors are explicit computational-basis vectors; Schur vectors
are compressed representatives with sector multiplicities. This comparison
measures the benefit of symmetry reduction and conversion, including the
different output sizes.

For Ising, `g=1`, `h=0.5`, `p=0.6`, and both Hamiltonian terms have the factor
`1/n` used in the thesis. The general route converts Pauli coefficients for
the Hamiltonian, product state and magnetization. The PIQS route uses native
`jspin` and `dicke_blocks` in every spin sector, with

```text
H_k = -2g Jz_k^2/n - 2h Jx_k/n + (g/2) I,
M_k = 2 Jx_k/n,
rho_k = exp(2 atanh(p) Jx_k) / (2 cosh(atanh(p)))^n.
```

The state is evaluated using its known eigenvalues in the `Jx` basis. Blocks
are unweighted; the common expectation routine inserts Specht multiplicities
once. Both routes use the same block diagonalization and spectral phase sum.
This is a model-specific construction comparison, with preparation and complete
curve times reported separately. PIQS supplies the collective operators; no
QuTiP time integrator is used.

## Accuracy checks

Every diagnostic must be finite and at most `1e-8`. Spectral trials check
Hermiticity, eigenpair residuals, reconstruction, orthogonality and spectrum
errors. Schur outputs are compared with optimized Anschuetz blocks; full-space
spectra are compared with their multiplicity-expanded eigenvalues. Literal
Pauli tensor matrices additionally check construction through `n=5`.
Independent unit tests cover complex coefficients, including Y phases.

Ising trials compare the Hamiltonian, state and observable with independent
collective-spin formulas, using multiplicity-weighted relative Frobenius
errors. They also check trace, negative state eigenvalue mass, eigenpairs and
absolute expectation errors, without renormalizing or repairing the state.
Small-system tests compare the dynamics with literal tensors and matrix
exponentials. The large-system curve reference shares the downstream spectral
solver, so its agreement alone does not independently validate that solver.

| Recorded maximum | Value |
| --- | ---: |
| Relative spectrum error | `8.83e-15` |
| Spectral eigenpair residual | `7.91e-15` |
| Absolute magnetization error | `3.74e-15` |

See `spectral/validation.json`, `ising/validation.json` and the `checks` fields
in each `runs.jsonl`. These timing-validation gates are distinct from the
[independent accuracy campaigns](../../../accuracy/README.md), including the
48-configuration application sweep. They apply to the sampled inputs and times.

## Reproduce

From the repository root:

```sh
python -m pauli_transforms.plot --data data/thesis --output figures
python -m pip install '.[benchmarks,piqs]'
python -m unittest discover -s tests -p test_application_baselines.py -v
python -m pauli_transforms.run_application_baselines spectral \
  --input-directory data/thesis/current/application_baselines/spectral/inputs \
  --output results/new/spectral
python -m pauli_transforms.run_application_baselines ising \
  --output results/new/ising_runtime
python -m pauli_transforms.plot --data results/new --output figures/new
```

The first command redraws historical observations. The runners create separate
measurements of the current package and require empty output directories.
Add `--smoke` for a small functional run. Saved-data plotting and the spectral
runner do not require QuTiP.

`manifest.json` records the original and included file hashes. Only absolute
BLAS library paths in environment metadata were reduced to basenames. Raw
observations, exact inputs, execution order, diagnostics and source hashes
retain their original bytes. Duplicate CSV timing exports, repeated source
copies, pilot runs and local editing files are omitted.

The measured runner is archived in
[`data/historical_sources/application_baselines_20260926`](../../../historical_sources/application_baselines_20260926).
Its package snapshot matches revision
`29fd29769c62fccda90a1ac3031adbe462510cd5` byte-for-byte. Instructions to
reconstruct that exact layout are in
[HISTORICAL_SOURCES.md](../../../../HISTORICAL_SOURCES.md).
