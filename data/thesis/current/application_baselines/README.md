# Spectral and Ising comparisons

Saved measurements from 26 September 2026 for the thesis application figures.

| Comparison | Sizes | Trials |
| --- | --- | --- |
| Complete spectra | 2,3,4,5,6,8,10,12,16,20; full space through 10 | 405 |
| Ising dynamics | 2,3,4,5,8,12,16,20,30,40 | 140 |

Trials include fresh construction and the complete eigensystem or expectation
curve. Imports, input generation, validation and file writes are outside the
timers. Spectral runs use three inputs, one warmup and five repetitions;
Ising runs use one warmup and seven repetitions. Both use one numerical-library
thread. The environment was an AMD Ryzen 7 7735U, Python 3.14.7, NumPy 2.4.6,
SciPy 1.17.1 and QuTiP 5.3.1; method order uses seed `20260926`.

Spectral methods return all eigenpairs, explicitly in full space or as Schur
block representatives with multiplicities. Ising uses `g=1`, `h=0.5`, `p=0.6`,
with the Hamiltonian scaled by `1/n` and 241 times in `[0,12]`. Its two routes
compare general conversion with PIQS collective-spin construction, followed by
the same block eigensolver and expectation routine.

Configurations, exact inputs, source hashes and per-trial diagnostics are in
`spectral/` and `ising/`. Every trial passes the recorded `1e-8` validation gates.
The largest relative spectrum error is `8.83e-15`; the largest absolute
magnetization error is `3.74e-15`. The
[independent accuracy checks](../../../accuracy/README.md) provide separate
conversion and application tests.

## Reproduce

From the repository root:

```bash
python -m pauli_transforms.plot --data data/thesis --output figures
python -m pip install '.[benchmarks,piqs]'
python -m pauli_transforms.run_application_baselines spectral \
  --input-directory data/thesis/current/application_baselines/spectral/inputs \
  --output results/new/spectral
python -m pauli_transforms.run_application_baselines ising --output results/new/ising_runtime
python -m pauli_transforms.plot --data results/new --output figures/new
```

Use fresh output directories. Add `--smoke` for a small check. QuTiP is needed
only for new Ising timing measurements.

[manifest.json](manifest.json) records file hashes and metadata redactions.
Raw observations retain their original bytes. The
[archive instructions](../../../historical_sources/README.md#spectral-and-ising-comparisons-from-26-september)
reconstruct the original drivers and their package snapshot at revision
`29fd29769c62fccda90a1ac3031adbe462510cd5`.
