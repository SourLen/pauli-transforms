# Data used in the thesis

These are the original observations, not timings of this cleaned-up repository.
From the repository root, recreate the seven current thesis benchmark figures with:

```sh
python -m pauli_transforms.plot --data data/thesis --output figures
```

The command writes PDF, SVG and PNG files, plus the unrounded plotted statistics.
Medians and interquartile ranges use complete trial groups. Failed, missing,
duplicate or nonfinite trials cause an error. The public implementation has no
cached curve and was measured only through n=5. Errors below 1e-17 are clipped
only for display. Fresh measurements will have different timings.

The current selection follows the manuscript on 19 September 2026: conversion
plots are in Appendix C.2 and the Anschuetz error plot is no longer displayed.
Add `--supplementary` for the older general-conversion and accuracy figures.

| Folder | Included thesis figures | Accepted trials |
|---|---|---:|
| `direct` | `general_conversion_public` | 924 |
| `fixed_locality` | `fixed_locality` | 1,344 |
| `matrix_units` | `matrix_unit_schur` | 120 |
| `spectral` | `spectral_comparison` | 300 |
| `dynamics` | `random_dynamics_comparison`, `random_dynamics_diagnostics` | 420 |
| `ising` | `ising_example` | No timings |

Each campaign has its configuration, raw trials, exact inputs and recorded
environment/checks. Matrix-unit conversion uses five inputs per size and method,
each in a fresh process; a cached observation is the median of seven subsequent
applications. Its plot checks complete groups, accuracy gates, timing sums,
cached medians and the hashes of the saved inputs. Dynamics includes the original
generated state blocks and reference curves. Direct conversion also retains 18 configured public-size-limit
records. The Ising spectrum has n=12; its curves have n=8,20,40, g=1, h=0.5,
p=0.6 and 241 times from 0 to 12. The old Ising timing campaign is unnecessary
for these plotted values and is omitted.

The source folders, relative to the original `Code/pauli_algorithm_comparison/results/`, are:

- `publication_comparisons_20260917/direct` and `.../fixed_locality`
- `spectral_comparison_20260917`
- `random_dynamics_20260918/final`
- `ising_example_20260918`
- `permqit_matrix_units_20260919`

[manifest.json](manifest.json) records the original and included SHA-256 hashes
for every data file. Trial rows, inputs, spectra and curves retain their original
bytes, including recorded library paths in the matrix-unit trial metadata.
Configuration/environment JSON omits machine-local paths and verbose
NumPy build output and unused generic-runner options; the manifest identifies
these changes. The small Ising
configuration transcribes the recorded model parameters. Historical code
snapshots and generated figures are omitted. Source hashes in the environment
records identify the implementations used for the measurements; they do not
refer to this repository's edited source.

For `matrix_units`, `manifest.json` contains the protocol, environment and source
hashes; `integrity_after.json` records the original source-integrity check.
`requirements.lock` retains the historical package versions, replacing only the
local permqit installation path with its verified upstream Git commit. The lock
describes the original Python 3.14 environment; it is not the installation
requirement for this library. Redrawing any of these figures needs no permqit
installation. The optional rerun is documented in [BENCHMARKS.md](../../BENCHMARKS.md).
