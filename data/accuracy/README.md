# Accuracy checks

Run from the repository root after installing `.[benchmarks]`:

```bash
python -m pauli_transforms.accuracy --output results/accuracy/quick.json
python -m pauli_transforms.accuracy --sweep --output results/accuracy/conversions.json
python -m pauli_transforms.application_accuracy --output results/accuracy/applications
python scripts/check_munoz_dictionary.py --max-n 6 --output results/accuracy/munoz.json
```

Use new output paths. The wide conversion sweep records failures without a
nonzero exit status; inspect its reported failure count.

## Saved results

| Check | Record | Result |
| --- | --- | --- |
| Conversions | [JSON](conversions_20260925.json) | 1,764 observations; 13 float64 shared-factor failures at selected inputs with `n=30,40`; all tested Hahn cases pass |
| Applications | [JSON](applications_20260925/results.json) | 48 configurations pass; maximum expectation error `1.87e-13` |
| Muñoz dictionary | [JSON](munoz_dictionary_20260925.json) | Exact agreement for all 209 Pauli orbits through `n=6` |

These checks used revision `3829b72425ab20ae0e3fa3790e0bc77f441dd503`,
Python 3.14.7, the versions in [requirements.txt](requirements.txt), and one
numerical-library thread. [PROTOCOL.md](PROTOCOL.md) gives the input families,
references, norms, tolerances and detailed tables. The results cover the tested
inputs and time grids; they do not establish uniform numerical stability.

The later application timing campaign separately retains
[405 spectral](../thesis/current/application_baselines/spectral/validation.json)
and [140 Ising](../thesis/current/application_baselines/ising/validation.json)
trials, all passing their `1e-8` gates.

[metadata_redactions.json](metadata_redactions.json) records the removal of
one machine-local library path. Numerical observations were preserved.
