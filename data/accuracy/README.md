# Independent accuracy checks, 25 September 2026

These are accuracy observations, not new runtime measurements. The conversion
and application sweeps were run from a clean checkout of
`3829b72425ab20ae0e3fa3790e0bc77f441dd503`, using Python 3.14.7, NumPy 2.4.6
and SciPy 1.17.1, with one numerical-library thread. Each file records its
references, generating rules, seeds, precision, source hashes and tolerances.

## Run the checks

Install `.[benchmarks]`, then use a new output path for each campaign:

```sh
python -m unittest discover -s tests -v
python -m pauli_transforms.accuracy --output results/accuracy/quick.json
python -m pauli_transforms.accuracy --sweep --output results/accuracy/conversions.json
python -m pauli_transforms.application_accuracy --output results/accuracy/applications
python scripts/check_munoz_dictionary.py --max-n 6 --output results/accuracy/munoz.json
```

The quick regressions assert a small deterministic domain. The wide conversion
sweep saves failures as observations and reports their count; a zero process
exit alone does not mean every numerical case passed. The application sweep
returns a nonzero status if a case fails. No tolerance was raised after a failure.

## Results and scope

- `conversions_20260925.json` contains 1,764 observations, including 13 failures
  of the float64 shared-factor backend on selected inputs at `n=30,40`.
  All six directions, both cached and uncached, pass for literal dense systems
  at `n=0,...,6`. The largest relative forward Hilbert–Schmidt error is
  `2.48e-15`. The `1e8` scaled input has an absolute block error around `2e-8`.
- Larger full transformations at `n=8,12,20,30,40` use analytic product
  projectors for all six directions and central-weight families for both
  matrix-unit/Schur directions. The latter references evaluate exact integer
  coefficient sums and 80-digit Decimal square roots and accumulation. Hahn's
  largest relative error is `2.46e-15`; shared-factor float64 reaches `5.12e-8`.
  Separate round-trip observations are labelled as such. These representative
  families do not validate arbitrary large inputs.
- Selected Hahn kernels through `n=40` agree with independent 80-digit
  references to spectral error `1.09e-14`; the largest orthogonality residual
  is `1.29e-14`. Kernel checks are distinct from full conversion checks.
- `applications_20260925/` contains 48 passing configurations. Literal dense
  tensor/exponential references cover `n=2,...,6`; independent collective-spin
  Ising references cover `n=8,12,20,40`. The maximum expectation error over
  241 times in `[0,12]` is `1.87e-13`. Conversion, eigensolver, state and
  phase-sum diagnostics are saved separately. References are double precision,
  not higher precision; this is not a uniform-in-time guarantee.
- `munoz_dictionary_20260925.json` records exact agreement for all 209 Pauli
  orbits through `n=6` (411,824 matrix entries and 2,057 binary-sum
  representatives), including one and two Y factors and orbit-sum norms.

Errors use the operator Hilbert–Schmidt norm, including orbit sizes and Schur
multiplicities, and largest-singular-value block norms. Zero references use
absolute errors. Small-block handling and tolerances are declared in the
scripts and metadata. Global relative error does not bound relative error in
every sector. Wider arrays are not full higher-precision computations: Hahn
normalization still uses binary64 logarithms and outputs are complex128.

The generated `.tex` summaries are included in the thesis. No uniform stability
or bit-complexity result is inferred from these finite checks. The repository
does not implement the quantum-copy measurement protocol discussed in the thesis.

For publication, one BLAS library path in the conversion metadata was reduced
to its basename. `metadata_redactions.json` records the original and included
hashes. No per-case observation or numerical result was changed.
