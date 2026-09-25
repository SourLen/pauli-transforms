# Accuracy protocols and detailed results

This document contains the accuracy details accompanying the concise summaries
in Appendix C of the thesis. The [accuracy report](README.md) gives reproduction
commands and campaign provenance. The observations below are the existing
25 September 2026 campaign, not new measurements.

## Error norms

For a computed operator $\widetilde A$ and reference $A$, the relative error is
$\|\widetilde A-A\|_{HS}/\|A\|_{HS}$. The coordinate norms include the orbit sizes
and Schur multiplicities. With $\Delta A=\widetilde A-A$, orthogonality gives

$$
\begin{aligned}
\|\Delta A\|_{HS}^2
&=\sum_{r,s,t}v^t_{rs}|\Delta a^t_{rs}|^2\\
&=2^n\sum_{\mathbf{k}}\frac{n!}{k_I!k_X!k_Y!k_Z!}
  |\Delta\alpha_{\mathbf{k}}|^2\\
&=\sum_k\mu_k\|\Delta H_k\|_F^2.
\end{aligned}
$$

Here $v^t_{rs}$ is the matrix-unit orbit size, $\mathbf{k}$ gives the counts of
the four Pauli letters, $\mu_k=\binom nk-\binom n{k-1}$ is the Schur multiplicity,
and $F$ denotes the Frobenius norm of one representative block. Pauli coefficients
multiply sums of distinct words, as described in the [library README](../../README.md).
Zero references use absolute errors.

Block errors use the spectral norm (largest singular value), denoted
$\|\cdot\|_\infty$ in the thesis. A global Hilbert–Schmidt error implies
$\|\Delta H_k\|_\infty\leq\|\Delta A\|_{HS}/\sqrt{\mu_k}$, but does not ensure
a small relative error in every sector.

## Coordinate conversions

### Inputs and independent references

For `n=0,...,6`, literal Pauli tensor sums and matrix entries provide independent
coordinates. Pauli coefficients are obtained by traces, and Schur blocks by
projection onto explicit normalized singlet/Dicke representatives. All six
directed conversions are checked with and without prepared tables. Inputs are
zero and identity operators, one and two Y factors where available, random
Hermitian and complex non-Hermitian operators, a boundary matrix-unit orbit,
and the symmetric-sector projector divided by $\sqrt{n+1}$. Their Hilbert–Schmidt
norm is one except for zero and the alternating diagonal family
$10^8Z^{\otimes n}/\sqrt{2^n}$. The three coordinate norm formulas above are also
checked against the dense Frobenius norm.

At `n=8,12,20,30,40`, all six full conversions are tested on
$(|+\rangle\langle+|)^{\otimes n}$. Its matrix entries and nonzero Pauli
coefficients are $2^{-n}$. Its only nonzero Schur block has entries
$2^{-n}\sqrt{\binom nr\binom ns}$.

Additional matrix-unit/Schur tests use inputs supported on the central weight
pair $r=s=\lfloor n/2\rfloor$: random coordinates and a central Dicke projector.
Forward random entries are independent complex Gaussians divided by
$\sqrt{v^t_{rs}}$. Independently generated inverse inputs have one random central
entry per sector, divided by $\sqrt{\mu_k}$. References evaluate the Gijswijt
coefficient sum with exact integers and 80-digit decimal square roots and
accumulation. The actual binary64 inputs enter this calculation exactly.
Reference outputs are rounded to binary64 only for comparison. Round trips are
recorded separately from comparisons with independently generated references.

Separate Hahn-kernel checks at `n=0,1,2,6,12,20,30,40` cover selected central and
boundary weight pairs, both weight orders and zero polynomial entries. They
measure spectral error against the 80-digit reference and the orthogonality
residual $\|UU^T-I\|_\infty$.

The generating rules and seeds are recorded in
[`accuracy.py`](../../pauli_transforms/accuracy.py) and
[`conversions_20260925.json`](conversions_20260925.json).

### Tolerances and precision

The absolute and relative tolerances were fixed before the campaign at `1e-11`
and `1e-10`. An error with reference norm $b$ passes when its absolute value is
at most $10^{-11}+10^{-10}b$. For Schur output, both the weighted global error
and every block's spectral error must pass. Relative block errors are reported
only when the reference-block norm exceeds `1e-12` times the largest reference
block norm. The other blocks retain absolute errors and the same acceptance
formula, including the zero-reference case.

Default computations use binary64 tables and complex128 outputs. The extended
array type on the recorded platform has a 64-bit significand. It is not a fully
higher-precision conversion: Hahn normalization still uses binary64 logarithms,
and output coordinates remain complex128. Kernel comparisons are also evaluated
in binary64 after construction.

### Recorded results and tables

The [conversion summary](conversions_20260925.md) lists errors and failures by
family. The retained [LaTeX table](conversions_20260925.tex) separates the dense,
central-weight and product-projector comparisons, then the kernel checks.
In that table, `M`, `P` and `H` mean matrix-unit, Pauli and Schur coordinates.
“Block absolute” is the maximum forward block spectral error. The `1e8` input
scaling accounts for the larger dense absolute errors. “64” means binary64 and
“ext.” the platform's extended arithmetic described above.

All dense checks pass, with directed relative errors below `2.5e-15` and
matrix-unit/Schur round-trip errors below `7.5e-16`. All tested larger Hahn
conversions pass. Of the 1,764 recorded observations, 13 fail: shared-factor
binary64 cases at `n=30,40`. Its largest directed relative error is `5.12e-8`,
and selected central round trips reach `3.95e-8`. Extended arithmetic reduces
the errors in the tested central-weight family, without establishing a general
accuracy guarantee. Kernel observations are distinct from full conversions.

## Applications

### Inputs and references

Both Hahn and shared-factor conversions are tested in double precision. At
`n=2,...,6`, Ising and random Hermitian Hamiltonians and observables are built
from literal Pauli tensors, with reference blocks obtained by normalized
singlet/Dicke projection. Random operators have Hilbert–Schmidt norm one.
The initial states are independently constructed tensor powers of single-qubit
states with Bloch vectors `(0.6,0,0)` for Ising and `(0.25,0.35,-0.2)` for the
random family. Both cached and uncached conversion are tested.

Ising checks at `n=8,12,20,40` use independent collective-spin Hamiltonian,
observable and product-state blocks, with `g=1`, `h=0.5`, `p=0.6`. They use cached
conversion. None of these reference blocks uses either coordinate conversion.
The [application script](../../pauli_transforms/application_accuracy.py) and
[configuration](applications_20260925/config.json) specify the generating rules
and seeds. References and computations use double precision.

Dynamics is checked at 241 equally spaced times in `[0,12]`, including zero.
The implemented phase sum is compared with direct matrix exponentials in the
full Hilbert space for `n<=6`, and in collective-spin blocks at larger sizes.

### Diagnostics and acceptance

Hamiltonian conversion errors are recorded separately from numerical
eigensystem residuals $\|\widehat H_kV_k-V_k\Lambda_k\|_\infty$. The observations
also retain Hermiticity before symmetrization, the size of the correction,
orthogonality residuals and block eigenvalue errors. Dense full spectra are
compared including the multiplicities $\mu_k$. Eigenvector entries are not an
accuracy criterion.

Converted blocks use the absolute-plus-relative threshold
$10^{-10}+10^{-8}\|A_k^{\mathrm{ref}}\|_\infty$. Relative errors are recorded for
every nonzero reference block, with a flag when the absolute tolerance dominates.
Zero blocks use absolute errors only. Expectation errors use
$10^{-10}+10^{-8}\|M^{\mathrm{ref}}\|_\infty$, where the observable norm is the
maximum reference-block norm. The script applies the corresponding spectral,
orthogonality, trace and positivity checks when deciding whether a configuration
passes.

State trace, Hermiticity, minimum eigenvalue and imaginary expectation residuals
are retained. Minimum eigenvalues are evaluated on the Hermitian part, with the
symmetrization correction recorded. Negative eigenvalues and imaginary parts
are not discarded from the diagnostics. The cancellation ratio is the sum of
absolute phase-sum summands divided by the absolute computed expectation, over
nonzero sampled values. Exact zeros are counted separately.

### Recorded results

All 48 configurations pass. Maxima from the saved
[summary](applications_20260925/summary.json) are shown below. Block errors and
eigensystem residuals use spectral norm, and dynamics errors are absolute
expectation errors over the stated time grid. The
[LaTeX table](applications_20260925/summary.tex) is also retained here.

| Reference/input | Method | Sizes | Hamiltonian block error | Eigensystem residual | Dynamics error |
| --- | --- | --- | ---: | ---: | ---: |
| Collective Ising | Shared factor | 8,12,20,40 | 5.3e-15 | 1.3e-14 | 1.9e-13 |
| Collective Ising | Hahn | 8,12,20,40 | 1.9e-14 | 1.3e-14 | 3.8e-15 |
| Dense Ising | Shared factor | 2–6 | 8.9e-16 | 6.6e-16 | 3.6e-15 |
| Dense Ising | Hahn | 2–6 | 4.5e-16 | 1.0e-15 | 3.7e-15 |
| Dense random | Shared factor | 2–6 | 1.7e-16 | 3.7e-16 | 7.2e-16 |
| Dense random | Hahn | 2–6 | 1.9e-16 | 5.0e-16 | 8.9e-16 |

The maximum dense full-spectrum absolute error is `1.78e-15`. Across all cases,
the largest state trace error is `4.09e-14`, the smallest state eigenvalue is
`-2.46e-18`, the largest imaginary expectation residual is `9.71e-17`, and the
largest cancellation ratio is `5.01e4`. The maximum relative state-block error
is `8.13e-10`, whereas the maximum multiplicity-weighted relative Hilbert–Schmidt
state error is `1.86e-13`. Per-block and per-time observations are preserved in
[`results.json`](applications_20260925/results.json).

These are finite tests of the stated families and time grid. They do not
establish accuracy for arbitrary random applications at `n=40`, arbitrarily
long times, or a uniform stability or bit-complexity bound.
