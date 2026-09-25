# Version 0.2.0 — thesis revision

This release supplies the numerical evidence and reproduction records used in
the revised thesis *Efficient Coordinate Transformations for
Permutation-Invariant Quantum Systems*.

- Independent checks cover all six conversion directions, literal dense
  references through six qubits, selected complete conversions through 40
  qubits, exact-integer/80-digit kernel references, weighted errors and
  block spectral errors. The 1,764 observations retain 13 failed float64
  shared-factor cases. Hahn passes the tested families.
- Separate application checks pass all 48 configurations, with conversion,
  eigensystem, state and finite-grid expectation diagnostics. The largest
  expectation error is `1.87e-13` over the specified 241-point grid.
- The Muñoz dictionary agrees exactly for all Pauli orbits through six qubits.
- Saved figures now include the general conversion through 40 qubits and
  all four fixed weights, with the common orbit-image comparison. Seven
  benchmark figures, two native diagrams and two accuracy tables are indexed
  by their LaTeX labels in `FIGURE_MANIFEST.json`.
- Recovered historical source files are hash-checked and supplied with
  executable replay commands. Original timing observations remain unchanged.

The accuracy campaigns were measured from commit
`3829b72425ab20ae0e3fa3790e0bc77f441dd503`; this later release packages their
results. Numerical kernels were not changed to make tests pass. The failed
shared-factor cases are documented limitations of those inputs and precision,
not a universal maximum system size. These finite checks do not prove uniform
floating-point stability, bit complexity or accuracy at arbitrarily long times.

Installation, tests and figure commands are documented in `README.md`,
`BENCHMARKS.md`, `data/accuracy/README.md` and `HISTORICAL_SOURCES.md`.
