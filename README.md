# Pauli transforms

Python code accompanying the bachelor's thesis,
*Efficient Coordinate Transformations for Permutation-Invariant Quantum Systems*.

The library converts between matrix-entry orbits, Pauli orbits and Schur blocks.
The accompanying experiments reproduce the conversion, complete-spectrum,
dynamics and Ising comparisons in the thesis.

## Install and use

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[benchmarks]'
python -m unittest discover -s tests -v
```

For the library alone, install with `python -m pip install -e .`.

```python
from pauli_transforms import prepare, pauli_to_schur, schur_to_pauli

n = 4
# H = sum_i Z_i + 0.2 sum_{i<j} X_i X_j
coefficients = {(0, 1, 0): 1.0, (2, 0, 0): 0.2}
tables = prepare(n)
blocks = pauli_to_schur(n, coefficients, tables)
recovered = schur_to_pauli(n, blocks, tables)
```

A Pauli key `(w, g0, g1)` means the counts
`(nI, nX, nY, nZ) = (n-w-g0, w-g1, g1, g0)`.
Its value is the coefficient of **each individual Pauli string** in that orbit.

Entry keys `(w, h0, h1)` are used to describe matrix entries, `w` is the number
of differing row/column bits, `h0` counts their common ones, and `h1` counts
column ones among the differing bits. The Chapter 5 matrix-unit indices are
`(r, s, t) = (w+h0-h1, h0+h1, h0)`.

Blocks are ordered by `k = 0, ..., floor(n/2)`, with side `n-2*k+1` and
multiplicity `binom(n,k)-binom(n,k-1)`.

## Code map

| Thesis construction | Module |
| --- | --- |
| Chapter 4, entry orbits ↔ Pauli orbits | `krawtchouk.py` (Krawtchouk recurrence) |
| Chapter 5, matrix-unit orbits ↔ Schur blocks | `schur_factorial.py` |
| Recurrence implementation used in the figures | `schur_hahn.py` |
| Composition, Pauli orbits ↔ Schur blocks | `transforms.py` |
| Comparison with Anschuetz et al. | `anschuetz_optimized.py`, `anschuetz_public.py` |
| Fixed-locality comparison with Chang et al. | `chang.py` |
| Appendix C.2, native permqit matrix-unit comparison | `permqit_adapter.py`, `run_permqit_comparison.py` |
| Spectra, invariant dynamics and the Ising model | `physical_models.py` |

`prepare(n)` builds the Krawtchouk/Hahn tables. Pass the returned tables
to reuse them; omitting tables includes preparation in the call.
`prepare(n, backend="factorial")` selects the Chapter 5 factorization.
Both backends are bidirectional. The factorial formula can suffer
floating-point cancellation; the thesis uses Hahn for most benchmarks.
Hahn kernels use analytical normalization evaluated with binary64 logarithms,
including when their recurrence uses an extended array dtype. The default
cached Hahn implementation has `O(n^4)` preparation and storage; the
fast-matrix-multiplication theorem applies to the shared-factor construction.
The code uses ordinary NumPy matrix products; we do not implement the fast
polynomial transforms discussed in the outlook.

## Thesis benchmarks and accuracy

```bash
python -m pauli_transforms.plot --data data/thesis --output figures
```

This redraws all eight data figures from saved observations, including the
full-space spectral baseline and the PIQS Ising comparison. No new timings
are collected. [BENCHMARKS.md](BENCHMARKS.md) gives the input families, timing
boundaries and commands for new measurements; the
[figure manifest](FIGURE_MANIFEST.json) maps outputs to thesis figures.

| Check | Recorded result |
| --- | --- |
| All six conversion directions, references through `n=6` | All pass |
| Selected larger conversions through `n=40` | Hahn passes; 13 shared-factor float64 cases fail at `n=30,40` |
| Independent application accuracy | 48 configurations pass; maximum expectation error `1.87e-13` |
| Spectral and Ising timing checks | All 405 spectral and 140 Ising trials pass their `1e-8` gates |

The [accuracy report](data/accuracy/README.md) links the raw results, norms,
references and tolerances. The timing gates are practical acceptance criteria;
the separate conversion sweep uses stricter tolerances. These finite checks
cover the stated inputs and time grids, without establishing uniform stability.

The optional comparisons require `.[benchmarks,permqit]` (Python 3.14+) or
`.[benchmarks,piqs]`. Saved-data plotting needs neither external package.
The PIQS comparison constructs the Ising inputs from collective-spin matrices;
it uses the same downstream block eigensolver and expectation routine.

## References and provenance

The Schur constructions follow [Gijswijt](https://arxiv.org/abs/0910.4515)
and [Vallentin](https://doi.org/10.1016/j.laa.2008.07.025), as developed in
Chapters 4–5 of the thesis. Comparison implementations evaluate the formulas
of [Anschuetz et al.](https://doi.org/10.22331/q-2023-11-28-1189)
(Appendix E) and [Chang, Larocca and Cerezo](https://arxiv.org/abs/2603.13072).
The optional native comparison uses Bergh and Parentin's
[permqit](https://github.com/bbbergh/permqit/tree/22af3cd245bd0e950df49f6ce16ccd422b6eb2c5).
The Ising baseline uses [PIQS (Shammah et al.)](https://doi.org/10.1103/PhysRevA.98.063815)
through QuTiP.

The implementation, tests and documentation were developed with OpenAI Codex
assistance. The measured-source revisions and original runtime observations
are preserved in the [data records](data/thesis/README.md) and
[historical source record](HISTORICAL_SOURCES.md). The current figure selection
includes the 26 September application comparisons added after release v0.2.0;
the original release remains unchanged.

Released under the [MIT license](LICENSE). NumPy, SciPy and the plotting
dependencies are installed separately under their own licenses.
