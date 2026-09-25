# Pauli transforms

Python code accompanying the bachelor's thesis,
*Efficient Coordinate Transformations for Permutation-Invariant Quantum Systems*.

The library converts between matrix-entry orbits, Pauli orbits and Schur blocks.
The accompanying experiments redraw the retained conversion, complete-spectrum,
dynamics and Ising measurements and provide separate commands for new measurements.

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
Its value is the coefficient of **each individual Pauli word** in that orbit.
Equivalently it multiplies the sum of distinct words, not the orbit average.

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
to reuse them, omitting tables includes preparation in the call.
`prepare(n, backend="factorial")` selects the Chapter 5 factorization.
Both backends are bidirectional. The factorial formula can suffer
floating-point cancellation, in the thesis we use Hahn for most benchmarks.
Independent tests of all six directions pass through `n=6`. Selected inputs
through `n=40` pass with Hahn, while the float64 shared-factor backend fails the
declared tolerances for some cancellation-sensitive inputs at `n=30,40`.
These sizes describe the tested families, not a universal accuracy threshold.
The [accuracy report](data/accuracy/README.md) includes the failures, independent
references, weighted Hilbert–Schmidt errors and block spectral errors.
Hahn kernels use analytical normalization evaluated with binary64 logarithms,
including when their recurrence uses an extended array dtype. The default
cached Hahn implementation has `O(n^4)` preparation and storage; the
fast-matrix-multiplication theorem applies to the shared-factor construction.
The code uses ordinary NumPy matrix products; we do not implement the fast
polynomial transforms discussed in the outlook.

The optional `permqit` comparison uses its native Gijswijt map and Gram-matrix
normalization. It converts matrix-unit orbit coefficients `(r, s, t)` to Schur
blocks and measures preprocessing, first use and cached conversion separately.
Rerunning it requires Python 3.14 or newer and the pinned upstream dependency;
the library and saved-data plots do not require permqit. We recommend checking out the features
of permqit for further comparison and a more sophisticated package.


## References and provenance

The Schur constructions follow [Gijswijt](https://arxiv.org/abs/0910.4515)
and [Vallentin](https://doi.org/10.1016/j.laa.2008.07.025), as developed in
Chapters 4–5 of the thesis. Comparison implementations evaluate the formulas
of [Anschuetz et al.](https://doi.org/10.22331/q-2023-11-28-1189)
(Appendix E) and [Chang, Larocca and Cerezo](https://arxiv.org/abs/2603.13072).
The optional native comparison uses Bergh and Parentin's
[permqit](https://github.com/bbbergh/permqit/tree/22af3cd245bd0e950df49f6ce16ccd422b6eb2c5).


The implementation, tests and documentation were developed with OpenAI Codex
assistance. This September 2026 release retains the thesis's numerical kernels
and selected measurements from the private development repository.
The new accuracy campaigns use immutable source revision
`3829b72425ab20ae0e3fa3790e0bc77f441dd503`. Historical runtime observations keep
their original identities. See [BENCHMARKS.md](BENCHMARKS.md), the
[figure manifest](FIGURE_MANIFEST.json) and
[historical source record](HISTORICAL_SOURCES.md) for the distinction between
redrawing saved data and collecting new timings.

Released under the [MIT license](LICENSE). NumPy, SciPy and the plotting
dependencies are installed separately under their own licenses.
