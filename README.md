# Pauli transforms

Python code accompanying the bachelor's thesis,
*Efficient Coordinate Transformations for Permutation-Invariant Quantum Systems*.

The library converts between matrix-entry orbits, Pauli orbits and Schur blocks.
The accompanying experiments reproduce the conversion, complete-spectrum,
dynamics and Ising figures in the thesis.

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

Entry keys `(w, h0, h1)` describe literal matrix entries: `w` is the number
of differing row/column bits, `h0` counts their common ones, and `h1` counts
column ones among the differing bits. The Chapter 5 matrix-unit indices are
`(r, s, t) = (w+h0-h1, h0+h1, h0)`.

Blocks are ordered by `k = 0, ..., floor(n/2)`, with side `n-2*k+1` and
multiplicity `binom(n,k)-binom(n,k-1)`. Multiplicities matter for traces,
spectra and expectation values. 

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

`prepare(n)` builds fresh Krawtchouk/Hahn tables. Pass the returned tables
to reuse them, omitting tables includes preparation in the call.
`prepare(n, backend="factorial")` selects the Chapter 5 factorization.
Both backends are bidirectional. The factorial formula can suffer
floating-point cancellation, in the thesis we use Hahn for most benchmarks.
The code uses ordinary NumPy matrix products; we do not implement the fast
polynomial transforms discussed in the outlook.

The optional `permqit` comparison uses its native Gijswijt map and Gram-matrix
normalization. It converts matrix-unit orbit coefficients `(r, s, t)` to Schur
blocks and measures preprocessing, first use and cached conversion separately.
Rerunning it requires Python 3.13 or newer and the pinned upstream dependency;
the library and saved-data plots do not require permqit.

## Figures and benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for commands to redraw all seven current thesis
benchmark figures from the included observations or to measure them again. Saved
input coefficients, seeds, timing records and environment metadata are under
[`data/thesis/`](data/thesis/). Fresh runs write to a separate output directory.
Historical runtime values describe the original machine, reruns produce new
measurements.

The default export follows the manuscript as of 19 September 2026. The earlier
general-conversion plot without public code and the conversion-accuracy plot
remain available with `--supplementary`.

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

Released under the [MIT license](LICENSE). NumPy, SciPy and the plotting
dependencies are installed separately under their own licenses. No external
Anschuetz or permqit source is distributed here.
