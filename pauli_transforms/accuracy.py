"""Independent accuracy checks, separate from timing campaigns.

Run ``python -m pauli_transforms.accuracy --output data/accuracy/quick.json``
or add ``--sweep`` for dense n=0,...,6 and selected larger kernels/conversions.
No implementation under test is used to construct an expected operator.
"""

import argparse
from decimal import Decimal, localcontext
import hashlib
from itertools import product
import json
from math import comb, factorial, sqrt
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from . import krawtchouk, schur_factorial, schur_hahn, transforms


# Fixed before the campaign: absolute tolerance for zero inputs and reference
# blocks below 1e-12 of the largest block, plus a relative tolerance otherwise.
ATOL = 1e-11
RTOL = 1e-10
SMALL_BLOCK_RATIO = 1e-12
SEED = 20260925
PAULIS = (np.eye(2), np.array([[0, 1], [1, 0]]),
          np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]))


def tensor(factors):
    result = np.ones((1, 1), dtype=complex)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def orbit_keys(n):
    return [(r, s, t) for r in range(n + 1) for s in range(n + 1)
            for t in range(max(0, r + s - n), min(r, s) + 1)]


def pauli_keys(n):
    return [(w, z, y) for w in range(n + 1)
            for z in range(n - w + 1) for y in range(w + 1)]


def orbit_weight(n, r, s, t):
    return factorial(n) // (factorial(n-r-s+t) * factorial(r-t)
                            * factorial(s-t) * factorial(t))


def pauli_weight(n, w, z, y):
    return 2**n * factorial(n) // (factorial(n-w-z) * factorial(w-y)
                                  * factorial(y) * factorial(z))


def multiplicities(n):
    return [comb(n, k) - (comb(n, k-1) if k else 0) for k in range(n//2 + 1)]


def dense_pauli_orbits(n):
    """Literal tensor sums, including Y phases, without a binary kernel."""
    result = {key: np.zeros((2**n, 2**n), dtype=complex) for key in pauli_keys(n)}
    for word in product(range(4), repeat=n):
        key = word.count(1) + word.count(2), word.count(3), word.count(2)
        result[key] += tensor(PAULIS[letter] for letter in word)
    return result


def dense_schur_basis(n):
    """Columns are k literal singlets tensor normalized Dicke vectors."""
    singlet = np.array([0, 1, -1, 0]) / sqrt(2)
    result = []
    for k in range(n//2 + 1):
        paired = np.ones(1)
        for _ in range(k):
            paired = np.kron(paired, singlet)
        remaining = n - 2*k
        vectors = [np.array([int(x.bit_count() == j) for x in range(2**remaining)])
                   / sqrt(comb(remaining, j)) for j in range(remaining + 1)]
        result.append(np.column_stack([np.kron(paired, v) for v in vectors]))
    return result


def dense_orbit_coordinates(n, matrix):
    buckets = {key: [] for key in orbit_keys(n)}
    for x in range(2**n):
        for y in range(2**n):
            buckets[x.bit_count(), y.bit_count(), (x & y).bit_count()].append(matrix[x, y])
    return {key: sum(values) / len(values) for key, values in buckets.items()}


def dense_pauli_coordinates(n, matrix, orbits=None):
    if orbits is None:
        orbits = dense_pauli_orbits(n)
    return {key: np.vdot(orbit, matrix) / pauli_weight(n, *key)
            for key, orbit in orbits.items()}


def block_metrics(n, actual, reference):
    if any(not np.all(np.isfinite(a)) for a in [*actual, *reference]):
        return dict(nonfinite_output=True, passed=False)
    norms = [float(np.linalg.norm(h, 2)) for h in reference]
    errors = [float(np.linalg.norm(a-b, 2)) for a, b in zip(actual, reference)]
    threshold = SMALL_BLOCK_RATIO * max(norms, default=0)
    relative = [e/v for e, v in zip(errors, norms) if v > threshold]
    abs_hs = sqrt(sum(mu * float(np.linalg.norm(a-b))**2
                      for mu, a, b in zip(multiplicities(n), actual, reference)))
    ref_hs = sqrt(sum(mu * float(np.linalg.norm(b))**2
                     for mu, b in zip(multiplicities(n), reference)))
    return dict(absolute_hs_error=abs_hs, reference_hs_norm=ref_hs,
                relative_hs_error=abs_hs/ref_hs if ref_hs else None,
                maximum_block_spectral_error=max(errors, default=0),
                maximum_block_relative_error=max(relative, default=None),
                absolute_only_blocks=sum(v <= threshold for v in norms),
                passed=bool(abs_hs <= ATOL + RTOL*ref_hs and
                            all(e <= ATOL + RTOL*v for e, v in zip(errors, norms))))


def coefficient_metrics(n, actual, reference, kind):
    keys = orbit_keys(n) if kind == "orbit" else pauli_keys(n)
    weight = orbit_weight if kind == "orbit" else pauli_weight
    # sqrt(weight)*coefficient avoids squaring large combinatorial integers.
    error = np.array([sqrt(weight(n, *key)) * (actual.get(key, 0)-reference.get(key, 0))
                      for key in keys])
    ref = np.array([sqrt(weight(n, *key))*reference.get(key, 0) for key in keys])
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(ref)):
        return dict(nonfinite_output=True, passed=False)
    absolute, norm = float(np.linalg.norm(error)), float(np.linalg.norm(ref))
    return dict(absolute_hs_error=absolute, reference_hs_norm=norm,
                relative_hs_error=absolute/norm if norm else None,
                passed=bool(absolute <= ATOL + RTOL*norm))


def binomial(a, b):
    return comb(a, b) if 0 <= b <= a else 0


def decimal_kernel(n, r, s, precision=80):
    """Gijswijt coefficient sum in exact integers, square root at 80 digits.

    Independent of the Hahn recurrence and factorial matrix products. All
    arithmetic before final comparison to binary64 uses integers or Decimal.
    """
    ts = list(range(max(0, r+s-n), min(r, s)+1))
    with localcontext() as ctx:
        ctx.prec = precision
        result = []
        for k in range(min(r, s, n-r, n-s)+1):
            scale = (Decimal(comb(n-2*k, r-k))/Decimal(comb(n-2*k, s-k))).sqrt()
            result.append([scale * sum((-1 if (s-t-j) % 2 else 1)
                                      * binomial(n-k-r, j) * binomial(r-k, s-k-j)
                                      * binomial(k, s-t-j) for j in range(s-k+1))
                           for t in ts])
        return ts, result


def dense_cases(n, orbits, basis):
    rng = np.random.default_rng(SEED+n)
    side = 2**n
    yield "zero", np.zeros((side, side), dtype=complex)
    yield "identity", np.eye(side, dtype=complex) / sqrt(side)
    for label, key in [("odd_y", (1, 0, 1)), ("even_y", (2, 0, 2))]:
        if key[0] <= n:
            yield label, orbits[key] / np.linalg.norm(orbits[key])
    matrix = sum((rng.normal()+1j*rng.normal()) * orbit / sqrt(pauli_weight(n, *key))
                 for key, orbit in orbits.items())
    yield "complex_nonhermitian", matrix / np.linalg.norm(matrix)
    matrix = matrix + matrix.conj().T
    yield "hermitian", matrix / np.linalg.norm(matrix)
    # Individual boundary matrix-unit orbit |0...0><1...1|.
    matrix = np.zeros((side, side), dtype=complex)
    matrix[0, side-1] = 1
    yield "boundary_orbit", matrix
    # Symmetric-sector projector built directly, without an inverse transform.
    yield "single_sector", basis[0] @ basis[0].T / sqrt(n+1)
    # Alternating diagonal entries are Z^tensor(n), scaled by 1e8 in HS norm.
    yield "scaled_alternating", 1e8 * orbits[0, n, 0] / sqrt(side)


def dense_campaign(max_n=3):
    observations = []
    for n in range(max_n+1):
        orbits, basis = dense_pauli_orbits(n), dense_schur_basis(n)
        prepared = {b: transforms.prepare(n, backend=b) for b in ("hahn", "factorial")}
        for family, dense in dense_cases(n, orbits, basis):
            a, p = dense_orbit_coordinates(n, dense), dense_pauli_coordinates(n, dense, orbits)
            h = [u.conj().T @ dense @ u for u in basis]
            dense_norm = float(np.linalg.norm(dense))
            norm_checks = [coefficient_metrics(n, {}, a, "orbit")["reference_hs_norm"],
                           coefficient_metrics(n, {}, p, "pauli")["reference_hs_norm"],
                           block_metrics(n, h, h)["reference_hs_norm"]]
            observations.append(dict(n=n, family=family, direction="norm_identity", backend="literal",
                                     cache="none", reference="dense Frobenius norm", precision="binary64",
                                     absolute_hs_error=max(abs(x-dense_norm) for x in norm_checks),
                                     relative_hs_error=max(abs(x-dense_norm) for x in norm_checks)/dense_norm
                                     if dense_norm else None,
                                     passed=all(abs(x-dense_norm) <= ATOL+RTOL*dense_norm for x in norm_checks)))
            for cached in (False, True):
                q = prepared["hahn"][0] if cached else None
                for direction, actual, reference, kind in (
                        ("orbit_to_pauli", transforms.orbit_to_pauli(n, a, q), p, "pauli"),
                        ("pauli_to_orbit", transforms.pauli_to_orbit(n, p, q), a, "orbit")):
                    observations.append(dict(n=n, family=family, direction=direction, backend="krawtchouk",
                                             cache="cached" if cached else "uncached",
                                             reference="literal tensors and traces", precision="binary64",
                                             **coefficient_metrics(n, actual, reference, kind)))
                for backend, module in (("hahn", schur_hahn), ("factorial", schur_factorial)):
                    tables = prepared[backend] if cached else None
                    kernels = tables[1] if cached else None
                    forward = module.orbit_to_schur(n, a, kernels)
                    calculations = (
                        ("orbit_to_schur", block_metrics(n, forward, h)),
                        ("schur_to_orbit", coefficient_metrics(n, module.schur_to_orbit(n, h, kernels), a, "orbit")),
                        ("pauli_to_schur", block_metrics(n, transforms.pauli_to_schur(n, p, tables, backend=backend), h)),
                        ("schur_to_pauli", coefficient_metrics(n, transforms.schur_to_pauli(n, h, tables, backend=backend), p, "pauli")),
                        ("orbit_roundtrip", coefficient_metrics(n, module.schur_to_orbit(n, forward, kernels), a, "orbit")))
                    for direction, metrics in calculations:
                        input_family = "central_random_schur_"+family if direction == "schur_to_orbit" else family
                        observations.append(dict(n=n, family=input_family, direction=direction, backend=backend,
                                                 cache="cached" if cached else "uncached",
                                                 reference="literal tensors, traces, singlets and Dicke vectors",
                                                 precision="binary64", **metrics))
    return observations


def kernel_campaign(sizes):
    observations = []
    for n in sizes:
        pairs = {(0, n), (n, 0), (n, n), (n//2, n//2), (n//3, 2*n//3),
                 (2*n//3, n//3), (n//2, n//2+1)}
        for r, s in sorted(pairs):
            if s > n:
                continue
            ts, coefficients = decimal_kernel(n, r, s)
            with localcontext() as ctx:
                ctx.prec = 80
                reference = np.array([[float(value * (Decimal(multiplicities(n)[k])
                                      / Decimal(orbit_weight(n, r, s, t))).sqrt())
                                      for t, value in zip(ts, row)] for k, row in enumerate(coefficients)])
            for dtype in (np.float64, np.longdouble):
                _, actual = schur_hahn.build_U(n, r, s, dtype=dtype)
                # Residual measured in binary64, after wider kernel construction.
                actual = np.asarray(actual, dtype=float)
                error = float(np.linalg.norm(actual-reference, 2))
                residual = float(np.linalg.norm(actual@actual.T-np.eye(len(ts)), 2))
                zero = reference == 0
                observations.append(dict(n=n, family=f"weights_{r}_{s}", direction="kernel",
                                         backend="hahn", cache="not_applicable", dtype=np.dtype(dtype).name,
                                         reference="exact integer Gijswijt sum and Decimal square root",
                                         precision="80 decimal digits; final comparison binary64",
                                         kernel_spectral_error=error, orthogonality_residual=residual,
                                         maximum_zero_entry_error=float(np.max(np.abs(actual[zero]))) if zero.any() else None,
                                         passed=bool(error <= ATOL+RTOL and residual <= ATOL+RTOL)))
    return observations


def larger_campaign(sizes):
    """Full outputs for inputs supported on one weight pair, independently checked.

    This samples the benchmark range; it does not test every dense input at n=40.
    Both random weighted coordinates and the cancellation-prone symmetric
    projector at a central weight are covered. Decimal sums include the actual
    binary64 input exactly; conversion to binary64 occurs only at the end.
    """
    observations = []
    for n in sizes:
        r = s = n//2
        ts, kernel = decimal_kernel(n, r, s)
        rng = np.random.default_rng(SEED+n)
        for family in ("central_random", "central_symmetric_projector"):
            values = ((rng.normal(size=len(ts))+1j*rng.normal(size=len(ts)))
                      / np.array([sqrt(orbit_weight(n, r, s, t)) for t in ts]))
            if family == "central_symmetric_projector":
                values = np.full(len(ts), 1/comb(n, r), dtype=complex)
            a = {(r, s, t): v for t, v in zip(ts, values)}
            h = [np.zeros((n-2*k+1, n-2*k+1), dtype=complex) for k in range(n//2+1)]
            with localcontext() as ctx:
                ctx.prec = 80
                for k, row in enumerate(kernel):
                    real = sum(c*Decimal(float(v.real)) for c, v in zip(row, values))
                    imag = sum(c*Decimal(float(v.imag)) for c, v in zip(row, values))
                    h[k][r-k, s-k] = complex(float(real), float(imag))
                # Independent inverse input: one central entry in each sector.
                inverse_h = [b.copy() for b in h]
                for k, block in enumerate(inverse_h):
                    block[r-k, s-k] = complex(rng.normal(), rng.normal()) / sqrt(multiplicities(n)[k])
                inverse_a = {}
                for j, t in enumerate(ts):
                    real = imag = Decimal(0)
                    for k, block in enumerate(inverse_h):
                        factor = Decimal(multiplicities(n)[k])*kernel[k][j]/Decimal(orbit_weight(n, r, s, t))
                        real += factor*Decimal(float(block[r-k, s-k].real))
                        imag += factor*Decimal(float(block[r-k, s-k].imag))
                    inverse_a[r, s, t] = complex(float(real), float(imag))
            for backend, module in (("hahn", schur_hahn), ("factorial", schur_factorial)):
                for dtype in (np.float64, np.longdouble):
                    prepared = module.prepare(n, dtype=dtype)
                    forward = module.orbit_to_schur(n, a, prepared)
                    for direction, metrics in (
                            ("orbit_to_schur", block_metrics(n, forward, h)),
                            ("schur_to_orbit", coefficient_metrics(n, module.schur_to_orbit(n, inverse_h, prepared), inverse_a, "orbit")),
                            ("orbit_roundtrip", coefficient_metrics(n, module.schur_to_orbit(n, forward, prepared), a, "orbit"))):
                        observations.append(dict(n=n, family=family, direction=direction, backend=backend,
                                                 dtype=np.dtype(dtype).name, cache="cached",
                                                 reference="exact integer Gijswijt sum and Decimal accumulation",
                                                 precision="80 decimal digits; inputs and final comparison binary64",
                                                 **metrics))
    return observations


def metadata():
    root = Path(__file__).resolve().parents[1]
    sources = [root/"pauli_transforms"/name for name in
               ("accuracy.py", "common.py", "krawtchouk.py", "schur_factorial.py", "schur_hahn.py", "transforms.py")]
    return dict(source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(set(sources))},
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                threadpools=threadpool_info(), seed=SEED, absolute_tolerance=ATOL, relative_tolerance=RTOL,
                small_block_relative_threshold=SMALL_BLOCK_RATIO,
                longdouble_precision_bits=int(np.finfo(np.longdouble).nmant+1),
                scope="finite accuracy observations, not comparative timings or stability guarantees",
                hahn_wider_dtype="recurrence is wider, but math.log normalization factors remain binary64",
                output_dtype="complex128; kernel residuals and error metrics evaluated in binary64")


def write_summary(document, output):
    observations = document["observations"]
    groups = [("Dense six directions", lambda r: r["campaign"] == "dense" and r["direction"] not in ("norm_identity", "orbit_roundtrip")),
              ("Dense round trips", lambda r: r["campaign"] == "dense" and r["direction"] == "orbit_roundtrip")]
    for backend in ("hahn", "factorial"):
        for dtype in ("float64", np.dtype(np.longdouble).name):
            groups.append((f"Selected full / {backend} / {dtype}",
                           lambda r, b=backend, d=dtype: r["campaign"] == "larger" and r["backend"] == b and r["dtype"] == d and r["direction"] != "orbit_roundtrip"))
    lines = ["| Cases | Sizes | Maximum relative HS error | Maximum block spectral error | Failed / total |",
             "|---|---|---:|---:|---:|"]
    for label, choose in groups:
        rows = [r for r in observations if choose(r)]
        if not rows:
            continue
        rel = max((r.get("relative_hs_error") or 0 for r in rows), default=0)
        spectral = max((r.get("maximum_block_spectral_error") or 0 for r in rows), default=0)
        lines.append(f"| {label} | {','.join(map(str, sorted({r['n'] for r in rows})))} | {rel:.3e} | {spectral:.3e} | {sum(not r['passed'] for r in rows)} / {len(rows)} |")
    output.write_text("\n".join(lines)+"\n")


def write_tex_table(document, output):
    """Generate the compact thesis table directly from the saved observations."""
    rows = document["observations"]
    def scientific(value):
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    groups = []
    for backend, title in (("krawtchouk", "Krawtchouk"), ("hahn", "Hahn"), ("factorial", "Shared-factor")):
        directions = {"orbit_to_pauli", "pauli_to_orbit"} if backend == "krawtchouk" else {
            "orbit_to_schur", "schur_to_orbit", "pauli_to_schur", "schur_to_pauli"}
        groups.append((title, "$M\\leftrightarrow P$" if backend == "krawtchouk" else "$M,P\\leftrightarrow H$", "$0$--$6$",
                       [r for r in rows if r["campaign"] == "dense" and r["backend"] == backend and r["direction"] in directions]))
    for backend, title in (("hahn", "Hahn"), ("factorial", "Shared-factor")):
        for dtype, suffix in (("float64", "64"), (np.dtype(np.longdouble).name, "ext.")):
            groups.append((f"{title} ({suffix})", "$M\\leftrightarrow H$", "$8,12,20,30,40$",
                           [r for r in rows if r["campaign"] == "larger" and r["backend"] == backend
                            and r["dtype"] == dtype and r["direction"] != "orbit_roundtrip"]))
    lines = ["% Generated by python -m pauli_transforms.accuracy. Do not edit values.",
             "\\begin{tabular}{lllll}", "\\hline", "Backend & Direction & $n$ & Relative HS & Block absolute \\\\", "\\hline"]
    for title, direction, sizes, selected in groups:
        if not selected:
            continue
        rel = max(r.get("relative_hs_error") or 0 for r in selected)
        spectral = max((r.get("maximum_block_spectral_error") or 0 for r in selected))
        block = scientific(spectral) if any("maximum_block_spectral_error" in r for r in selected) else "--"
        lines.append(f"{title} & {direction} & {sizes} & {scientific(rel)} & {block} \\\\")
    lines.extend(["\\hline", "Kernel & Precision & $n$ & $\\|U-U_{\\rm ref}\\|_\\infty$ & $\\|UU^T-I\\|_\\infty$ \\\\", "\\hline"])
    for dtype, title in (("float64", "64"), (np.dtype(np.longdouble).name, "ext.")):
        selected = [r for r in rows if r["campaign"] == "kernel" and r["dtype"] == dtype]
        if selected:
            lines.append("Hahn & "+title+" & $0$--$40$ (selected) & "+scientific(max(r["kernel_spectral_error"] for r in selected))
                         +" & "+scientific(max(r["orthogonality_residual"] for r in selected))+" \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    output.write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path to preserve earlier observations")
    with threadpool_limits(limits=1):
        document = dict(metadata=metadata(), observations=[])
        campaigns = [("dense", dense_campaign(6 if args.sweep else 3)),
                     ("kernel", kernel_campaign([0, 1, 2, 6, 12, 20, 30, 40] if args.sweep else [0, 1, 2, 3]))]
        if args.sweep:
            campaigns.append(("larger", larger_campaign([8, 12, 20, 30, 40])))
        for name, rows in campaigns:
            for row in rows:
                row.update(campaign=name, seed=SEED+row["n"], absolute_tolerance=ATOL, relative_tolerance=RTOL)
                row.setdefault("dtype", "float64")
            document["observations"].extend(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False)+"\n")
    write_summary(document, args.output.with_suffix(".md"))
    write_tex_table(document, args.output.with_suffix(".tex"))
    failed = sum(not r["passed"] for r in document["observations"])
    print(f"Saved {len(document['observations'])} observations, including {failed} failures, to {args.output}")
    # A campaign reports observed limitations. Quick regression tests assert their
    # chosen domain separately, so a wide sweep keeps its complete result file.


if __name__ == "__main__":
    main()
