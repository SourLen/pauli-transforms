"""Anschuetz Appendix-E F formula, evaluated by constrained dynamic programming."""

from __future__ import annotations
from math import comb, factorial, sqrt
from typing import Sequence
import numpy as np
from .common import OrbitKey, all_orbit_keys, pauli_type_from_key, block_shapes, zero_schur_blocks


def _validate_n(n: int) -> None:
    if n < 0:
        raise ValueError('n must be nonnegative')


def _validate_pauli_type(n: int, pauli_type: Sequence[int]) -> tuple[int, int, int, int]:
    if len(pauli_type) != 4:
        raise ValueError('pauli_type must contain (nI, nX, nY, nZ)')
    counts = tuple((int(value) for value in pauli_type))
    if any((value < 0 for value in counts)) or sum(counts) != n:
        raise ValueError(f'Pauli counts must be nonnegative and sum to n={n}')
    return counts


def _local_options(
    count: int, species: str, factorials: Sequence[int]
) -> tuple[tuple[int, int, int, complex], ...]:
    """Return valid local Eq. (71) triples and their scaled weights."""
    options: list[tuple[int, int, int, complex]] = []
    for f_value in range(count // 2 + 1):
        remaining = count - 2 * f_value
        for g0 in range(remaining + 1):
            g1 = remaining - g0
            integer_weight = factorials[count] // (factorials[f_value] * factorials[g0] * factorials[g1])
            if species == 'I':
                a_total = b_total = g0
                phase = 1.0 + 0j
            elif species == 'X':
                a_total, b_total = (g0, g1)
                phase = complex((-1) ** f_value)
            elif species == 'Y':
                a_total, b_total = (g0, g1)
                phase = 1j ** (2 * f_value - g0 + g1)
            elif species == 'Z':
                a_total = b_total = g0
                phase = complex((-1) ** (f_value + g1))
            else:
                raise ValueError(f'unknown Pauli species {species!r}')
            options.append((f_value, a_total, b_total, phase * integer_weight))
    return tuple(options)


def _accumulate_options(current: np.ndarray, options: Sequence[tuple[int, int, int, complex]]) -> np.ndarray:
    """Multiply one sparse local generating polynomial into ``current``."""
    max_f = max((option[0] for option in options))
    max_a = max((option[1] for option in options))
    max_b = max((option[2] for option in options))
    result = np.zeros(
        (current.shape[0] + max_f, current.shape[1] + max_a, current.shape[2] + max_b),
        dtype=np.complex128,
    )
    f_side, a_side, b_side = current.shape
    for f_value, a_total, b_total, weight in options:
        result[f_value:f_value + f_side, a_total:a_total + a_side, b_total:b_total + b_side] += weight * current
    return result


def _coefficient_table(n: int, counts: tuple[int, int, int, int]) -> tuple[np.ndarray, int]:
    """Accumulate the numerator of every valid Eq. (70) component."""
    factorials = tuple((factorial(value) for value in range(n + 1)))
    labelled_options = [
        (species, _local_options(count, species, factorials))
        for species, count in zip(('I', 'X', 'Y', 'Z'), counts, strict=True)
    ]
    labelled_options.sort(key=lambda item: (len(item[1]), item[0]))
    accumulated = np.ones((1, 1, 1), dtype=np.complex128)
    for _, options in labelled_options:
        accumulated = _accumulate_options(accumulated, options)
    padded = np.zeros((n // 2 + 1, n + 1, n + 1), dtype=np.complex128)
    padded[:accumulated.shape[0], :accumulated.shape[1], :accumulated.shape[2]] = accumulated
    count_factorials = 1
    for count in counts:
        count_factorials *= factorials[count]
    return (padded, count_factorials)


def f_blocks_for_pauli_type(n: int, pauli_type: Sequence[int], *, dtype=np.complex128) -> tuple[np.ndarray, ...]:
    """Evaluate one complete Pauli-orbit column of ``F``."""
    _validate_n(n)
    counts = _validate_pauli_type(n, pauli_type)
    output_dtype = np.dtype(dtype)
    if output_dtype.kind != 'c':
        raise ValueError('F blocks require a complex dtype')
    accumulated, count_factorials = _coefficient_table(n, counts)
    blocks: list[np.ndarray] = []
    for k, side in enumerate(block_shapes(n)):
        remaining = n - 2 * k
        scale = float(factorial(k) * factorial(remaining) / count_factorials)
        block = np.empty((side, side), dtype=np.complex128)
        binomials = tuple((comb(remaining, q) for q in range(side)))
        for q_out in range(side):
            a_total = remaining - q_out
            for q_in in range(side):
                b_total = remaining - q_in
                numerator = accumulated[k, a_total, b_total]
                block[q_out, q_in] = scale * numerator / sqrt(binomials[q_out] * binomials[q_in])
        block = block.astype(output_dtype, copy=False)
        block.setflags(write=False)
        blocks.append(block)
    return tuple(blocks)


def f_blocks_for_key(n: int, key: OrbitKey, *, dtype=np.complex128) -> tuple[np.ndarray, ...]:
    """Evaluate one ``F`` column selected by compressed Pauli orbit key."""
    return f_blocks_for_pauli_type(n, pauli_type_from_key(n, key), dtype=dtype)


def prepare(n, keys=None):
    """Build selected F columns, or the complete bank when keys is None."""
    selected = all_orbit_keys(n) if keys is None else dict.fromkeys(keys)
    return {key: f_blocks_for_key(n, key) for key in selected}


def pauli_to_schur(n, coefficients, tables=None):
    """Combine selected F columns; missing tables are built inside this call."""
    if tables is None:
        tables = prepare(n, [key for key, value in coefficients.items() if value])
    blocks = zero_schur_blocks(n)
    for key, value in coefficients.items():
        if value:
            for k, column in enumerate(tables[key]):
                blocks[k] += value * column
    return blocks
