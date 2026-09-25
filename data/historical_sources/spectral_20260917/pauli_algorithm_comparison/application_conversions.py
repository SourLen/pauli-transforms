"""Fresh conversion setup shared by the physical-application comparisons.

Local implementations build fresh tables for the active input orbit keys.
The optional pinned public implementation instead calls its original complete
block-construction routine separately for every input, without adapter tables.
"""

import hashlib
from pathlib import Path

import numpy as np

from . import anschuetz_optimized, anschuetz_public, schur_separated


DEFAULT_CONVERSION_METHODS = ("thesis", "anschuetz_optimized")
PUBLIC_METHOD = "anschuetz_public_original"
CONVERSION_METHODS = (*DEFAULT_CONVERSION_METHODS, PUBLIC_METHOD)
PUBLIC_N_VALUES = (2, 3, 4, 5)


def public_source_snapshot(cfg):
    """Record the external source separately from locally maintained adapters."""
    if PUBLIC_METHOD not in cfg.get("methods", ()):
        return None
    if not cfg.get("anschuetz_source"):
        raise ValueError("--anschuetz-source is required for anschuetz_public_original")
    path = Path(cfg["anschuetz_source"]).expanduser().resolve()
    if path.is_dir():
        path /= "utils.py"
    return dict(path=str(path), repository=anschuetz_public.REPOSITORY,
                commit=anschuetz_public.COMMIT, expected_sha256=anschuetz_public.UTILS_SHA256,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)


def configure_public_source(cfg):
    """Verify the pin before launching any jobs and serialize its provenance."""
    source = public_source_snapshot(cfg)
    if source is not None:
        if source["sha256"] != source["expected_sha256"]:
            raise ValueError(f"expected utils.py from commit {anschuetz_public.COMMIT}")
        cfg["anschuetz_source"] = source["path"]
        cfg["anschuetz_public_source"] = source
        cfg["anschuetz_public_contract"] = (
            "Pinned construct_matrix_blocks called independently for each input, with all "
            "spin sectors and no prepared get_matrices bank. Loading and hash verification "
            "excluded; input mapping, complete block construction and output wrapping "
            "included in prepare_seconds. The downstream block solver is unchanged.")


def load_public_reference(method, cfg):
    """Load and verify outside each sample timer; never prepare or cache F blocks."""
    if method != PUBLIC_METHOD:
        return None
    if not cfg.get("anschuetz_source"):
        raise ValueError("--anschuetz-source is required for anschuetz_public_original")
    return anschuetz_public.load(cfg["anschuetz_source"])


def prepare(n, method, inputs, cfg, *, reference=None):
    """Return a converter and fresh tables for all inputs in one sample."""
    if method == PUBLIC_METHOD:
        if reference is None:
            raise ValueError("load the pinned public reference before starting the timer")
        # pauli_to_schur's third positional argument is the loaded reference,
        # not an adapter-generated table bank. Its original routine runs afresh.
        return anschuetz_public.pauli_to_schur, reference
    if method == "thesis":
        tables = schur_separated.prepare(
            n, backend=cfg.get("schur_backend", "hahn"),
            dtype=np.dtype(cfg.get("schur_dtype", "float64")))
        return schur_separated.pauli_to_schur, tables
    if method != "anschuetz_optimized":
        raise ValueError(f"Unknown conversion method: {method}")
    keys = dict.fromkeys(key for coefficients in inputs
                         for key, value in coefficients.items() if value)
    return anschuetz_optimized.pauli_to_schur, anschuetz_optimized.prepare(n, keys=keys)


def sizes(cfg, method):
    """A separate grid can respect each implementation's practical limits."""
    if method == "full":
        values = cfg["full_n_values"]
    else:
        values = cfg.get(method + "_n_values")
        if values is None:
            values = PUBLIC_N_VALUES if method == PUBLIC_METHOD else cfg["n_values"]
    return sorted(set(values))
