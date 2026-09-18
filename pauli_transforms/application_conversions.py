"""Fresh conversion setup shared by the physical-application comparisons.

Local implementations build fresh tables for the active input orbit keys.
The optional pinned public implementation instead calls its original complete
block-construction routine separately for every input, without adapter tables.
"""

import hashlib
from pathlib import Path

import numpy as np

from . import anschuetz_optimized, anschuetz_public, transforms


DEFAULT_CONVERSION_METHODS = ("thesis", "anschuetz_optimized")
PUBLIC_METHOD = "anschuetz_public_original"
CONVERSION_METHODS = (*DEFAULT_CONVERSION_METHODS, PUBLIC_METHOD)


def configure_public_source(cfg):
    """Verify the optional external file and record its pinned revision."""
    if PUBLIC_METHOD not in cfg.get("methods", ()):
        return
    if not cfg.get("anschuetz_source"):
        raise ValueError("--anschuetz-source is required for anschuetz_public_original")
    path = Path(cfg["anschuetz_source"]).expanduser().resolve()
    if path.is_dir():
        path /= "utils.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != anschuetz_public.UTILS_SHA256:
        raise ValueError(f"expected utils.py from commit {anschuetz_public.COMMIT}")
    cfg["anschuetz_source"] = str(path)
    cfg["anschuetz_public_source"] = dict(repository=anschuetz_public.REPOSITORY,
                                        commit=anschuetz_public.COMMIT, sha256=digest)


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
        return anschuetz_public.pauli_to_schur, reference
    if method == "thesis":
        tables = transforms.prepare(
            n, backend=cfg.get("schur_backend", "hahn"),
            dtype=np.dtype(cfg.get("schur_dtype", "float64")))
        return transforms.pauli_to_schur, tables
    if method != "anschuetz_optimized":
        raise ValueError(f"Unknown conversion method: {method}")
    keys = dict.fromkeys(key for coefficients in inputs
                         for key, value in coefficients.items() if value)
    return anschuetz_optimized.pauli_to_schur, anschuetz_optimized.prepare(n, keys=keys)
