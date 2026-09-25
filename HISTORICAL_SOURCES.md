# Sources for the retained thesis measurements

The figures retain their original observations. They are not timings of the
current `pauli_transforms` package. The selected implementations and drivers
used for those observations are now archived in
[`data/historical_sources`](data/historical_sources), separately from the
current implementation and the new accuracy campaign.

[`manifest.json`](data/historical_sources/manifest.json) records 86 source files
whose bytes match the SHA-256 values in the original campaign records. Identical
files may occur in several bundles. It also identifies six transitive dependency
files that were omitted from two original records, and eight new inert package
markers. The numerical kernels were not edited during recovery. Development Git
history, correspondence, unrelated research files and the excluded tensor-network
implementation are not included. External comparator source is fetched separately
at the pinned revisions below.

## Dataset and source correspondence

Paths in the first column are relative to `data/thesis`. For the merged direct
figure, retain the distinction between the initial campaign and its extension.

| Retained data | Archived source bundle |
|---|---|
| `direct`, `current/direct/baseline_*` | `direct_20260917` |
| `current/direct/extension_*` | `conversion_20260921` |
| `fixed_locality` (older supplementary observations) | `fixed_locality_20260914` |
| `current/fixed_locality/low`, `current/fixed_locality/high` | `conversion_20260921` |
| `current/fixed_weight_cache/low`, `current/fixed_weight_cache/high` | `fixed_weight_cache_20260923` |
| `spectral` | `spectral_20260917` |
| `dynamics` | `dynamics_20260918` |
| `matrix_units` | `matrix_units_20260919` |
| `ising` | `ising_20260918` |

All selected source hashes in the low- and high-weight September 21–23 records
were checked against their assigned bundles. The older fixed-locality driver and
conversion wrapper were recovered as individual hash-matching source files. No
development repository or history was copied into this archive.

## Verification and execution

Run commands below from the repository root. Verification reads files and does
not perform measurements:

```sh
python data/historical_sources/run.py --verify
```

It verifies all 100 package files and the separately hashed launch adapters.
The archive runner copies a selected package into a temporary directory and
invokes its original driver. Historical generic workers eagerly import algorithms
that were not selected for these figures. A new import adapter supplies rejecting
placeholders for those unused modules. Calling an excluded algorithm raises an
error. The original selected numerical kernels and measured functions remain
unchanged. These launch adapters are identified separately in the manifest.

Historical records identify Python 3.14.7, NumPy 2.4.6 and SciPy 1.17.1.
The drivers also use `threadpoolctl`, `psutil` and Matplotlib. The old Ising
plotter additionally uses SciencePlots. The September 25 smoke runs used Linux,
Python 3.14.7, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.10.9, psutil 7.2.2,
threadpoolctl 3.7.0 and SciencePlots 2.2.2. These are the smoke environment's
versions, not a claim that every dependency was recorded for every old campaign.
The original environment files remain beside the observations. These historical
drivers use Linux process/environment facilities and are not promised portable
to every operating system.

For example, install the replay dependencies in a separate environment:

```sh
python -m pip install numpy==2.4.6 scipy==1.17.1 matplotlib==3.10.9 \
  psutil==7.2.2 threadpoolctl==3.7.0 SciencePlots==2.2.2
```

### A quick smoke run

This reads the retained configuration, reduces it to one small input and one
repetition per selected method, and writes fresh outputs to a new directory:

```sh
python data/historical_sources/rerun.py current/fixed_weight_cache/low \
  --smoke --output /tmp/pauli-historical-cache-smoke
```

The script refuses an existing output directory. Use `--show` to inspect the
constructed command without executing it. This command is a correctness and
execution check, not a replacement for any saved timing campaign.

### Full historical configurations

Remove `--smoke` to use the saved sizes, seeds, repetition counts, preparation
policy, precision and process limits. For example:

```sh
python data/historical_sources/rerun.py current/fixed_locality/high \
  --output /tmp/pauli-historical-fixed-high
python data/historical_sources/rerun.py current/fixed_weight_cache/low \
  --output /tmp/pauli-historical-cache-low
python data/historical_sources/rerun.py current/fixed_weight_cache/high \
  --output /tmp/pauli-historical-cache-high
python data/historical_sources/rerun.py current/direct-extension \
  --output /tmp/pauli-historical-general-extension
python data/historical_sources/rerun.py dynamics \
  --output /tmp/pauli-historical-dynamics
```

Other accepted dataset names are `direct`, `fixed_locality`,
`current/direct-baseline`, `current/fixed_locality/low`, `spectral` and
`matrix_units`. The full configurations were reconstructed and their drivers
were smoke-tested. Full timing campaigns were not rerun during source recovery.
Fresh timings will differ across machines and runs and must be kept separately
from the retained observations.

### External comparators

The original Anschuetz adapter verifies `utils.py` from the following revision:

```sh
git clone https://github.com/bkiani/symmetric_hamiltonians.git /tmp/symmetric-hamiltonians
git -C /tmp/symmetric-hamiltonians checkout 24ce1a5fe4f3234f5f9a2f1ad65c2909423d7dc8
python data/historical_sources/rerun.py current/direct-baseline \
  --anschuetz-source /tmp/symmetric-hamiltonians \
  --output /tmp/pauli-historical-general-baseline
python data/historical_sources/rerun.py spectral \
  --anschuetz-source /tmp/symmetric-hamiltonians \
  --output /tmp/pauli-historical-spectral
```

The required `utils.py` SHA-256 is
`ea99d1dda523facaea60ba2cf46b6c636ecc2465dd660b581d90e59dfc7995cd`.
The original adapter's two in-memory import compatibility changes are preserved:
the NumPy factorial alias is replaced with `math.factorial`, and an unused
SymPy full-matrix helper is disabled. The measured complete block-construction
routine is unchanged.

Use the tested Python 3.14 environment for the matrix-unit driver, with a clean
permqit checkout matching its installed package:

```sh
git clone https://github.com/bbbergh/permqit.git /tmp/permqit-historical
git -C /tmp/permqit-historical checkout 22af3cd245bd0e950df49f6ce16ccd422b6eb2c5
python -m pip install /tmp/permqit-historical
python data/historical_sources/rerun.py matrix_units \
  --permqit-source /tmp/permqit-historical \
  --output /tmp/pauli-historical-matrix-units
```

The driver compares every installed upstream Python source with that checkout
and rejects a modified checkout. Both external comparators were checked at the
pinned revisions in the smoke runs. Their source and licenses remain in their
own repositories and are not silently relicensed by this archive.

## Ising illustration and provenance limits

The Ising figure contains numerical values rather than timing comparisons.
Its recovered plotter and immediate dependencies match the original provenance.
The source record did not list the transitive conversion kernels. The archive
supplies those from the separately recorded September 17 campaign, and records
that distinction. The original spectral campaign similarly omitted the
`schur_factorial.py` import needed for backend type detection. That dependency
is supplied from the September 17 direct record. The spectral timings selected
Hahn evaluation, not factorial evaluation.

This new verification command regenerates the Ising numerical values with the
archived kernels without needing the old timing dataset:

```sh
python data/historical_sources/check_ising.py \
  --output /tmp/pauli-historical-ising-check.json
```

On September 25, all saved spectrum values at n=12 and all 241 curve points for
each n=8,20,40 agreed exactly in the recorded environment. The absolute acceptance
tolerance was 1e-10. The original small dense and collective-spin checks also
passed. Results are in
[`ising_regeneration_check.json`](data/historical_sources/ising_regeneration_check.json).
This is a finite numerical regeneration check, not a stability result.

The other smoke runs covered the original direct driver with both optimized and
pinned public Anschuetz, both fixed-locality driver versions, the common
orbit-image cache driver, spectral and random-dynamics drivers, and all three
matrix-unit methods. Every saved smoke record passed its original validation
gate. Details are in [`smoke_checks.json`](data/historical_sources/smoke_checks.json).
The source archive preserves the measured code and executable rerun paths with
the stated metadata gaps. It does not assert byte-identical timing reproduction
or a clean installation test of every historical dependency environment.
