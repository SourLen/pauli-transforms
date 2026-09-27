# Replaying the original experiments

These sources accompany the saved thesis measurements. Run commands from the
repository root, using a separate environment for the historical dependencies:

```bash
python -m pip install -r data/accuracy/requirements.txt psutil==7.2.2 SciencePlots==2.2.2
python data/historical_sources/run.py --verify
```

The recorded environment uses Linux and Python 3.14.7.
[manifest.json](manifest.json) maps datasets to sources and hashes, including
supplemental dependencies and launch adapters. Verification checks the
100 archived package files and the adapters; numerical kernels retain their
original bytes.

## Replay a saved configuration

```bash
python data/historical_sources/rerun.py current/fixed_weight_cache/low \
  --smoke --output results/replay/cache-low
```

Remove `--smoke` to use the full saved configuration. `--show` prints the
command without running it. Output directories must be new; runtimes depend
on the machine and environment.

Other dataset names are `direct`, `fixed_locality`, `current/direct-baseline`,
`current/direct-extension`, `current/fixed_locality/low`,
`current/fixed_locality/high`, `current/fixed_weight_cache/high`, `spectral`,
`dynamics` and `matrix_units`.

To regenerate and check the Ising illustration's numerical values:

```bash
python data/historical_sources/check_ising.py --output results/replay/ising-check.json
```

## External comparisons

For the original public Anschuetz implementation:

```bash
git clone https://github.com/bkiani/symmetric_hamiltonians.git _external/symmetric_hamiltonians
git -C _external/symmetric_hamiltonians checkout 24ce1a5fe4f3234f5f9a2f1ad65c2909423d7dc8
python data/historical_sources/rerun.py current/direct-baseline \
  --anschuetz-source _external/symmetric_hamiltonians --output results/replay/direct
```

The current `pauli_transforms.benchmark --comparison anschuetz` command also
accepts `--anschuetz-source _external/symmetric_hamiltonians`.

For native permqit, use Python 3.14 or newer and a clean pinned checkout:

```bash
git clone https://github.com/bbbergh/permqit.git _external/permqit
git -C _external/permqit checkout 22af3cd245bd0e950df49f6ce16ccd422b6eb2c5
python -m pip install ./_external/permqit
python data/historical_sources/rerun.py matrix_units \
  --permqit-source _external/permqit --output results/replay/matrix_units
```

The current runner is `python -m pauli_transforms.run_permqit_comparison
--output results/new/matrix_units`. It uses the same checkout and includes
permqit by default. Both adapters verify the external sources against their
pinned revisions; external code retains its own license.

## Spectral and Ising comparisons from 26 September

These measured drivers use package revision
`29fd29769c62fccda90a1ac3031adbe462510cd5`. Reconstruct their layout in a new
directory, then run them with the recorded dependencies above and QuTiP:

```bash
python -m pip install qutip==5.3.1
mkdir -p results/replay/application_baselines/source_snapshot
git archive 29fd29769c62fccda90a1ac3031adbe462510cd5 pauli_transforms LICENSE | tar -x -C results/replay/application_baselines/source_snapshot
cp data/historical_sources/application_baselines_20260926/*.py results/replay/application_baselines/
cp -r data/thesis/current/application_baselines/spectral/inputs results/replay/application_baselines/archived_inputs
cp data/thesis/current/application_baselines/spectral/spectrum_example.npz results/replay/application_baselines/
python results/replay/application_baselines/run_comparisons.py spectral --output results/replay/application_baselines/spectral
python results/replay/application_baselines/run_comparisons.py ising --output results/replay/application_baselines/ising
python results/replay/application_baselines/plot_comparisons.py \
  --spectral results/replay/application_baselines/spectral \
  --ising results/replay/application_baselines/ising --output results/replay/application_baselines/figures
```

Add `--sizes 2 3 --instances 1 --repeats 1` to the measurement commands for
a small check. Their source hashes are recorded in the
[application data manifest](../thesis/current/application_baselines/manifest.json).
