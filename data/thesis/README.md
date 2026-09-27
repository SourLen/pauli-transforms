# Thesis data

Saved measurements for the thesis figures. From the repository root:

```bash
python -m pauli_transforms.plot --data data/thesis --output figures
```

The plotter selects the following datasets and exports PDF, SVG, PNG and
unrounded plot statistics. Plots show medians and interquartile ranges of
complete, validated trial groups.

| Figure | Data directory |
| --- | --- |
| General conversion | `current/direct` |
| Fixed locality | `current/fixed_locality` |
| Common orbit-image cache | `current/fixed_weight_cache` |
| Matrix-unit conversion | `matrix_units` |
| Complete spectra | `current/application_baselines/spectral` |
| Random dynamics | `dynamics` |
| Ising illustration | `ising` |
| Ising timing comparison | `current/application_baselines/ising` |

Each campaign contains its configuration, observations, inputs and environment
records. Older campaigns remain available in `direct`, `fixed_locality` and
`spectral`; the latter also supplies the example spectrum. Use `--supplementary`
for the additional plots. The [figure index](../../manuscript/figures.json)
maps outputs to thesis labels.

[manifest.json](manifest.json), [current/selection_manifest.json](current/selection_manifest.json)
and the [application manifest](current/application_baselines/manifest.json)
record source locations, file hashes and metadata redactions. Raw observations
retain their original bytes. These are timings of the
[archived implementations](../historical_sources/README.md); new runs produce
separate measurements.

The comparisons use [Anschuetz et al.](https://doi.org/10.22331/q-2023-11-28-1189),
[Chang, Larocca and Cerezo](https://arxiv.org/abs/2603.13072),
[permqit](https://github.com/bbbergh/permqit/tree/22af3cd245bd0e950df49f6ce16ccd422b6eb2c5)
and [PIQS](https://doi.org/10.1103/PhysRevA.98.063815).
The Schur constructions follow [Gijswijt](https://arxiv.org/abs/0910.4515)
and [Vallentin](https://doi.org/10.1016/j.laa.2008.07.025), as developed in the thesis.
