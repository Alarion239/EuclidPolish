# Lens-system isolation experiment

This additive experiment teaches an isolated ensemble to reconstruct each
complete gravitational-lens system — foreground deflector plus lensed source —
while suppressing ordinary galaxies and stars. It does not alter production
generation, records, models, or ensemble members.

All artifacts live only below:

```text
data/experiments/lens_isolation/{records,ensemble,evaluation}
```

Generation uses ordinary, unbiased pure-TNG fields:

```text
sersic_density_arcmin2 = 0
tng_density_arcmin2    = 60
tng_redshift_mode      = true
lens_density_arcmin2   = 20
```

Every normal zero-, one-, and multi-lens outcome is accepted. A scoped adapter
captures each existing lens render once into the clean target; ordinary TNG
galaxies and stars never enter that target. The same normal observation path
then creates the dirty input from the complete scene plus fixed stars.

For every split the generator writes position-aligned normal-format pairs:

```text
dirty_{train,validate,test}.tfrecord
lens_{train,validate,test}.tfrecord
sources_{train,validate,test}.csv
dataset.json
```

The source sidecars are reproducibility and analysis metadata only. They do not
choose training or evaluation crops. Records carry schema/config fingerprints;
incompatible existing experiment artifacts require regeneration with `--force`.

Run the stages in order:

```bash
python scripts/lens_isolation_generate.py --workers 16
python scripts/lens_isolation_train.py --sources member_01,member_04
python scripts/lens_isolation_evaluate.py
```

Training forks each selected production member to a virgin experiment directory,
checks that the source fingerprint is unchanged, and invokes the unchanged
record-mode interface with `forward_onthefly=False`:

```text
lr_path = dirty_train.tfrecord
hr_path = lens_train.tfrecord
```

Normal random block-aligned crops, augmentation, asinh normalization,
optimisation, validation, checkpoint selection, and logs remain owned by
`Model.train`. Evaluation fixes random, block-aligned held-out cutouts first,
then reports aggregate error, target-present reconstruction/flux retention,
zero-target residual output, optional crop ROC/AUC, and an all-zero baseline.

The local UI exposes the same generate, train, evaluate, status, and sync
controls at `/lens-isolation` and `/app/lens-isolation`. Sync defaults to the
small evaluation report; records and checkpoints require explicit opt-in.
