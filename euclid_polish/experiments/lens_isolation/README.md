# Lens-isolation ensemble

This experiment reconstructs only a complete gravitational-lens system:
foreground deflector plus lensed background source. Unrelated galaxies and
stars appear in the dirty input but not in the target. Negative examples are
galaxy-rich scenes with an exactly zero target.

Every artifact is isolated below
`data/experiments/lens_isolation/{records,ensemble,evaluation}`. Existing
production records and ensemble members are read-only sources and are guarded
against output-path overlap.

Run the stages in order:

```bash
python scripts/lens_isolation_generate.py --workers 16
python scripts/lens_isolation_train.py --sources member_01,member_04
python scripts/lens_isolation_evaluate.py
python scripts/lens_isolation_infer.py input_lr.fits --out-dir data/lens_isolation_inference
```

Generation writes aligned `scene_*` and `lens_*` TFRecords; fixed validation
and test splits also receive `dirty_*`. Training always performs a live
full-field PSF/noise/artifact/star forward pass and forks each selected source
into a virgin experiment member with a fresh optimizer at step zero.

The localhost UI exposes the same generate, train, evaluate, status, and sync
controls at `/lens-isolation` and `/app/lens-isolation`. Sync pulls only the
small evaluation report unless records or checkpoints are explicitly enabled.
