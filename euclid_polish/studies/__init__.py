"""Model studies: frozen, whole-ensemble comparisons for the paper.

A study freezes everything needed to compare the ensemble's members — every
active member's recipe and scores, the production gate, per-field
PSNR-vs-knee curves, training curves, the gate diagnostic and the combiner
comparison, the real-tile metrics — into an immutable record that outlives
the members (archiving a member never touches a study), plus up to
:data:`~euclid_polish.studies.store.MAX_FIELDS` fields stored on holylabs and
fetched on demand.

* :mod:`~euclid_polish.studies.store` — the study store (manifest, numbers,
  sidecars, guarded delete, remote locations);
* :mod:`~euclid_polish.studies.candidates` — what a freeze would capture now
  (read-only);
* :mod:`~euclid_polish.studies.numbers` — the numbers files;
* :mod:`~euclid_polish.studies.fields` — packing, verified upload and fetch
  of attached fields;
* :mod:`~euclid_polish.studies.cache` — the local fetched-field cache (LRU,
  disk margin);
* :mod:`~euclid_polish.studies.freeze` — the freeze / resume job;
* :mod:`~euclid_polish.studies.stats` — paired bootstrap and group bands;
* :mod:`~euclid_polish.studies.render` — publication figures + their CSVs.
"""
