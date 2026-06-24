# Morphology space — separate LR and SR+HR PCAs

**Date:** 2026-06-24
**Page:** `/evaluation` → "Morphology space" panel

## Motivation

The Morphology space panel currently fits **one** PCA jointly over all Zoobot
view vectors — `before` (LR), `after` (SR), `hr` (HR) — so LR/SR/HR share a
single coordinate system and the before→after→HR "shift" arrows are meaningful.

That joint fit is dominated by the LR→SR shift and obscures the question we
actually care about: **how does the morphological variance structure differ
between the low-res inputs and the super-resolved + high-res outputs?** A PCA
fit on a subset captures that subset's own variance, so comparing a PCA fit on
LR alone against a PCA fit on SR+HR alone exposes the change in variance
structure directly (we do not care about the absolute shift here).

## Design

Keep all existing plots. Add two more PCA fits, each on its own subset, each
rendered exactly like the existing Joint fit (3D + 2D + per-PC variance label).

### Backend — `euclid_polish/eval/zoobot_morph.py`

`morphology_embedding_payload` keeps the joint `pca` and `mds` fits unchanged
and adds two subset fits to `embeddings`:

- `pca_lr` — `_pca3` fit on rows where `view == "before"` (LR only)
- `pca_srhr` — `_pca3` fit on rows where `view in {"after", "hr"}` (SR fit
  jointly with HR so both land in one space)

Each emits the same `{points, variance_pct}` shape as the existing fits, with
each point carrying its existing `key`/`id`/`view`/`group`/`color`/`plens`.
`points` and the reordered `X` are already row-aligned, so a subset fit is just
index selection + `_pca3` on `X[idx]`. No new variance metric — per-PC
explained-variance % only, matching the joint plot.

### Frontend — `euclid_polish/web/templates/evaluation.html`

- Keep the current Joint (LR+SR+HR) row (PCA·3D + PCA·2D) unchanged.
- Add two labeled rows, each mirroring it (3D + 2D + variance readout):
  - **LR only** (□ markers)
  - **SR+HR** (● SR, ★ HR)
- Reuse `makeSpace3dScene` for the four new canvases. Wire them into
  `updateSpace3d` (`setData` from `embeddings.pca_lr` / `embeddings.pca_srhr`),
  the group + HR-stars filters, `reset`, and `resize`.
- **No arrows** in the new rows (separate coordinate systems → cross-view arrows
  are not defined). The existing arrows toggle keeps driving only the joint
  plot; the new scenes get no edges.

### Tests — `tests/test_eval_catalog.py`

- Extend the embeddings-key assertion to
  `{"pca", "mds", "pca_lr", "pca_srhr"}`.
- Assert `pca_lr` contains only `before` points and `pca_srhr` only
  `after`/`hr` points, each with 3-component `xyz` and a positive leading
  `variance_pct`.

## Out of scope

- No new variance metric (absolute total variance / scree) — per-PC % only.
- No change to the joint fit, MDS, arrows, opacity-by-P(lens), or filters.
