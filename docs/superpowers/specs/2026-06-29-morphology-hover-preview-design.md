# Morphology hover-preview popup — design

**Date:** 2026-06-29
**Status:** Approved, implementing

## Goal

On the `/evaluation` page, let the user inspect what each point in the
morphology PCA/MDS space looks like by **hovering** it. Hovering a point
already highlights it and shows its name in the per-canvas readout; now it
also pops up a small floating image preview near the cursor, rendered at the
tier that matches the hovered view.

## Behavior

- Hover a point in any of the 6 morphology canvases (3D + 2D, global +
  LR-only + SR+HR) → a small floating cutout preview appears near the cursor.
- The preview shows that object at the tier matching the point's view:
  `before → LR`, `after → SR`, `hr → HR`.
- It is **ephemeral**: it follows the cursor and updates as you move between
  points; it disappears the moment you leave the plot or move off all points.
- The existing name/group readout (`metaEl`) is unchanged.
- The top-of-page main cutout viewer is untouched — this is a separate,
  dedicated preview instance.

## Components

### 1. Shared reusable preview popup (evaluation.html)

A single floating `<div>` mounted **once** (not per hover), containing **one**
`mountCutoutViewer(..., { collection: "evaluation" })` instance with its
toolbar/nav chrome hidden — just the image frame, compact (~220 px).

API:
- `showAt(subdir, view, clientX, clientY)` — jump the preview viewer to the
  object and tier, position the popup near the cursor (offset, flipped near
  screen edges), and reveal it.
- `hide()` — hide the popup.

Mounting once matters: the viewer caches cubes per `tier:index`, so
re-hovering an object is instant and avoids network thrash.

### 2. Cutout viewer API additions (cutout_viewer.js)

- `api.setTiers([key])` — set `state.tiers`, then `syncChips()` + `show()`
  (the same path the tier chip handler uses). Respects disabled/missing tiers.
- A `compact` / `chrome:false` mount option (or a CSS class on the root) that
  suppresses the toolbar and nav so the popup shows only the frame.

`showAt` internally calls `goTo(index)` then `setTiers([tier])`.

### 3. Hover → preview wiring (makeSpace3dScene in evaluation.html)

Where `hover` is computed on `pointermove`:
- When the hovered point's **object or tier changes**, call
  `preview.showAt(subdir, view, x, y)`.
  - `subdir`/`view` parsed from the point `id` by splitting on the **last**
    `__` (ids look like `<object_id>__<view>`).
  - `subdir → viewer index` via the existing `window.__evalSubIndex`.
  - `view → tier` via `{ before: "LR", after: "SR", hr: "HR" }`.
- When the cursor merely moves within the same point, just reposition the
  popup (no refetch).
- On `pointerleave`, or when `nearest()` returns null, call `preview.hide()`.

## Edge cases

- Only the streamed cube is updated when object/tier actually changes, so
  dragging the cursor across one marker does not refetch.
- If an object isn't in `window.__evalSubIndex` (shouldn't happen), the
  preview stays hidden rather than erroring.
- Real objects (A/B/C/gal) have no HR — but they also have no `hr` points in
  the plot, so that view/tier combination never arises.

## Out of scope

- No click-to-pin, no interaction with the preview's own controls (ephemeral
  by design).
- No backend changes; reuses existing `/viewer/meta/evaluation` and
  `/viewer/cube/evaluation/<i>` endpoints.
- No new charting library; the canvases are already interactive.
