# Synthetic workspace (`/synthetic/*`)

What goes into a synthetic scene, whether each ingredient matches real Euclid
Q1, whether the records are built from the current ingredients, and whether
you can generate (console regrouping, `docs/superpowers/specs/2026-09-27-console-regrouping-design.md`).
It absorbed the old Realism (`/realism/*`) and Data (`/data/*`) workspaces;
`../realism/register.ts` and `../data/register.ts` are one-line re-exports of
`./register.ts` while the shell still imports them.

Every ingredient tab has the same layout, top to bottom: the check against
real data (at most ONE `SummaryLine` per page), then the prior (the `?prior=1`
drawer: fit and activate), then the "How this is produced" drawer (`?how=1`)
with its real reference data and FASRC steps. Drawers are collapsible
sections at the foot of the page (`common.tsx Drawer / DrawerButton`): they
scroll with the page, never cover a figure, and render nothing while closed.
Every job is keyed (`jobs.ts`), confirmed and never started by a visit.
Numbers follow the statistics rule (FOUNDATION §9.2): summary line, facts,
small comparison tables, counts on the controls, captions, provenance-only
`Details`; badges only on a problem.

| Tab | Module | Holds |
|---|---|---|
| Status | `tabs/Status.tsx` (+ `statusModel.ts`) | the synthetic_generate gate ("Ready to generate" / "Blocked by N", the confirmed "Generate validate+test on FASRC" dialog), then "Blocks generation" (Galaxies, Stars, Noise, PSF, TNG radii, Saturation rule, Training catalogue) and "Diagnostic caches" (galaxy plots; field statistics, which also owns a missing or changed real reference) rows: a dot, ONE verdict number with its unit, the "records built with it?" tick, the fix (the single copies of Validate TNG radii and Rebuild field statistics), the tab link and the `readiness` inspector (fingerprints). |
| Records | `tabs/Records.tsx` (+ `recordsModel.ts`) | one toolbar row (split with counts, train disabled; truth overlay Off / HR / All + type chips as legend and filter; a badge only for a corrupt / missing file; the Generate and sync drawer `?gen=1`), the viewer (LR, HR at matched surface brightness, Blurred HR, Clean; the SR tier stays viewable), the record's truth sources, then the census (`?section=census`): Σ VIS and brightest-star histograms (a click opens the record), sources per arcmin² generated · prior · Q1, the per-record table. |
| Galaxies | `tabs/Galaxies.tsx` + `galaxies/` | views distributions (trust boxes, brightness, size, colours, shape; a caption each), relations (radius and FWHM laws, slope/scatter at 2 s.f.), joint (corner with n on hover and forward-noised model colours, `corner.model_noise`; pair explorer; magnitude × radius map) and templates (`galaxies/Templates.tsx`: the TNG50-1 atlas, SFR = 0 on a floor strip, the histogram as the explorer's marginal, the grid thumbnails, ONE radius-manifest line). Prior drawer: the density against the generated fields, the laws as facts, the colour-forest diagnostic, Fit (from the cached Q1 brackets, `POST /api/galaxy-distributions/fit`) and Activate. How drawer: the Q1 MER + PHZ query, the cones, the plot rebuild, the TNG steps. The old model view opens the Prior drawer; the figure view is Figures › Plates. |
| Stars | `tabs/Stars.tsx` + `stars/` | one view: the verdict, the legend with sample sizes, the VIS density panel with the trusted window shaded, six unit-area colour PDFs (the model's colour draws forward noised with the flux errors of VIS-nearest Q1 stars, `model_color_noise`), the caption. Prior drawer: the fit sample sentence, Fit / Activate, the Gaia colour fields collapsed. How drawer: the confirmed MER + PHZ + Gaia query and its result as facts. |
| Noise | `tabs/Noise.tsx` (+ `noiseModel.ts`) | how a scene gets its noise (three sentences), the level histograms (field legend on top), the realised background σ synthetic vs real (the tab's summary line), band pairs (pair switch in the card, 4 × 4 on demand), depth steps, the measured positions on demand, the provenance footer. How drawer: the jitter switch, the MER downloader commands (a local script), vis_noise_sample. |
| PSF | `tabs/Psf.tsx` + `psf/` | the header sentence (stars, usable ones), then catalogue (filters, table, magnitude histogram with the euclid_query windows), cutouts (target marked, catalogue vs whole-cutout magnitude apart, gallery on a per-star stretch `stretch=star`, one stacked validity bar per band) and epsf (what generation uses, the kernel viewer with FWHM and live warps, ePSF vs Gaussian table, the cluster FWHM map). How drawer: the chain's steps, Pull stars.csv and the two ePSF syncs. |
| Fields | `tabs/Fields.tsx` + `fields/` | "Synthetic vs real LR fields": the VIS scale-similarity verdict (a stale cache keeps its last result behind a badge + Measure), views look (`fields/Look.tsx`, two lanes on one locked transfer), stats (six figures; the background vs robust noise figure lives on Noise) and detection (`fields/Stats.tsx`), sample chips with sizes. Real reference drawer (`?ref=1`): the archive fields, their sync, archive_field_sample. |

Shared: `api.ts` / `dataApi.ts` (payload types, resources), `common.tsx`
(load states, info popovers, sky links, the drawers), `dataCommon.tsx`
(Records' compactable toolbar, freshness, the job starter), `jobs.ts`,
`header.tsx` (the include-training toggle, on Galaxies, Stars and Records
only), `chartKit.ts`, `inspectors.tsx` / `dataInspectors.tsx` (+ `register.ts`:
`readiness`, `noisepos`, `archivefield`, `star`, `truth`, `psf`, `tng`),
`realism.css` (`rl-*`), `data.css` (`dt-*`), `synthetic.css` (`syn-*`).

## Old keys the tabs still accept

The redirect rules (`spa_routes.json`) move every old URL to its v2 address.
The tabs also read the absorbed pages' own values once: Galaxies
`?view=model` (opens the Prior drawer) and `?view=figure` (distributions);
Stars `?view=prior|colours|gaia`; Records `?view=census`; PSF cutouts
`?gband=` (the gallery band is `?band=`); Fields `?view=pixels|census|inputs`
(statistics).

## Tests

`synthetic.test.tsx` (Status, the header, Noise, Galaxies, Stars, Fields,
inspectors, the tab list), `recordsPsf.test.tsx` (Records, PSF, the TNG
templates, the data inspector cards), and the pure models:
`statusModel.test.ts`, `recordsModel.test.ts`, `noiseModel.test.ts`,
`psf/psfModel.test.ts`, `galaxies/templates.test.ts`, `dataModel.test.ts`,
`models.test.ts`, `chartKit.test.ts`, `fields/sync.test.ts`,
`archiveFields.test.ts`, `galaxyFwhm.test.ts`.
