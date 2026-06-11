/* FasrcStepCard — reusable per-step submission card.
 *
 * Renders one (or many) FASRC pipeline step into a form with three
 * fieldsets — Task parameters / Previous runs / Resources — wires the
 * submit + confirmation + history-prefill flow, and hands the live
 * status display off to :file:`static/job_status.js` after a successful
 * submit.
 *
 * Two entry points, both keyed by the step id used in
 * ``/api/fasrc/hst/<id>/submit``:
 *
 *   FasrcStepCard.renderMany(containerEl, steps, artifactStatus, opts)
 *     — used by fasrc.html, which lists every registered step.
 *
 *   FasrcStepCard.mountOne(containerEl, stepId, opts)
 *     — used by single-step pages (cutouts, PSFs, …). Fetches the
 *       step metadata via /api/fasrc/hst/status, then renders + wires
 *       just that one card.
 *
 * ``opts.onSubmitted(jobid, payload)`` fires after a successful POST
 * to /submit; pages use it to refresh their inspection UI once a job
 * is on the cluster.
 *
 * The submit flow has three defences in this exact order:
 *   1. Form-level ``Enter`` keypresses are blocked (stops the browser's
 *      default "Enter in a number field submits the form" behavior
 *      that turned autofilled values into accidental SLURM jobs).
 *   2. A blocking ``confirm()`` dialog shows the full payload before
 *      anything reaches FASRC.
 *   3. The ``confirm=yes`` token is only set AFTER OK is clicked, so
 *      a programmatic POST from cached JS / a browser extension can't
 *      smuggle a job through.
 */

(function (window) {
  "use strict";

  const RESOURCE_FIELDS = ['partition', 'n_cpus', 'n_gpus', 'memory', 'time_limit'];

  function escapeHtml(s) {
    // Defer to the global helper if present; otherwise minimal fallback.
    if (window.polishUI && typeof window.polishUI.escapeHtml === 'function') {
      return window.polishUI.escapeHtml(String(s ?? ''));
    }
    return String(s ?? '').replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;',
      '"': '&quot;', "'": '&#39;',
    }[ch]));
  }

  function fmtRuntime(s) {
    if (s == null) return '—';
    if (s < 60)    return `${s.toFixed(0)} s`;
    if (s < 3600)  return `${(s / 60).toFixed(1)} min`;
    return `${(s / 3600).toFixed(2)} h`;
  }

  function fmtIsoLocal(iso) {
    if (!iso) return '';
    return iso.replace('T', ' ').replace('Z', '').slice(0, 16);
  }

  // ── Per-step task-fields markup ────────────────────────────────────
  //
  // Centralised so adding a new task page is one switch-case change
  // here, not a new copy of the HTML per page. (Long-term we may move
  // this to a server-supplied schema on /api/fasrc/hst/status; for
  // Phase 1 of the per-page-task refactor, the JS-side dictionary is
  // the cheapest DRY win.)

  function taskFields(step) {
    switch (step.step_id) {
      case 'download':
        return `
          <label>Number of HLSP tiles
            <input type="number" name="n_tiles" value="25" min="1" max="81"></label>`;
      case 'extract_psf':
        return `
          <label>Target N stars
            <input type="number" name="n_stars" value="200" min="20" max="5000"></label>
          <label>PSF half-side (px)
            <input type="number" name="half_side" value="255" min="31" max="767"
                   title="Half-side of the FINAL ePSF → spans (2·half+1) px at the ~0.05″/pix HLSP scale: 255 → 511², 511 → 1023². Always odd, so the PSF stays centred. Changing it forces a full tile re-scan (cached stamps are size-specific)."></label>
          <label>Extraction margin (frac)
            <input type="number" name="extract_margin_frac" value="0.08" step="0.01" min="0" max="0.25"
                   title="Extract star stamps this fraction larger than the half-side, then trim the extra border off the built ePSF. EPSFBuilder's smoothing leaves edge artifacts on the outermost pixels; this margin pushes them into the trimmed region so the final PSF borders are clean. 0.08 = 8%; 0 disables."></label>
          <span class="js-extract-mem muted" style="flex-basis:100%; font-size:12px;"
                title="EPSFBuilder holds several float64 copies of every star cutout, so peak RAM ≈ n_stars · (2·half+1)² · ~60 B. Set the Memory field at or above this."></span>`;
      case 'kernel':
        return `
          <label>Wiener regularisation
            <input type="number" name="regularisation" value="0.001" step="any" min="0"></label>`;
      case 'tfrecords':
        return _tfrecordsFields();
      case 'euclid_sky_download':
        // Field names mirror the EuclidSkyDownloadStep.build_command
        // params (n_positions, vis_pixels, ra_centre, dec_centre,
        // radius_deg). RA/Dec defaults point at EDF-N (the field with
        // Q1 coverage); the script uniformly samples ``n_positions``
        // points inside the cos-Dec-corrected disk, then pulls one
        // 4-band cutout per surviving position from the Euclid SAS.
        return `
          <label>N positions
            <input type="number" name="n_positions" value="200" min="1" max="5000"
                   title="How many random sky positions to draw inside the disk. Positions outside Euclid coverage are silently dropped at mosaic-lookup time, so the final cutout count is typically &lt; this."></label>
          <label>VIS cutout (px)
            <input type="number" name="vis_pixels" value="512" min="64" max="4096" step="32"
                   title="Side length of the VIS cutout in 0.10″/pix Euclid pixels. NIR cutouts are sized to match the same angular footprint."></label>
          <label>RA centre (deg)
            <input type="number" name="ra_centre" value="270.0" step="any" min="0" max="360"
                   title="Disk centre in ICRS degrees. Default 270° = EDF-N (matches Euclid Q1 coverage)."></label>
          <label>Dec centre (deg)
            <input type="number" name="dec_centre" value="66.0" step="any" min="-90" max="90"
                   title="Default +66° = EDF-N."></label>
          <label>Radius (deg)
            <input type="number" name="radius_deg" value="2.0" step="any" min="0.01" max="10"
                   title="Disk radius around (RA, Dec). 2° covers most of the contiguous EDF-N tile grid."></label>`;
      case 'euclid_roundtrip_tfrecords':
        return `
          <label>Max records
            <input type="number" name="max_records" value="1000" min="10" max="50000"
                   title="LR-only TFRecords pulled from the sky cutouts. Each
record stores VIS + Y_E + J_E + H_E."></label>
          <label>Stamp size
            <input type="number" name="stamp_size" value="128" min="32" max="512"
                   title="LR side (0.10\"/pix) of each training stamp chopped out of
a large cutout. Each large cutout yields (vis_pixels / stamp_size)² stamps."></label>
          <label>Validate fraction
            <input type="number" name="valid_fraction" value="0.1" step="0.05" min="0" max="0.5"
                   title="Split at the position level so stamps from one
large cutout don't leak across train/validate."></label>`;
      case 'download_euclid_cutouts':
        // Mirrors EuclidCutoutDownloadStep.build_command (vis_pixels,
        // workers). One shared angular field; each band fetches its own
        // native pixel count for the same footprint.
        return `
          <label>Parallel workers
            <input type="number" name="workers" value="8" min="1" max="32"
                   title="Concurrent per-band downloads from the Euclid archive."></label>
          <p class="hint" style="flex-basis:100%;">VIS cutout (px) is set on the
             <a href="/config">⚙️ Config</a> tab and sent with this job.</p>`;
      case 'extract_euclid_psf':
        // Mirrors EuclidPSFExtractStep.build_command (stars_per_psf,
        // vis_pixels, num_stars cap, output_size). Extracts ALL four bands
        // in one job; each band's good stars are spatially clustered into
        // groups of 'stars per PSF' → one ePSF per cluster.
        return `
          <label>Stars per PSF (N) <span class="muted">(average per cluster)</span>
            <input type="number" name="stars_per_psf" value="100" min="10" max="2000"
                   title="Target average stars per spatial cluster. K = round(n_good / N) clusters via K-Means++ on sky position, one ePSF per cluster (the PSF varies across the field). 3000 good stars at N=100 → ~30 PSFs."></label>
          <label>Minimum stars per cluster
            <input type="number" name="min_stars_per_psf" value="50" min="1" max="2000"
                   title="Clusters smaller than this are merged into their nearest neighbour, so no ePSF is ever built from fewer than this many stars (avoids noisy, under-sampled PSFs)."></label>
          <label>Max stars per band <span class="muted">(blank = all)</span>
            <input type="number" name="num_stars" value="" min="10" max="100000"
                   placeholder="all"
                   title="Optional cap on stars considered per band before clustering. Blank/0 = use ALL good cutouts (recommended — that's how you get many PSFs)."></label>
          <p class="hint" style="flex-basis:100%;">VIS cutout (px) is set on the
             <a href="/config">⚙️ Config</a> tab; output ePSF size is locked to
             2·(VIS cutout)+1 oversampled px.</p>`;
      case 'download_tng_skirt':
        // Mirrors TngSkirtAtlasDownloadStep.build_command (workers, limit,
        // keep_archive). Downloads the whole TNG50 SKIRT atlas (~1153
        // galaxies) as dusty Euclid VIS+NISP FITS, 5 orientations each, in a
        // thread pool. The TNG API token is read on the node from
        // $TNG_API_KEY or ~/.tng_api_key — never sent through this form.
        return `
          <label>Parallel workers <span class="muted">(blank = 1 per CPU)</span>
            <input type="number" name="workers" value="" min="1" max="256"
                   placeholder="= CPUs"
                   title="Concurrent galaxy downloads in flight. Blank = one per allocated CPU. Downloads are network-I/O bound, so a single CPU can drive many transfers — set this ABOVE the CPU count (e.g. 64) to saturate the network without allocating more cores."></label>
          <label>Executor
            <select name="executor"
                    title="'process' (default) runs each galaxy's download+extract in its own process → true multi-core, no GIL contention (threads cap extraction at ~2 cores and starve each other's download loops). 'thread' is lighter; pick it only if a run shows you're purely network-bound.">
              <option value="process">process (true multi-core)</option>
              <option value="thread">thread (lighter, I/O-only)</option>
            </select></label>
          <label>Limit galaxies <span class="muted">(blank = all ~1153)</span>
            <input type="number" name="limit" value="" min="1" max="2000"
                   placeholder="all"
                   title="Only download the first N atlas entries. Blank/0 = the full ~1153-galaxy atlas. Use a small value (e.g. 5) for a smoke test before committing the full multi-hour run."></label>
          <label class="checkbox-field"
                 title="Keep each galaxy's source .tar.gz next to its extracted FITS. Default OFF — the archive is deleted after the ~20 Euclid frames are extracted, so transient disk stays bounded by (workers × tarball size) instead of the full multi-TB atlas.">
            <input type="checkbox" name="keep_archive" value="1">
            Keep source .tar.gz archives</label>
          <span class="muted" style="flex-basis:100%; font-size:12px;">
            Workers run in parallel (process pool by default; blank = one per
            allocated CPU). The job logs live MB/s + ETA and an "effective
            concurrency" factor so you can see if it's network- or CPU-bound.
            Set your TNG API token in the field above first. Finished galaxies
            get a <code>.done</code> marker, so re-submitting only fills the
            gaps.</span>`;
      case 'tng_grid':
        // Mirrors TngGridStep.build_command (band, downsample, mode,
        // temperature). The 5 galaxies are selected locally from the histogram
        // property cache; the most/least modes need histograms rendered first.
        return `
          <label>Band
            <select name="band">
              <option value="VIS">VIS</option>
              <option value="Y">NISP Y</option>
              <option value="J">NISP J</option>
              <option value="H">NISP H</option>
              <option value="RGB">RGB (VIS+NISP, Lupton)</option>
            </select></label>
          <label>Downsample
            <select name="downsample"
                    title="×1 = 1600², ×2 = 800², ×4 = 400² (block-mean).">
              <option value="1">×1</option>
              <option value="2">×2</option>
              <option value="4">×4</option>
            </select></label>
          ${_tngModeFields()}`;
      case 'tng_stack':
        // Mirrors TngStackStep.build_command (band, galaxy_id, mode,
        // temperature). Explicit id wins; otherwise the galaxy is picked by mode.
        return `
          <label>Galaxy id <span class="muted">(blank = pick by Mode)</span>
            <input type="text" name="galaxy_id" value="" placeholder="by Mode"
                   inputmode="numeric"
                   title="Subhalo id (folder name under tng_skirt). Blank → choose by the Mode below."></label>
          <label>Band
            <select name="band">
              <option value="VIS">VIS</option>
              <option value="Y">NISP Y</option>
              <option value="J">NISP J</option>
              <option value="H">NISP H</option>
            </select></label>
          ${_tngModeFields()}`;
      case 'poster_cutout':
        // Mirrors PosterCutoutStep.build_command (mode, image_size, seed).
        // The Mode toggle picks which random object to generate; a blank seed
        // re-rolls a fresh random object each submit.
        return `
          <label>Mode
            <select name="mode"
                    title="Which random object to render as a clean 4-band cutout. Sérsic = analytic bulge+disk galaxy; Star = point source; Lens = SIE+shear gravitational lens; TNG = real TNG50 SKIRT galaxy (needs the atlas downloaded).">
              <option value="sersic">Sérsic galaxy</option>
              <option value="star">Star (point source)</option>
              <option value="lens">Gravitational lens</option>
              <option value="tng">TNG50 galaxy</option>
              <option value="field">Random field (clean + dirty + SR)</option>
            </select></label>
          <label>HR image size (px) <span class="muted">(blank = 256)</span>
            <input type="number" name="image_size" value="" min="32" max="1024" step="2"
                   placeholder="256"
                   title="HR field side in 0.05″/pix pixels. Blank = the 256² default."></label>
          <label>Seed <span class="muted">(blank = random)</span>
            <input type="number" name="seed" value="" min="0"
                   placeholder="random"
                   title="RNG seed for a reproducible object. Blank → a fresh random object each submit."></label>`;
      case 'euclid_query':
        // Mirrors EuclidQueryStep.build_command (num_stars, mag window,
        // snr_min). Writes stars.csv. Run first, then download, then verify.
        return `
          <label>Number of stars
            <input type="number" name="num_stars" value="3000" min="1" max="100000"
                   title="Server-side TOP N by PSF flux (flux_vis_psf) within the magnitude window."></label>
          <label>Min magnitude <span class="muted">(brighter rejected)</span>
            <input type="number" name="magnitude_min" step="0.1" placeholder="e.g. 16"
                   title="Bright-end cutoff (AB mag from flux_vis_psf µJy). Skips stars bright enough to saturate."></label>
          <label>Max magnitude <span class="muted">(dimmer rejected)</span>
            <input type="number" name="magnitude_limit" step="0.1" placeholder="e.g. 19"
                   title="Faint-end cutoff (AB mag)."></label>
          <label>Min SNR <span class="muted">(blank = off)</span>
            <input type="number" name="snr_min" step="1" placeholder="e.g. 50"
                   title="Keep only well-measured stars: flux_vis_psf / fluxerr_vis_psf ≥ this."></label>`;
      case 'euclid_verify_photometry':
        // Mirrors EuclidVerifyPhotometryStep.build_command (n, size).
        // Read-only scale check — run AFTER the cutout download.
        return `
          <label>N stars
            <input type="number" name="n" value="40" min="1" max="1000"
                   title="How many stars to aperture-measure for the median measured/catalog ratio."></label>
          <label>Cutout size (px, as downloaded)
            <input type="number" name="size" value="256" min="32" max="4096" step="32"
                   title="Cutout side to read — must match the size the cutouts were downloaded at (reads star_NNNN_<size>.fits)."></label>`;
      case 'euclid_star_anchor_tfrecords':
        // Mirrors EuclidStarAnchorTFRecordStep.build_command (size, stamp,
        // valid_every, snr_min, limit). Assembles the downloaded star
        // cutouts + catalog PSF flux → (dirty_anchor, hr_anchor) single-
        // pixel delta-target pairs. Operator-free (no PSF).
        return `
          <label>Cutout size (px, as downloaded)
            <input type="number" name="size" value="256" min="32" max="4096" step="32"
                   title="Side of the per-star cutouts to read — must match the size the cutout-download job used. Stars are read from star_NNNN_<size>.fits, so a mismatch finds no files."></label>
          <label>LR stamp (px)
            <input type="number" name="stamp" value="128" min="64" max="2048"
                   title="Star-centred LR stamp side written to each record. Must be ≥ the training LR crop (HR_CROP/2) so the random training crop always keeps the star in-frame."></label>
          <label>Validate every Nth
            <input type="number" name="valid_every" value="10" min="2" max="100"
                   title="Every Nth usable star goes to the validate split; the rest train."></label>
          <label>Min SNR <span class="muted">(blank = off)</span>
            <input type="number" name="snr_min" value="" step="1" min="0" max="1000"
                   placeholder="e.g. 50"
                   title="Skip stars whose PSF flux_psf_uJy / fluxerr_psf_uJy is below this (keep only well-measured flux). Blank/0 = keep all."></label>
          <label>Limit stars <span class="muted">(blank = all)</span>
            <input type="number" name="limit" value="" min="1" max="100000"
                   placeholder="all"
                   title="Only process the first N catalog rows (debugging). Blank = all."></label>`;
      case 'synthetic_generate':
        // Mirrors run_pipeline.py --skip-train (RunPipelineStep.build_command
        // reads n_train / n_valid / image_size). Renders synthetic clean HR
        // scenes + forward-models to dirty Euclid LR.
        return `
          <p class="hint" style="flex-basis:100%;">Train scenes, Validate scenes and
             HR image size are set on the <a href="/config">⚙️ Config</a> tab and sent
             with this job.</p>
          <label style="flex-basis:100%;">TNG galaxy fraction
            <output class="muted" style="margin-left:6px;">1.00</output>
            <span class="muted">(1 = pure TNG with redshift realism; &lt;1 mixes in Sérsic)</span>
            <input type="range" name="tng_fraction" min="0" max="1" step="0.05" value="1"
                   style="width:100%;"
                   oninput="this.parentNode.querySelector('output').value=(+this.value).toFixed(2)"
                   title="Proportion of galaxies drawn as real TNG50 SKIRT stamps instead of analytic Sérsic profiles. 1 = pure-TNG mode: every stamp gets a redshift draw (D_A sizing + dimming + spectral drift), lenses take σ_v from the subhalo mass, and the COSMOS catalog is skipped. <1 mixes in COSMOS Sérsic profiles (legacy path)."></label>`;
      case 'train':
        return _hstTrainFields();
      default:
        return '';
    }
  }

  function _tngModeFields() {
    // Shared galaxy-selection controls for the TNG grid + stack cards.
    return `
      <label>Mode
        <select name="mode"
                title="Which galaxies to show. Random samples uniformly; the most/least modes rank by stellar mass / SFR / effective radius from the histogram property cache (render the histograms first). Re-submit to re-roll.">
          <option value="random">Random</option>
          <option value="most_massive">Most massive (stellar)</option>
          <option value="least_massive">Least massive (stellar)</option>
          <option value="most_star_forming">Most star-forming</option>
          <option value="least_star_forming">Least star-forming</option>
          <option value="biggest_radius">Biggest radius</option>
          <option value="smallest_radius">Smallest radius</option>
        </select></label>
      <label>Temperature <span class="muted">(0 = exact extreme)</span>
        <input type="number" name="temperature" value="0.3" min="0" max="1" step="0.05"
               title="For the most/least modes: 0 picks the deterministic extreme; higher temperatures randomly sample from a broader set of the top galaxies, weighted toward the extreme (not a fixed top-N). Ignored for Random."></label>`;
  }

  function _tfrecordsFields() {
    // Analytic-A forward model is single-stamp, CPU-only — no GPU, no
    // batch_size knob. Each HLSP mosaic is diced into a grid of
    // ``image_size``² HR chunks; coverage + bright + star filters drop
    // chunks until ``n_train`` + ``n_valid`` pairs are written.
    return `
      <label>Train scenes
        <input type="number" name="n_train" value="2000" min="100" max="50000"></label>
      <label>Validate scenes
        <input type="number" name="n_valid" value="200" min="20" max="5000"></label>
      <label>Image size
        <input type="number" name="image_size" value="256" step="2" min="64" max="2048"
               title="HR chunk side in 0.05″/pix pixels. Mosaics are diced into a non-overlapping grid of these squares; smaller → more chunks per mosaic."></label>
      <label>Max relative noise (k)
        <input type="number" name="max_relative_noise" value="5.0" step="0.5" min="0.5" max="50"
               title="Reject a chunk when √S_max·|A|_peak > k·σ_LR — i.e. when the brightest HR pixel's Poisson noise propagated through A would exceed k× the per-pixel Euclid LR noise floor. Smaller k → stricter rejection; larger k → more chunks kept. Default 5."></label>
      <label>Star reject (σ)
        <input type="number" name="star_threshold_sigma" value="20" step="5" min="0" max="100"
               title="Reject a chunk if DAOStarFinder finds a point source brighter than this many σ above the chunk background. Stars forward-model to unlearnable A(ε) ringing; sharpness/roundness cuts spare resolved galaxies. With grid tiling, rejected chunks just advance to the next cell, so be aggressive — lower (10–15) chases fainter stars; 0 disables. Default 20 catches every star bright enough to ring."></label>
      <label>Min source (σ)
        <input type="number" name="min_source_sigma" value="5" step="1" min="0" max="50"
               title="Reject a chunk as empty unless a few pixels exceed this many σ above its background. COSMOS is full of blank sky that clears the coverage check (noise is non-zero) but holds no object — a useless noise→noise pair. Default 5 (standard detection floor); raise to demand brighter objects; 0 disables."></label>`;
  }

  function _hstTrainFields() {
    return `
      <label>Training steps
        <input type="number" name="steps" value="400000" min="1000" max="2000000"></label>
      <label># synthetic / batch
        <input type="number" name="n_syn" value="4" min="1" max="256"
               title="Synthetic examples per batch (|SR − scene|). Always present; must be ≥ 1. The batch is the fixed layout [n_syn | n_hst | n_anchor] and its size is the sum. At HR-crop 192/LR-crop 96, 4 keeps activation memory ~constant vs the old 96-crop; on an A100 you can go much higher (e.g. 24) to raise GPU utilisation — watch the live GPU gauge."></label>
      <label># HST / batch
        <input type="number" name="n_hst" value="0" min="0" max="256"
               title="HST examples per batch — the SR=sky lane |asinh(H⊛SR) − HST_image|. 0 disables it. Needs the HST records (records_v2_hst) + the F814W ePSF. A whole integer, so no rounding surprises: e.g. 1 HST per 4 synthetic = 20% HST."></label>
      <label># star-anchor / batch
        <input type="number" name="n_anchor" value="0" min="0" max="256"
               title="Star-anchor examples per batch — operator-free masked |SR − delta| at the catalog star pixel (pins real Euclid stars to points of known flux; no PSF). 0 disables it. Needs the star-anchor records (records_v2_star_anchor)."></label>
      <label>Learning rate
        <input type="number" name="learning_rate" value="" step="0.0001" min="0" max="0.01"
               placeholder="blank = 1e-3→5e-4 schedule"
               title="Constant Adam LR for the whole run (e.g. 0.001). Leave blank for the default two-phase schedule 1e-3 → 5e-4 at steps//2. Constant is simpler for an exploratory restart; the decay gives a slightly sharper final result. Scale up with a larger batch."></label>
      <label>SR non-negativity λ
        <input type="number" name="nonneg_sr_weight" value="" step="0.5" min="0" max="100"
               placeholder="blank = config default"
               title="Weight of the SR non-negativity penalty λ·mean(relu(-SR)) added to the loss. SR is the model's single sky output, so this constrains all three lanes toward physical (≥0) flux at once. Blank = the config default (1.0); 0 disables. Tune by watching the negative-pixel fraction — raise if negatives persist, lower if it flattens faint structure."></label>
      <label>Loss weight · synthetic
        <input type="number" name="synthetic_loss_weight" value="1" step="0.5" min="0" max="100"
               title="Per-example TRAINING-loss multiplier for synthetic records. 1 = default; raise to up-weight that source's gradient, 0 to ablate (kept in the batch mix but zero gradient)."></label>
      <label>Loss weight · HST
        <input type="number" name="hst_loss_weight" value="1" step="0.5" min="0" max="100"
               title="Per-example TRAINING-loss multiplier for HST records. 1 = default. Distinct from the data fraction: fraction sets how many HST examples per batch, this scales each one's loss."></label>
      <label>Loss weight · star-anchor
        <input type="number" name="star_anchor_loss_weight" value="1" step="0.5" min="0" max="100"
               title="Per-example TRAINING-loss multiplier for the star-anchor lane (masked |SR − delta| at the star pixel). 1 = default; raise to up-weight the anchor, 0 to ablate (anchor data still loaded, loss contribution zeroed). Only used when #star-anchor > 0."></label>
      <label>Forward-op PSF crop (½)
        <input type="number" name="forward_op_crop_half" value="0" step="8" min="0" max="512"
               title="Optional central crop of the F814W PSF for the HST forward op → (2·crop+1)² kernel. 0 = full PSF (the forward op convolves via FFT, so the full PSF is exact AND fast — no crop needed). Set >0 only to ablate the PSF wings. Only used when #HST > 0."></label>
      <label>Save-best w·synthetic
        <input type="number" name="save_best_w_syn" value="1.0" step="0.5" min="0" max="100"
               title="Weight of synthetic PSNR (dB) in the composite save-best score."></label>
      <label>Save-best w·HST
        <input type="number" name="save_best_w_hst" value="1.0" step="0.5" min="0" max="100"
               title="Weight of HST PSNR (dB) in the composite save-best score. No effect if no HST validate split exists."></label>
      <label>Save-best w·star-anchor
        <input type="number" name="save_best_w_anchor" value="0" step="0.5" min="0" max="100"
               title="Weight of the star-anchor PSNR (dB, ADDED — higher is better) in the composite save-best score. Masked to the star pixel; on the same dB scale as the other two PSNRs. Default 0 = monitored only."></label>
      <label class="checkbox-field" style="flex-basis:100%;"
             title="On a resumed run: UNCHECKED (default) validates the restored checkpoint and uses its score as the bar to beat (no save until genuinely beaten). CHECKED ignores the previous best and lets this run overwrite it on its first eval — use after an architecture change (e.g. new output head / lane counts), when the old score is meaningless. No effect on a fresh run.">
        <input type="checkbox" name="overwrite_best" value="1">
        Overwrite previous best (skip resume baseline)</label>
      <label class="checkbox-field" style="flex-basis:100%;"
             title="Feed the model VIS only (1 input channel) instead of the full VIS+NISP stack (4 channels). The TFRecords keep all 4 bands; the loader slices to VIS in-graph and the model is built with 1 input channel. Checkpoints go to a separate '-vis' directory so a 1-channel model never collides with (or overwrites) your 4-channel checkpoints.">
        <input type="checkbox" name="vis_only" value="1">
        VIS-only input (1 channel, no NISP)</label>`;
  }

  // ── Resource-field markup ──────────────────────────────────────────
  //
  // Resource inputs start EMPTY. Either the history-driven prefill
  // (fetchHistoryAndPrefill) fills them from a past matching successful
  // run, or the user types values explicitly. The submit route rejects
  // blanks via StepResources.from_form_strict — no silent fallback.

  function resourceFields(step) {
    const d = step.defaults;
    // Locked CPU steps still emit ``n_cpus`` via a hidden input (same as
    // the GPU field below) — otherwise the value is never submitted and
    // StepResources.from_form_strict rejects the missing required field.
    const cpuField = (step.fixed_cpus != null)
      ? `<input type="hidden" name="n_cpus" value="${step.fixed_cpus}">
         <span class="muted" title="this step is single-threaded; allocation is locked">CPUs: <b>${step.fixed_cpus}</b> (locked)</span>`
      : `<label>CPUs
          <input type="number" name="n_cpus" min="1" max="64"
                 placeholder="e.g. ${d.n_cpus}"></label>`;
    // GPU field is hidden + locked to 0 for CPU-only steps. The form
    // still emits ``n_gpus=0`` via a hidden input so StepResources.from
    // _form_strict doesn't reject the submission for the missing field.
    const gpuField = step.needs_gpu
      ? `<label>GPUs
          <input type="number" name="n_gpus" min="0" max="8"
                 placeholder="e.g. ${d.n_gpus}"></label>`
      : `<input type="hidden" name="n_gpus" value="0">
         <span class="muted" title="this step is CPU-only">GPUs: <b>0</b> (locked)</span>`;
    return `
      <label>Partition
        <input type="text" name="partition" size="8"
               placeholder="e.g. ${d.partition}"></label>
      ${cpuField}
      ${gpuField}
      <label>Memory
        <input type="text" name="memory" size="6"
               placeholder="e.g. ${d.memory}"></label>
      <label>Wall time
        <input type="text" name="time_limit" size="9"
               placeholder="e.g. ${d.time_limit}"
               title="SLURM time format:
  30          → 30 minutes
  30:00       → 30 min 0 sec
  0:30:00     → 0 hr 30 min
  30:00:00    → 30 HOURS (not minutes!)
  1-06:00:00  → 1 day 6 hr"></label>`;
  }

  // ── Card markup ────────────────────────────────────────────────────

  function cardMarkup(step, artifactStatus) {
    artifactStatus = artifactStatus || {};
    // Map step id → which artifact existence this step PRODUCES.
    const produces = {
      download: 'tiles', extract_psf: 'psf', kernel: 'kernel',
      tfrecords: 'records', train: 'ckpt',
      euclid_sky_download: 'euclid_sky',
      euclid_roundtrip_tfrecords: 'roundtrip_records',
      // New per-page tasks (registered in Phase 2 of the migration):
      euclid_query:            'catalog',
      download_euclid_cutouts: 'euclid_cutouts',
      extract_euclid_psf:      'euclid_psf',
      euclid_star_anchor_tfrecords: 'star_anchor_records',
      synthetic_generate:      'synthetic_records',
      download_tng_skirt:      'tng_skirt',
    }[step.step_id];
    const status = artifactStatus[produces];
    let statusBadge = '';
    if (status === true)  statusBadge = '<span class="badge badge-done">✓ output present</span>';
    else if (status === false) statusBadge = '<span class="badge badge-running">○ output missing</span>';

    return `
    <section class="card">
      <h3>${escapeHtml(step.label)} ${statusBadge}</h3>

      <form data-step-id="${step.step_id}" class="hst-step-form"
            autocomplete="off">
        <fieldset class="task-fields">
          <legend>Task parameters</legend>
          <div class="form-row" style="flex-wrap:wrap; gap:12px;">
            ${taskFields(step)}
          </div>
        </fieldset>

        <fieldset class="history-fields">
          <legend>Previous runs</legend>
          <div class="js-history-panel"
               data-step-id="${step.step_id}">
            <span class="muted">loading history…</span>
          </div>
        </fieldset>

        <fieldset class="resource-fields">
          <legend>Resources
            <span class="js-prefill-hint muted"
                  style="font-weight:normal; margin-left:8px;"></span>
          </legend>
          <div class="form-row" style="flex-wrap:wrap; gap:12px;">
            ${resourceFields(step)}
          </div>
        </fieldset>

        <button type="submit" style="margin-top:8px;">Submit ${escapeHtml(step.step_id)}</button>
      </form>
      <div class="hst-step-status" data-step-id="${step.step_id}"></div>
    </section>`;
  }

  // ── Browser-autofill defence ───────────────────────────────────────
  //
  // Chrome and Safari ignore autocomplete="off" on type=number inputs
  // and aggressively replace the explicit value="..." attribute with
  // whatever the user most recently submitted under the same field
  // name — even across distinct forms. Force every input's runtime
  // ``.value`` back to its HTML ``value`` attribute right after render.

  function forceFormDefaults(area) {
    area.querySelectorAll('input[type="number"], input[type="text"]')
      .forEach(el => {
        const explicit = el.getAttribute('value');
        if (explicit != null && explicit !== '') {
          el.value = explicit;
        }
      });
  }

  // ── History fetch + resources prefill ──────────────────────────────

  function taskParamsFromForm(form) {
    const skip = new Set([...RESOURCE_FIELDS, 'confirm', 'label', 'preset']);
    const out = new FormData();
    for (const [k, v] of new FormData(form).entries()) {
      if (skip.has(k)) continue;
      out.append(k, v);
    }
    return out;
  }

  function renderHistoryRows(rows) {
    if (!rows || !rows.length) {
      return `<span class="muted">no previous runs for this step yet</span>`;
    }
    const head = `
      <tr>
        <th>submitted</th><th>state</th><th>task</th>
        <th>cpus</th><th>gpus</th><th>mem</th><th>time</th>
        <th>elapsed</th>
        <th title="CPU efficiency = CPU-time used ÷ (elapsed × allocated CPUs). ~100% means the allocated cores were fully busy; low means cores sat idle.">CPU util</th>
        <th title="Peak resident memory (MaxRSS) vs the memory you requested.">mem util</th>
        <th title="GPU compute utilisation (mean, with peak) sampled by nvidia-smi during the run. Blank for CPU-only jobs or runs from before GPU sampling.">GPU util</th>
      </tr>`;
    const rowHtml = rows.slice(0, 20).map(r => {
      const st = (r.state || '').toUpperCase();
      // CANCELLED / TIMEOUT are not crashes — the job ran and was stopped
      // (by the user or the wall-clock limit). Flag them amber, not red,
      // and still show whatever utilisation sacct captured for the partial
      // run rather than treating the row as a failure with no data.
      // Prefix-match: sacct emits variants like "CANCELLED by 1234" or
      // "CANCELLED+" depending on version.
      const stateClass = (st === 'COMPLETED' || st === 'DONE') ? 'badge-done'
                       : (st === '') ? 'badge-running'
                       : (st.startsWith('CANCELLED') || st.startsWith('TIMEOUT')) ? 'badge-cancelled'
                       : 'badge-failed';
      const taskShort = r.params_json
        ? escapeHtml(r.params_json.length > 60
            ? r.params_json.slice(0, 57) + '…' : r.params_json)
        : '—';
      const elapsed = r.elapsed_seconds ? fmtRuntime(parseFloat(r.elapsed_seconds)) : '—';
      // CPU utilisation: sacct's cpu_efficiency (a 0–1 fraction). Recorded
      // for cancelled/timeout jobs too; blank when sacct had nothing
      // (e.g. cancelled before the batch step ran) → show "—".
      const eff     = parseFloat(r.cpu_efficiency);
      const cpuUtil = Number.isFinite(eff) ? `${(eff * 100).toFixed(0)}%` : '—';
      // Memory utilisation: peak RSS, and a % of the allocation when known.
      const rssMb   = parseFloat(r.max_rss_mb);
      const allocMb = parseFloat(r.alloc_memory_mb);
      let memUtil = '—';
      if (Number.isFinite(rssMb)) {
        memUtil = (Number.isFinite(allocMb) && allocMb > 0)
          ? `${(100 * rssMb / allocMb).toFixed(0)}% (${rssMb.toFixed(0)} MB)`
          : `${rssMb.toFixed(0)} MB`;
      }
      // GPU utilisation: mean (+ peak) from the run's nvidia-smi samples,
      // folded into the post-mortem. Blank for CPU-only jobs.
      const gpuMean = parseFloat(r.gpu_util_mean);
      const gpuPeak = parseFloat(r.gpu_util_peak);
      let gpuUtil = '—';
      if (Number.isFinite(gpuMean)) {
        gpuUtil = Number.isFinite(gpuPeak)
          ? `${gpuMean.toFixed(0)}% (peak ${gpuPeak.toFixed(0)}%)`
          : `${gpuMean.toFixed(0)}%`;
      }
      return `<tr>
        <td><code>${escapeHtml(fmtIsoLocal(r.submitted_at))}</code></td>
        <td><span class="badge ${stateClass}">${escapeHtml(r.state || 'pending')}</span></td>
        <td><code title="${escapeHtml(r.params_json || '')}">${taskShort}</code></td>
        <td>${escapeHtml(r.req_cpus || '')}</td>
        <td>${escapeHtml(r.req_gpus || '')}</td>
        <td>${escapeHtml(r.req_memory || '')}</td>
        <td>${escapeHtml(r.req_time_limit || '')}</td>
        <td>${elapsed}</td>
        <td>${cpuUtil}</td>
        <td>${memUtil}</td>
        <td>${gpuUtil}</td>
      </tr>`;
    }).join('');
    return `<table class="history-table"><thead>${head}</thead><tbody>${rowHtml}</tbody></table>`;
  }

  function maybePrefillResources(form, match) {
    if (!match) return false;
    let filled = 0;
    for (const f of RESOURCE_FIELDS) {
      const el = form.querySelector(`input[name="${f}"]`);
      if (!el) continue;
      if (el.value && el.value.trim() !== '') continue;
      const sourceKey = (f === 'partition') ? 'partition' : `req_${f}`;
      const value = match[sourceKey];
      if (value == null || value === '') continue;
      el.value = value;
      filled++;
    }
    return filled > 0;
  }

  async function fetchHistoryAndPrefill(form) {
    const stepId = form.dataset.stepId;
    const panel  = form.querySelector('.js-history-panel');
    const hint   = form.querySelector('.js-prefill-hint');
    if (!panel) return;
    try {
      const resp = await fetch(`/api/fasrc/hst/${stepId}/history`, {
        method: 'POST',
        body:   taskParamsFromForm(form),
      });
      if (!resp.ok) {
        panel.innerHTML = `<span class="muted">history unavailable (HTTP ${resp.status})</span>`;
        return;
      }
      const data = await resp.json();
      panel.innerHTML = renderHistoryRows(data.history);
      if (data.match) {
        const filled = maybePrefillResources(form, data.match);
        if (hint) {
          hint.textContent = filled
            ? `prefilled from job ${data.match.jobid} (${data.match.state || 'unknown'})`
            : `match found (job ${data.match.jobid}) — your manual entries kept`;
        }
      } else if (hint) {
        hint.textContent = 'no exact match in history — enter resources manually';
      }
    } catch (err) {
      panel.innerHTML = `<span class="muted">history fetch failed: ${escapeHtml(String(err))}</span>`;
    }
  }

  function wireHistoryAndPrefill(scope) {
    scope.querySelectorAll('.hst-step-form').forEach(form => {
      fetchHistoryAndPrefill(form);
      wireExtractMemEstimate(form);
      let timer = null;
      const debounced = () => {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => fetchHistoryAndPrefill(form), 300);
      };
      form.querySelectorAll('.task-fields input').forEach(el => {
        el.addEventListener('input', debounced);
      });
    });
  }

  // ── HST-PSF-extract memory estimate ────────────────────────────────
  //
  // EPSFBuilder works in float64 and keeps several copies of *every*
  // star cutout through its iterations, so peak RAM is dominated by
  // ``n_stars · (2·half_extract+1)² · ~60 B`` (empirically measured: a
  // 500-star / 1023² run peaked ~32 GB). Surface a live estimate so a
  // big PSF / high n_stars doesn't OOM by surprise.

  function _extractMemEstimateGB(nStars, halfSide, marginFrac) {
    if (!(nStars > 0) || !(halfSide > 0)) return null;
    const marginPx = marginFrac > 0
      ? Math.max(1, Math.round(halfSide * marginFrac)) : 0;
    const stamp = 2 * (halfSide + marginPx) + 1;
    // ~60 B/star-px × 1.4 safety, + ~2 GB for the materialised HLSP tile.
    return Math.ceil(nStars * stamp * stamp * 60e-9 * 1.4) + 2;
  }

  function wireExtractMemEstimate(form) {
    if (form.dataset.stepId !== 'extract_psf') return;
    const hint   = form.querySelector('.js-extract-mem');
    const memEl  = form.querySelector('input[name="memory"]');
    const nEl    = form.querySelector('input[name="n_stars"]');
    const hEl    = form.querySelector('input[name="half_side"]');
    const mEl    = form.querySelector('input[name="extract_margin_frac"]');
    if (!hint || !nEl || !hEl) return;
    const num = (el, fb) => parseFloat((el && (el.value || el.placeholder)) || fb);
    const recompute = () => {
      const gb = _extractMemEstimateGB(
        num(nEl, '0'), num(hEl, '0'), num(mEl, '0'));
      if (gb == null) { hint.textContent = ''; return; }
      const stamp = 2 * (num(hEl, '0')
        + (num(mEl, '0') > 0 ? Math.max(1, Math.round(num(hEl, '0') * num(mEl, '0'))) : 0)) + 1;
      hint.textContent =
        `≈ ${gb} GB RAM for ${num(nEl, '0')} stars at ${stamp}² — set Memory ≥ this.`;
      // Guide the Memory field (placeholder only, so history prefill /
      // a typed value still win).
      if (memEl) memEl.placeholder = `e.g. ${gb}G`;
    };
    [nEl, hEl, mEl].forEach(el => el && el.addEventListener('input', recompute));
    recompute();
  }

  // ── Submit handler ─────────────────────────────────────────────────

  async function _onSubmit(e, form, statusEl, opts) {
    e.preventDefault();
    const stepId = form.dataset.stepId;

    // Defence — confirmation dialog showing the FULL payload.
    const body    = new FormData(form);
    const summary = [];
    for (const [k, v] of body.entries()) {
      if (v == null || v === '') continue;
      summary.push(`  ${k} = ${v}`);
    }
    const n_gpus_val   = parseInt(body.get('n_gpus') ?? '0', 10);
    const partition_val = (body.get('partition') ?? '').trim();
    const cpuOnlyPartitions = ['shared', 'serial_requeue', 'test'];
    let warning = '';
    if (n_gpus_val > 0 && cpuOnlyPartitions.includes(partition_val)) {
      warning =
        `\n⚠️  WARNING: you asked for ${n_gpus_val} GPU(s) on the ` +
        `"${partition_val}" partition, which is CPU-only on FASRC. ` +
        `sbatch will reject this with "Requested node configuration ` +
        `is not available".\n` +
        `Change Partition to "gpu" (or "gpu_test" for short jobs) ` +
        `before submitting.\n`;
    }
    // Fixed-layout batch sanity for the train step: the synthetic lane
    // must have ≥1 row (it's always on) and the total batch must be ≥1.
    if (stepId === 'train') {
      const nSyn = parseInt(body.get('n_syn') ?? '0', 10) || 0;
      const nHst = parseInt(body.get('n_hst') ?? '0', 10) || 0;
      const nAnchor = parseInt(body.get('n_anchor') ?? '0', 10) || 0;
      if (nSyn < 1) {
        warning +=
          `\n⚠️  WARNING: # synthetic = ${nSyn}, but the synthetic lane is ` +
          `always required (≥ 1). The job aborts at startup.\n`;
      } else {
        // Friendly confirmation of the resulting batch composition.
        const batch = nSyn + nHst + nAnchor;
        warning +=
          `\nBatch layout: ${nSyn} synthetic + ${nHst} HST + ${nAnchor} ` +
          `star-anchor = batch ${batch}.\n`;
      }
    }
    const msg =
      `Submit a SLURM job to FASRC?\n\n` +
      `Step: ${stepId}\n\n` +
      `Parameters:\n${summary.join('\n') || '  (no overrides)'}\n` +
      warning +
      `\nThis will run on FASRC and consume real cluster resources. ` +
      `Click OK to proceed, Cancel to abort.`;
    if (!window.confirm(msg)) {
      statusEl.innerHTML = '<span class="muted">cancelled by user</span>';
      return;
    }
    // Set the explicit confirmation token AFTER OK so a programmatic
    // POST from cached JS / extensions can't smuggle one through.
    body.set('confirm', 'yes');

    statusEl.innerHTML = '<span class="muted">submitting…</span>';
    try {
      const resp = await fetch(`/api/fasrc/hst/${stepId}/submit`, {
        method: 'POST', body,
      });
      const data = await resp.json();
      if (data.ok && data.queued) {
        // A job is already running — this one was queued locally. It will
        // be submitted automatically when the running job succeeds (and
        // not at all if it fails). Shown on the FASRC Current Submission
        // tab's queue list.
        const n = (data.queue && data.queue.count) || 0;
        statusEl.innerHTML = `<span class="badge badge-pending">queued</span>
          <span class="muted">behind the running job — position ${n}.
          See the queue on the Current Submission tab.</span>`;
      } else if (data.ok) {
        statusEl.innerHTML = _liveStatusMarkup(data.jobid);
        const cardEl = statusEl.querySelector(".job-status-card");
        if (cardEl && window.JobStatusCard) {
          new window.JobStatusCard(cardEl).start();
        }
        if (opts && typeof opts.onSubmitted === 'function') {
          // Page-specific follow-up (refresh inspection UI, badges …).
          try { opts.onSubmitted(data.jobid, data); } catch (e) { /* swallow */ }
        }
      } else {
        statusEl.innerHTML = `<span class="badge badge-failed">failed</span>
          <code>${escapeHtml(data.error || 'unknown')}</code>`;
      }
    } catch (err) {
      statusEl.innerHTML = `<span class="badge badge-failed">network error</span>
        <code>${escapeHtml(String(err))}</code>`;
    }
  }

  function _liveStatusMarkup(jobid) {
    return `
      <div class="job-status-card" data-jobid="${jobid}">
        <div class="job-status-head">
          <span class="badge badge-done">submitted</span>
          SLURM job <code>${jobid}</code> —
          <span class="js-stage is-empty">waiting for first event…</span>
          <span class="js-step-label"></span>
        </div>
        <progress class="js-progress" max="1" value="0" hidden></progress>
        <div class="js-metrics job-status-metrics" hidden></div>
        <details class="job-status-warnings">
          <summary>Warnings (<span class="js-warn-count">0</span>)</summary>
          <ul class="js-warn-list event-list"></ul>
        </details>
        <details class="job-status-errors">
          <summary>Errors (<span class="js-err-count">0</span>)</summary>
          <ul class="js-err-list event-list"></ul>
        </details>
      </div>`;
  }

  function _wireForms(scope, opts) {
    // Block Enter on every input — prevents accidental form submit.
    scope.querySelectorAll('.hst-step-form input').forEach(el => {
      el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          e.stopPropagation();
        }
      });
    });
    scope.querySelectorAll('.hst-step-form').forEach(form => {
      const stepId = form.dataset.stepId;
      const statusEl = scope.querySelector(
        `.hst-step-status[data-step-id="${stepId}"]`,
      );
      form.addEventListener('submit', e => _onSubmit(e, form, statusEl, opts));
    });
  }

  // ── Public entry points ────────────────────────────────────────────

  function renderMany(containerEl, steps, artifactStatus, opts) {
    if (!containerEl) return;
    containerEl.innerHTML = (steps || []).map(s => cardMarkup(s, artifactStatus)).join('');
    forceFormDefaults(containerEl);
    wireHistoryAndPrefill(containerEl);
    _wireForms(containerEl, opts || {});
  }

  async function mountOne(containerEl, stepId, opts) {
    if (!containerEl) return;
    opts = opts || {};
    containerEl.innerHTML = '<span class="muted">loading step…</span>';
    let payload;
    try {
      const r = await fetch('/api/fasrc/hst/status', { credentials: 'same-origin' });
      payload = await r.json();
    } catch (e) {
      containerEl.innerHTML =
        `<span class="muted">step config unavailable: ${escapeHtml(String(e))}</span>`;
      return;
    }
    const step = (payload.steps || []).find(s => s.step_id === stepId);
    if (!step) {
      containerEl.innerHTML =
        `<span class="muted">step ${escapeHtml(stepId)} is not registered on the server</span>`;
      return;
    }
    renderMany(containerEl, [step], payload.artifacts || {}, opts);
  }

  window.FasrcStepCard = {
    renderMany,
    mountOne,
    // Exposed so existing pages can call individual helpers if needed.
    _internal: {
      cardMarkup, taskFields, resourceFields,
      forceFormDefaults, wireHistoryAndPrefill,
      fetchHistoryAndPrefill, taskParamsFromForm,
      renderHistoryRows, maybePrefillResources,
      fmtRuntime, fmtIsoLocal,
    },
  };
})(window);
