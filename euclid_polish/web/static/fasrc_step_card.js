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
          <label>VIS cutout (px)
            <input type="number" name="vis_pixels" value="512" min="32" max="4096" step="32"
                   title="Cutout side in 0.10″/pix VIS pixels. NISP bands fetch the same angular footprint at their own native pixel count."></label>
          <label>Parallel workers
            <input type="number" name="workers" value="8" min="1" max="32"
                   title="Concurrent per-band downloads from the Euclid archive."></label>`;
      case 'extract_euclid_psf':
        // Mirrors EuclidPSFExtractStep.build_command (num_stars,
        // vis_pixels, output_size). Extracts ALL four bands in one job.
        return `
          <label>Stars per band
            <input type="number" name="num_stars" value="200" min="10" max="5000"
                   title="Up to this many cutouts per band feed EPSFBuilder."></label>
          <label>VIS cutout (px)
            <input type="number" name="vis_pixels" value="512" min="32" max="4096" step="32"
                   title="Shared angular field (in VIS px) used to pick each band's native cutout size — must match what was downloaded."></label>
          <label>Output PSF size (oversampled px)
            <input type="number" name="output_size" value="0" min="0" max="4096"
                   title="Final ePSF side in oversampled px. 0 → photutils' default (cutout_size × oversampling + 1). Even values are bumped down to odd."></label>`;
      case 'synthetic_generate':
        // Mirrors run_pipeline.py --skip-train (RunPipelineStep.build_command
        // reads n_train / n_valid / image_size). Renders synthetic clean HR
        // scenes + forward-models to dirty Euclid LR.
        return `
          <label>Train scenes
            <input type="number" name="n_train" value="6400" min="1" max="50000"></label>
          <label>Validate scenes
            <input type="number" name="n_valid" value="100" min="1" max="5000"></label>
          <label>HR image size (px)
            <input type="number" name="image_size" value="252" step="2" min="60" max="2048"
                   title="HR scene side in 0.05″/pix pixels. Forward-modelled to 0.10″ LR (and 0.30″→Lanczos for NISP)."></label>`;
      case 'train':
        return _hstTrainFields();
      default:
        return '';
    }
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
      <label>Batch size
        <input type="number" name="batch_size" value="4" min="1" max="64"
               title="Examples per step at HR-crop 192 / LR-crop 96. 4 keeps activation memory ~constant vs the old 96-crop; on an A100 you can go much higher (32–64) to raise GPU utilisation — watch the live GPU gauge. NOTE fixed-layout constraint: batch × each non-zero fraction must round to ≥1 row, or the job aborts."></label>
      <label>Learning rate
        <input type="number" name="learning_rate" value="" step="0.0001" min="0" max="0.01"
               placeholder="blank = 1e-3→5e-4 schedule"
               title="Constant Adam LR for the whole run (e.g. 0.001). Leave blank for the default two-phase schedule 1e-3 → 5e-4 at steps//2. Constant is simpler for an exploratory restart; the decay gives a slightly sharper final result. Scale up with a larger batch."></label>
      <label>HST fraction
        <input type="number" name="hst_fraction" value="0.1" step="0.05" min="0" max="1"
               title="Share of each batch drawn from HST records. Trains |H⊛SR − HST_image| (SR=sky). Fixed layout: n_hst = round(batch × this) and must be ≥1 — e.g. batch 4 needs fraction ≥ 0.25, batch 32 allows 0.03+. 0 disables the HST lane."></label>
      <label>Round-trip fraction
        <input type="number" name="roundtrip_fraction" value="0" step="0.05" min="0" max="1"
               title="Share of each batch drawn from real-Euclid round-trip records. Trains |E⊛SR − LR_vis| (cycle consistency). Same ≥1-row rule: round(batch × this) must be ≥1. 0 disables the round-trip lane."></label>
      <label>Loss weight · synthetic
        <input type="number" name="synthetic_loss_weight" value="1" step="0.5" min="0" max="100"
               title="Per-example TRAINING-loss multiplier for synthetic records. 1 = default; raise to up-weight that source's gradient, 0 to ablate (kept in the batch mix but zero gradient)."></label>
      <label>Loss weight · HST
        <input type="number" name="hst_loss_weight" value="1" step="0.5" min="0" max="100"
               title="Per-example TRAINING-loss multiplier for HST records. 1 = default. Distinct from the data fraction: fraction sets how many HST examples per batch, this scales each one's loss."></label>
      <label>Loss weight · round-trip
        <input type="number" name="roundtrip_loss_weight" value="1" step="0.5" min="0" max="100"
               title="Per-example TRAINING-loss multiplier for round-trip records. Round-trip gradient share ≈ roundtrip_fraction × this. Caution: the round-trip loss is minimized by an UN-sharpened (blurry) SR, so very high values push the model away from super-resolving — it's a regularizer, not a sharpening driver. Only used when round-trip fraction > 0."></label>
      <label>Forward-op PSF crop (½)
        <input type="number" name="forward_op_crop_half" value="0" step="8" min="0" max="512"
               title="Optional central crop of the VIS PSF for the round-trip forward op → (2·crop+1)² kernel. 0 = full 1023×1023 PSF (the forward op convolves via FFT, so the full PSF is exact AND fast — no crop needed). Set >0 only to ablate the PSF wings. Only used when round-trip fraction > 0."></label>
      <label>Save-best w·synthetic
        <input type="number" name="save_best_w_syn" value="1.0" step="0.5" min="0" max="100"
               title="Weight of synthetic PSNR (dB) in the composite save-best score."></label>
      <label>Save-best w·HST
        <input type="number" name="save_best_w_hst" value="1.0" step="0.5" min="0" max="100"
               title="Weight of HST PSNR (dB) in the composite save-best score. No effect if no HST validate split exists."></label>
      <label>Save-best w·round-trip
        <input type="number" name="save_best_w_rt" value="0" step="0.5" min="0" max="100"
               title="Weight of the round-trip PSNR (dB, ADDED — higher is better) in the composite save-best score. Now on the same dB scale as the other two PSNRs, so a weight near 1 is comparable; note round-trip PSNR is measured at LR resolution so it sits higher in absolute dB. Default 0 = monitored only."></label>`;
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
      download_euclid_cutouts: 'euclid_cutouts',
      extract_euclid_psf:      'euclid_psf',
      synthetic_generate:      'synthetic_records',
    }[step.step_id];
    const status = artifactStatus[produces];
    let statusBadge = '';
    if (status === true)  statusBadge = '<span class="badge badge-done">✓ output present</span>';
    else if (status === false) statusBadge = '<span class="badge badge-running">○ output missing</span>';

    const med = step.median_runtime_s;
    const medLine = med != null
      ? `<span class="muted">median runtime: ${fmtRuntime(med)} (last 5)</span>`
      : `<span class="muted">no runtime history yet</span>`;

    return `
    <section class="card">
      <h3>${escapeHtml(step.label)} ${statusBadge}</h3>
      <p class="hint">${escapeHtml(step.description)}</p>
      <p class="hint">${medLine}</p>

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
    // Fixed-layout batch sanity for the train step: each non-zero source
    // fraction must yield ≥1 row at this batch size, and synthetic rows
    // must remain — otherwise fasrc_train_with_hst.py aborts at startup.
    if (stepId === 'train') {
      const bs = parseInt(body.get('batch_size') ?? '0', 10);
      const hf = parseFloat(body.get('hst_fraction') ?? '0') || 0;
      const rf = parseFloat(body.get('roundtrip_fraction') ?? '0') || 0;
      const nHst = Math.round(bs * hf);
      const nRt  = Math.round(bs * rf);
      const nSyn = bs - nHst - nRt;
      const issues = [];
      if (hf > 0 && nHst < 1)
        issues.push(`HST fraction ${hf} × batch ${bs} rounds to 0 HST rows`);
      if (rf > 0 && nRt < 1)
        issues.push(`round-trip fraction ${rf} × batch ${bs} rounds to 0 rows`);
      if (Number.isFinite(nSyn) && nSyn < 1)
        issues.push(`no synthetic rows left (n_syn=${nSyn})`);
      if (issues.length) {
        warning +=
          `\n⚠️  WARNING: fixed batch layout problem — ${issues.join('; ')}. ` +
          `The job aborts at startup. Raise the batch size or adjust the ` +
          `fraction(s) so batch × fraction ≥ 1.\n`;
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
      if (data.ok) {
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
