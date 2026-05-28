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
            <input type="number" name="n_stars" value="200" min="20" max="5000"></label>`;
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
      case 'train':
        return _hstTrainFields();
      default:
        return '';
    }
  }

  function _tfrecordsFields() {
    // Analytic-A forward model is single-stamp, CPU-only — no GPU, no
    // batch_size knob. The script convolves one HR cube at a time and
    // appends to the output TFRecord. ``max_relative_noise`` is the
    // bright-stamp-rejection threshold (√S_max · |A|_peak > k · σ_LR).
    return `
      <label>Train scenes
        <input type="number" name="n_train" value="2000" min="100" max="50000"></label>
      <label>Validate scenes
        <input type="number" name="n_valid" value="200" min="20" max="5000"></label>
      <label>Image size
        <input type="number" name="image_size" value="510" step="2" min="64" max="2048"></label>
      <label>Max relative noise (k)
        <input type="number" name="max_relative_noise" value="5.0" step="0.5" min="0.5" max="50"
               title="Reject a stamp when √S_max·|A|_peak > k·σ_LR — i.e. when the brightest HR pixel's Poisson noise propagated through A would exceed k× the per-pixel Euclid LR noise floor. Smaller k → stricter rejection (fewer stars sneak in); larger k → more stamps kept. Default 5."></label>`;
  }

  function _hstTrainFields() {
    return `
      <label>Training steps
        <input type="number" name="steps" value="400000" min="1000" max="2000000"></label>
      <label>Batch size
        <input type="number" name="batch_size" value="16" min="1" max="64"></label>
      <label>HST fraction
        <input type="number" name="hst_fraction" value="0.1" step="0.05" min="0" max="1"></label>
      <label>Round-trip fraction
        <input type="number" name="roundtrip_fraction" value="0" step="0.05" min="0" max="1"></label>
      <label>Save-best w·synthetic
        <input type="number" name="save_best_w_syn" value="1.0" step="0.5" min="0" max="100"
               title="Weight of synthetic PSNR (dB) in the composite save-best score."></label>
      <label>Save-best w·HST
        <input type="number" name="save_best_w_hst" value="1.0" step="0.5" min="0" max="100"
               title="Weight of HST PSNR (dB) in the composite save-best score. No effect if no HST validate split exists."></label>
      <label>Save-best w·round-trip
        <input type="number" name="save_best_w_rt" value="0" step="1" min="0" max="100"
               title="Weight of the round-trip recon loss (SUBTRACTED — lower is better). RT loss is asinh-L1 (~0.1–1) vs PSNRs in dB (~20–30), so this needs ~10–30 to matter, and it can be gamed by under-sharpening. Default 0 = monitored only."></label>`;
  }

  // ── Resource-field markup ──────────────────────────────────────────
  //
  // Resource inputs start EMPTY. Either the history-driven prefill
  // (fetchHistoryAndPrefill) fills them from a past matching successful
  // run, or the user types values explicitly. The submit route rejects
  // blanks via StepResources.from_form_strict — no silent fallback.

  function resourceFields(step) {
    const d = step.defaults;
    const cpuField = (step.fixed_cpus != null)
      ? `<span class="muted" title="this step is single-threaded; allocation is locked">CPUs: <b>${step.fixed_cpus}</b> (locked)</span>`
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
        <th>elapsed</th><th>max RSS</th>
      </tr>`;
    const rowHtml = rows.slice(0, 20).map(r => {
      const stateClass = (r.state === 'COMPLETED') ? 'badge-done'
                       : (r.state === '') ? 'badge-running'
                       : 'badge-failed';
      const taskShort = r.params_json
        ? escapeHtml(r.params_json.length > 60
            ? r.params_json.slice(0, 57) + '…' : r.params_json)
        : '—';
      const elapsed = r.elapsed_seconds ? fmtRuntime(parseFloat(r.elapsed_seconds)) : '—';
      const rss     = r.max_rss_mb ? `${parseFloat(r.max_rss_mb).toFixed(0)} MB` : '—';
      return `<tr>
        <td><code>${escapeHtml(fmtIsoLocal(r.submitted_at))}</code></td>
        <td><span class="badge ${stateClass}">${escapeHtml(r.state || 'pending')}</span></td>
        <td><code title="${escapeHtml(r.params_json || '')}">${taskShort}</code></td>
        <td>${escapeHtml(r.req_cpus || '')}</td>
        <td>${escapeHtml(r.req_gpus || '')}</td>
        <td>${escapeHtml(r.req_memory || '')}</td>
        <td>${escapeHtml(r.req_time_limit || '')}</td>
        <td>${elapsed}</td>
        <td>${rss}</td>
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
