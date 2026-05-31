/* JobStatusCard — structured live progress for one FASRC job.
 *
 * Polls /api/fasrc/jobs/<jobid>/status every ~1.5s and updates real
 * DOM elements (stage chip, progress bar, warnings/errors lists). One
 * card per row; the same class drives the HST pipeline page and the
 * legacy training page.
 *
 * Usage (in a template):
 *
 *   <div class="job-status-card" data-jobid="12345">
 *     <span class="js-stage"></span>
 *     <progress class="js-progress" max="1" value="0"></progress>
 *     <span class="js-step-label"></span>
 *     <ul class="js-warn-list"></ul>
 *     <ul class="js-err-list"></ul>
 *     <details class="js-raw-log">
 *       <summary>Show raw log</summary>
 *       <pre class="js-raw-log-body"></pre>
 *     </details>
 *   </div>
 *
 *   <script>
 *     document.querySelectorAll(".job-status-card").forEach(el => {
 *       new JobStatusCard(el).start();
 *     });
 *   </script>
 *
 * The card is dumb about layout — caller picks CSS. We only touch the
 * elements identified by their ``js-*`` class so styling stays in CSS.
 */

class JobStatusCard {
  /**
   * @param {HTMLElement} root - the element with .job-status-card +
   *   data-jobid="..."
   * @param {Object} [opts]
   * @param {number} [opts.intervalMs=1500] - poll cadence
   */
  constructor(root, opts = {}) {
    this.root      = root;
    this.jobid     = root.dataset.jobid;
    this.intervalMs = opts.intervalMs || 1500;
    this.timerId   = null;
    this.lastSig   = "";          // cheap dedupe to avoid useless DOM writes

    // Cache the slot elements once. Missing slots are tolerated — the
    // caller may opt out of any particular surface.
    this.slots = {
      stage:     root.querySelector(".js-stage"),
      progress:  root.querySelector(".js-progress"),
      stepLabel: root.querySelector(".js-step-label"),
      resources: root.querySelector(".js-resources"),
      warnList:  root.querySelector(".js-warn-list"),
      errList:   root.querySelector(".js-err-list"),
      rawLog:    root.querySelector(".js-raw-log-body"),
    };
  }

  start() {
    if (this.timerId) return;
    this.tick();
    this.timerId = setInterval(() => this.tick(), this.intervalMs);
  }

  stop() {
    if (this.timerId) clearInterval(this.timerId);
    this.timerId = null;
  }

  async tick() {
    let payload;
    try {
      const r = await fetch(`/api/fasrc/jobs/${this.jobid}/status`,
                            { credentials: "same-origin" });
      if (!r.ok) {
        this.markOffline(`HTTP ${r.status}`);
        return;
      }
      payload = await r.json();
    } catch (e) {
      this.markOffline(String(e));
      return;
    }
    if (!payload || !payload.ok) {
      this.markOffline(payload && payload.error || "unknown error");
      return;
    }

    // Stop polling once the job is in a terminal state — the server
    // tags state via the existing reconcile loop. Cards on terminated
    // jobs still show the last-known structured status.
    const terminal = new Set([
      "COMPLETED", "DONE", "TIMEOUT", "FAILED", "CANCELLED",
    ]);
    if (terminal.has(payload.state)) this.stop();

    this.render(payload.status || {});
  }

  /**
   * Apply a JobStatus.to_dict() payload to the DOM. Slots that the
   * template chose not to include are silently skipped.
   *
   * @param {Object} status
   */
  render(status) {
    // Skip the DOM write if nothing meaningful changed since last tick.
    const r = status.resources;
    const sig = JSON.stringify({
      stage:    status.stage,
      step:     status.step,
      n_warn:   (status.warnings || []).length,
      n_err:    (status.errors   || []).length,
      // Round so sub-percent jitter doesn't force a DOM write every poll,
      // but real movement (and the sparkline) still updates.
      res:      r ? [Math.round(r.gpu_percent), Math.round(r.cpu_percent),
                     Math.round(r.gpu_mem_percent), (r.gpu_series || []).length]
                  : null,
    });
    if (sig === this.lastSig) return;
    this.lastSig = sig;

    if (this.slots.stage) {
      this.slots.stage.textContent = status.stage || "";
      this.slots.stage.classList.toggle("is-empty", !status.stage);
    }

    if (this.slots.progress) {
      const step = status.step;
      if (step && step.total > 0) {
        this.slots.progress.hidden = false;
        this.slots.progress.max   = step.total;
        this.slots.progress.value = step.current;
      } else {
        this.slots.progress.hidden = true;
      }
    }

    if (this.slots.stepLabel) {
      const step = status.step;
      if (step) {
        let label = step.label
          ? `${step.label} (${step.current}/${step.total})`
          : `${step.current}/${step.total}`;
        // Parallel phase: append the live worker count (cumulative count is
        // already the step current/total).
        const par = status.parallel;
        if (par && par.n_workers > 0) {
          label += ` · ${par.active_workers}/${par.n_workers} workers active`;
        }
        this.slots.stepLabel.textContent = label;
      } else {
        this.slots.stepLabel.textContent = "";
      }
    }

    this._renderResources(this.slots.resources, status.resources);

    this._renderList(this.slots.warnList, status.warnings, "warn");
    this._renderList(this.slots.errList,  status.errors,   "err");
  }

  /**
   * Render the smoothed GPU/CPU utilisation gauges + a GPU sparkline.
   * Hidden until the first ``resource`` sample arrives.
   *
   * @param {HTMLElement} el - the .js-resources slot (may be absent)
   * @param {Object|null} res - status.resources
   */
  _renderResources(el, res) {
    if (!el) return;
    if (!res || !res.n_samples) { el.hidden = true; return; }
    el.hidden = false;

    const pct = v => (v == null || !Number.isFinite(v))
      ? null : `${v.toFixed(0)}%`;
    const gauges = [];
    const gpu = pct(res.gpu_percent);
    if (gpu !== null) {
      const peak = pct(res.gpu_peak);
      const mem  = pct(res.gpu_mem_percent);
      gauges.push(
        `<span class="res-gauge" title="GPU compute utilisation `
        + `(smoothed; peak ${peak ?? '—'}, mem ${mem ?? '—'})">`
        + `GPU <b>${gpu}</b></span>`);
      if (mem !== null) {
        gauges.push(`<span class="res-gauge res-gauge-sub" `
          + `title="GPU memory utilisation">mem ${mem}</span>`);
      }
    }
    const cpu = pct(res.cpu_percent);
    if (cpu !== null) {
      const peak = pct(res.cpu_peak);
      gauges.push(
        `<span class="res-gauge" title="Node CPU utilisation `
        + `(smoothed; peak ${peak ?? '—'})">CPU <b>${cpu}</b></span>`);
    }
    const spark = this._sparkline(res.gpu_series && res.gpu_series.length
      ? res.gpu_series : res.cpu_series);
    el.innerHTML = gauges.join("")
      + (spark ? `<span class="res-spark" aria-hidden="true">${spark}</span>` : "");
  }

  /**
   * Tiny inline unicode sparkline (0–100%) from a numeric series.
   * Returns "" for an empty series.
   */
  _sparkline(series) {
    if (!series || !series.length) return "";
    const blocks = "▁▂▃▄▅▆▇█";
    // Fixed 0–100 scale so the height reads as absolute utilisation,
    // not auto-ranged — a flat-low GPU should look low, not full.
    return series.slice(-40).map(v => {
      const c = Math.max(0, Math.min(100, v));
      return blocks[Math.min(blocks.length - 1,
                             Math.round((c / 100) * (blocks.length - 1)))];
    }).join("");
  }

  _renderList(ul, events, kind) {
    if (!ul) return;
    events = events || [];
    // Drop & re-fill rather than diffing — events lists stay short
    // (typically <100 entries) and the DOM cost is negligible.
    ul.replaceChildren(...events.map(ev => {
      const li = document.createElement("li");
      li.className = `event event-${kind}`;
      const tsStr = new Date((ev.ts || 0) * 1000).toLocaleTimeString();
      const ts = document.createElement("time");
      ts.className = "event-ts";
      ts.textContent = tsStr;
      const msg = document.createElement("span");
      msg.className = "event-msg";
      msg.textContent = ev.msg || "";
      li.append(ts, msg);
      return li;
    }));
  }

  markOffline(reason) {
    if (this.slots.stage && !this.slots.stage.textContent) {
      this.slots.stage.textContent = `(offline: ${reason})`;
      this.slots.stage.classList.add("is-empty");
    }
  }
}

// Auto-bind every card on the page so templates don't need their own
// boot script. Pages that want manual control can opt out by adding
// data-autoinit="false".
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".job-status-card").forEach(el => {
    if (el.dataset.autoinit === "false") return;
    new JobStatusCard(el).start();
  });
});

window.JobStatusCard = JobStatusCard;
