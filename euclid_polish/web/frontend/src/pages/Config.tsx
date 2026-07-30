/* Config — universal per-job knobs (mirror of the classic /config page).
   useResource seeds a local editable copy; a sticky Save posts ALL fields via
   postForm. Numeric inputs are NumberField, enums are Select. Server-side
   coercion (e.g. VIS cutout forced odd) is reflected back on save. */
import { useEffect, useMemo, useState } from "react";
import { postForm } from "../api";
import { useResource } from "../hooks";
import {
  Badge, Button, Card, CardBody, CardHead, Empty, LogTail, NumberField, Page,
  PageHead, Select, Spinner,
} from "../ui";

/* Every JobConfig field. Kept as strings in form state (inputs are strings);
   the interface documents the underlying numeric/enum shape. */
interface JobConfig {
  // Cutouts & PSF
  vis_pixels: number;
  // Synthetic scenes & HR size
  n_train: number;
  n_valid: number;
  n_test: number;
  hr_image_size: number;
  galaxy_density_arcmin2: number;
  // Lens-finder dataset
  lensfinder_n_fields: number;
  lensfinder_n_valid: number;
  lensfinder_image_size: number;
  // Lens-finder training
  lensfinder_epochs: number;
  lensfinder_patience: number;
  lensfinder_batch_size: number;
  lensfinder_learning_rate: number;
  lensfinder_training_mode: "head_only" | "full";
  // Star field
  star_density_arcmin2: number;
  star_mag_slope: number;
  star_mag_bright: number;
  star_mag_faint: number;
  // SR training · PSF distribution
  psf_warp_prob: number;
  psf_warp_alpha_max: number;
  psf_warp_sigma: number;
  saturation_mask_prob: number;
  // Strong lenses
  lens_density_arcmin2: number;
  lens_sigma_v_min_kms: number;
  lens_sigma_v_max_kms: number;
  // Inference display
  asinh_scale: number;
  // SR training · LR schedule
  lr_peak: number;
  lr_final: number;
  lr_warmup_steps: number;
  // SR training · reduce-LR-on-plateau
  plateau_lr_enabled: 0 | 1;
  plateau_lr_metric: "combined_loss" | "psnr_stretched";
  plateau_lr_factor: number;
  plateau_lr_patience: number;
  plateau_lr_min_delta: number;
  plateau_lr_cooldown: number;
  plateau_lr_min_lr: number;
}

type Field = keyof JobConfig;
type FormState = Record<Field, string>;

type ConfigResp = { ok: boolean; config: Record<string, number | string> };
type SaveResp = { ok: boolean; config: Record<string, number | string>; note?: string | null; error?: string };

/* Ordered list of every field so we can build the form state + the POST body
   without a hand-maintained duplicate. */
const FIELDS: Field[] = [
  "vis_pixels",
  "n_train", "n_valid", "n_test", "hr_image_size", "galaxy_density_arcmin2",
  "lensfinder_n_fields", "lensfinder_n_valid", "lensfinder_image_size",
  "lensfinder_epochs", "lensfinder_patience", "lensfinder_batch_size",
  "lensfinder_learning_rate", "lensfinder_training_mode",
  "star_density_arcmin2", "star_mag_slope", "star_mag_bright", "star_mag_faint",
  "psf_warp_prob", "psf_warp_alpha_max", "psf_warp_sigma",
  "saturation_mask_prob",
  "lens_density_arcmin2", "lens_sigma_v_min_kms", "lens_sigma_v_max_kms",
  "asinh_scale",
  "lr_peak", "lr_final", "lr_warmup_steps",
  "plateau_lr_enabled", "plateau_lr_metric", "plateau_lr_factor",
  "plateau_lr_patience", "plateau_lr_min_delta", "plateau_lr_cooldown",
  "plateau_lr_min_lr",
];

function toForm(config: Record<string, number | string>): FormState {
  const out = {} as FormState;
  for (const f of FIELDS) {
    const v = config[f];
    out[f] = v == null ? "" : String(v);
  }
  return out;
}

const TRAINING_MODE_OPTS: { value: "head_only" | "full"; label: string }[] = [
  { value: "head_only", label: "head_only (freeze encoder)" },
  { value: "full", label: "full (fine-tune all)" },
];
const METRIC_OPTS: { value: "combined_loss" | "psnr_stretched"; label: string }[] = [
  { value: "combined_loss", label: "combined_loss (min)" },
  { value: "psnr_stretched", label: "psnr_stretched (max)" },
];
const PLATEAU_ENABLED_OPTS: { value: "0" | "1"; label: string }[] = [
  { value: "1", label: "on (L1 members)" },
  { value: "0", label: "off" },
];

const GRID = "repeat(auto-fit, minmax(220px, 1fr))";

export default function ConfigPage() {
  const { data, loading, reload } = useResource<ConfigResp>("/api/config");
  const [loaded, setLoaded] = useState<FormState | null>(null);
  const [form, setForm] = useState<FormState | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);

  // Seed the editable copy once the config arrives.
  useEffect(() => {
    if (data?.config) {
      const f = toForm(data.config);
      setLoaded(f);
      setForm(f);
    }
  }, [data]);

  const dirty = useMemo(() => {
    if (!form || !loaded) return false;
    return FIELDS.some((f) => form[f] !== loaded[f]);
  }, [form, loaded]);

  function set(field: Field, value: string) {
    setForm((prev) => (prev ? { ...prev, [field]: value } : prev));
  }

  async function save() {
    if (!form) return;
    setBusy(true); setNote(null);
    try {
      const body: Record<string, string> = {};
      for (const f of FIELDS) body[f] = form[f];
      const r = await postForm<SaveResp>("/api/config/save", body);
      if (!r.ok) {
        setNote({ ok: false, text: r.error || "save failed" });
        return;
      }
      // Reflect any server-side coercion (e.g. VIS cutout forced odd).
      const next = toForm(r.config);
      setLoaded(next);
      setForm(next);
      setNote({ ok: true, text: r.note ? `saved — ${r.note}` : "saved" });
      reload();
    } catch (e) {
      setNote({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  }

  const num = (field: Field, label: string, opts: {
    min?: number; max?: number; step?: number; hint?: string;
  } = {}) => (
    <NumberField label={label} value={form ? form[field] : ""}
      onChange={(v) => set(field, v)} disabled={!form}
      min={opts.min} max={opts.max} step={opts.step} hint={opts.hint} />
  );

  return (
    <Page>
      <PageHead eyebrow="ops · config" title="Universal job config"
        sub="Shared knobs used across the pipeline. Set them once — they persist across reloads (~/.euclid_polish/job_config.json) and are sent automatically with the relevant jobs."
        right={
          <div className="row" style={{ gap: 8, alignItems: "center" }}>
            {dirty && <Badge tone="warn">unsaved changes</Badge>}
            <Button variant="primary" disabled={!form || busy || !dirty} onClick={save}>
              {busy ? "Saving…" : "Save config"}
            </Button>
          </div>
        } />

      {loading && !form && (
        <Card><CardBody><Empty><Spinner /> loading…</Empty></CardBody></Card>
      )}

      {!loading && !form && (
        <Card><CardBody><Empty>could not load config — try the classic /config page</Empty></CardBody></Card>
      )}

      {form && (
        <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
          <Card>
            <CardHead title="Cutouts & PSF"
              sub="→ /cutouts (download) and /psfs (extract)." />
            <CardBody>
              <div className="grid" style={{ gridTemplateColumns: GRID, gap: "var(--s3)" }}>
                {num("vis_pixels", "VIS cutout (px) · must be odd",
                  { min: 31, max: 4095, step: 2,
                    hint: "VIS cutout side in 0.10″/pix pixels. Must be odd (true centre pixel). Shared by the cutout download and the ePSF extraction so they always match." })}
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Synthetic scenes & population"
              sub="→ /sky and lens-finder generation; HR size also feeds /inference." />
            <CardBody>
              <div className="grid" style={{ gridTemplateColumns: GRID, gap: "var(--s3)" }}>
                {num("n_train", "Train scenes", { min: 1, max: 50000 })}
                {num("n_valid", "Validate scenes", { min: 1, max: 5000 })}
                {num("n_test", "Test scenes",
                  { min: 0, max: 5000,
                    hint: "Held-out eval set — training and save-best never touch it; /ensemble and synthetic evals run here. 0 disables the test split." })}
                {num("hr_image_size", "HR image size (px)",
                  { min: 60, max: 2048, step: 6,
                    hint: "HR scene side in 0.05″/pix pixels. Kept a multiple of 6 (the NISP rebin factor). Feeds synthetic generation and inference." })}
                {num("galaxy_density_arcmin2", "Galaxies (/arcmin²)",
                  { min: 0, max: 1000, step: 1,
                    hint: "Raw TNG draw density. A multi-cone Euclid fit updates this value; COSMOS does not set its normalization." })}
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Lens-finder dataset"
              sub="→ the lensfinder-data FASRC step (fewer, bigger fields; own records dir)." />
            <CardBody>
              <div className="grid" style={{ gridTemplateColumns: GRID, gap: "var(--s3)" }}>
                {num("lensfinder_n_fields", "Fields",
                  { min: 1, max: 50000,
                    hint: "Large fields for the lens-finder, into a dedicated records dir (records_lensfinder) — separate from the main training set." })}
                {num("lensfinder_n_valid", "Validate fields",
                  { min: 0, max: 5000, hint: "Validation fields for the lens-finder dataset." })}
                {num("lensfinder_image_size", "Field size (px)",
                  { min: 120, max: 2048, step: 6,
                    hint: "Lens-finder field side in 0.05″/pix HR pixels. Bigger fields → more sources + realistic crowding. Multiple of 6." })}
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Lens-finder training (Zoobot)"
              sub="→ the lensfinder-train FASRC step. Early-stopping-driven: max epochs is just the ceiling." />
            <CardBody>
              <div className="grid" style={{ gridTemplateColumns: GRID, gap: "var(--s3)" }}>
                {num("lensfinder_epochs", "Max epochs",
                  { min: 1, max: 1000,
                    hint: "Maximum training epochs (the ceiling). Early stopping via patience usually halts sooner. We ship 10." })}
                {num("lensfinder_patience", "Patience (early stop)",
                  { min: 1, max: 100,
                    hint: "Halt if validation loss does not improve for this many epochs. We ship 6." })}
                {num("lensfinder_batch_size", "Batch size",
                  { min: 1, max: 512,
                    hint: "Training batch size per GPU. Larger = faster but more GPU memory." })}
                {num("lensfinder_learning_rate", "Learning rate",
                  { min: 0, max: 1, step: 0.0001,
                    hint: "Initial learning rate for the optimizer. Default 1e-4. Lower = steadier fine-tuning." })}
                <div className="ui-field">
                  <span title="head_only freezes the Zoobot encoder and trains only the linear head. full fine-tunes all 15M encoder params (slower, overfit-prone).">Fine-tune depth</span>
                  <Select value={form.lensfinder_training_mode as "head_only" | "full"}
                    options={TRAINING_MODE_OPTS}
                    onChange={(v) => set("lensfinder_training_mode", v)} />
                </div>
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Star field"
              sub="→ /sky (synthetic generation): dN/dm ∝ 10^(α·m) over [bright, faint]." />
            <CardBody>
              <div className="grid" style={{ gridTemplateColumns: GRID, gap: "var(--s3)" }}>
                {num("star_density_arcmin2", "Star density (/arcmin²)",
                  { min: 0, step: 0.001,
                    hint: "Stellar surface density (stars per arcmin²). Per-scene count is Poisson(density × field area). Default ≈ 5000/deg²." })}
                {num("star_mag_slope", "Mag slope α",
                  { min: 0, max: 1, step: 0.001,
                    hint: "Differential star-count slope α in dN/dm ∝ 10^(α·m). Larger = relatively more faint stars." })}
                {num("star_mag_bright", "Brightest star (VIS mag)",
                  { min: 6, max: 24, step: 0.1,
                    hint: "Brightest synthetic star (VIS mag). Lower = brighter. Very bright single-pixel stars saturate." })}
                {num("star_mag_faint", "Faintest star (VIS mag)",
                  { min: 16, max: 30, step: 0.1,
                    hint: "Faintest synthetic star (VIS mag) — roughly the VIS noise floor." })}
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Strong lenses"
              sub="→ /sky (synthetic generation): θ_E from the SIS law, σ_v uniform in [min, max]." />
            <CardBody>
              <div className="grid" style={{ gridTemplateColumns: GRID, gap: "var(--s3)" }}>
                {num("lens_density_arcmin2", "Lens density (/arcmin²)",
                  { min: 0, step: 0.1,
                    hint: "Strong-lens surface density (lenses/arcmin²). Training-augmentation density (far above the real sky rate). Default 16.5." })}
                {num("lens_sigma_v_min_kms", "σ_v min (km/s)",
                  { min: 50, max: 600, step: 1,
                    hint: "Minimum lens velocity dispersion (km/s). Sets θ_E via the SIS law. Default range [150,350] → θ_E ~ 0.3–2.0″." })}
                {num("lens_sigma_v_max_kms", "σ_v max (km/s)",
                  { min: 50, max: 600, step: 1,
                    hint: "Maximum lens velocity dispersion (km/s). Raise for bigger Einstein radii (θ_E ∝ σ_v²). θ_E clamped to 0.10–3.5″." })}
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Synthetic data + SR training · PSF distribution"
              sub="→ one elastic PSF is shared by every source; clean/HR targets stay nominal." />
            <CardBody>
              <div className="grid" style={{ gridTemplateColumns: GRID, gap: "var(--s3)" }}>
                {num("psf_warp_prob", "Warp probability",
                  { min: 0, max: 1, step: 0.01,
                    hint: "Probability that each dirty exposure receives an elastic deformation of its shared empirical PSF. In on-the-fly train mode, a fresh warp is drawn on every visit. 1 = every draw; 0 = disabled." })}
                {num("psf_warp_alpha_max", "Warp α max (HR px)",
                  { min: 0, max: 100, step: 0.1,
                    hint: "Per exposure, α is sampled uniformly from [0, max]. The seeded draw is fixed in generated validation/test records; default 20 matches polish-pub." })}
                {num("psf_warp_sigma", "Warp σ (HR px)",
                  { min: 0.1, max: 100, step: 0.1,
                    hint: "Gaussian smoothing scale of the shared four-band PSF displacement field in 0.05″ HR pixels. Default 3 matches polish-pub." })}
                {num("saturation_mask_prob", "Dark-core probability",
                  { min: 0, max: 0.5, step: 0.01,
                    hint: "Chance that each above-well source is replaced by a rectangular blackout. Default 0.2; capped at 0.5 so at least half of bright stars retain the intact cores seen in real Euclid fields." })}
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Inference display"
              sub="→ /inference (both reconstruction forms)." />
            <CardBody>
              <div className="grid" style={{ gridTemplateColumns: GRID, gap: "var(--s3)" }}>
                {num("asinh_scale", "asinh scale (e⁻)",
                  { min: 0.01, step: 0.1,
                    hint: "Brightness knee (e⁻) for the asinh display panels. Smaller = brighter faint sources." })}
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="SR training · LR schedule (warmup → cosine)"
              sub="→ the ensemble-train FASRC step (and all local WDSR training). Scaled to each run's total steps." />
            <CardBody>
              <div className="grid" style={{ gridTemplateColumns: GRID, gap: "var(--s3)" }}>
                {num("lr_peak", "Peak LR",
                  { min: 0, max: 1, step: 0.00001,
                    hint: "Peak learning rate at the end of warmup — the cosine decay starts here. Old flat value was 5e-4." })}
                {num("lr_final", "Final LR",
                  { min: 0, max: 1, step: 0.00001,
                    hint: "Final learning rate at the last step (the cosine floor). Old schedule ended at 2e-5." })}
                {num("lr_warmup_steps", "Warmup steps",
                  { min: 0, max: 100000, step: 1,
                    hint: "Linear warmup length (steps): LR ramps up to Peak LR, then cosine-decays. Default 2000." })}
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="SR training · reduce-LR-on-plateau guard"
              sub="→ the ensemble-train FASRC step. Shares the LR-scale knob with the gradient-spike guard; halvings compound." />
            <CardBody>
              <div className="grid" style={{ gridTemplateColumns: GRID, gap: "var(--s3)" }}>
                <div className="ui-field">
                  <span title="Cut the LR (and roll back to best) when the validation metric stalls. Applied to L1 members ONLY (L2/L3/BerHu have no such basin and misfire).">Plateau guard (L1 only)</span>
                  <Select value={form.plateau_lr_enabled === "1" ? "1" : "0"}
                    options={PLATEAU_ENABLED_OPTS}
                    onChange={(v) => set("plateau_lr_enabled", v)} />
                </div>
                <div className="ui-field">
                  <span title="Metric to watch. combined_loss (lower better) is the checkpoint metric and least noisy; psnr_stretched (higher better) is the dB curve.">Monitor</span>
                  <Select value={form.plateau_lr_metric as "combined_loss" | "psnr_stretched"}
                    options={METRIC_OPTS}
                    onChange={(v) => set("plateau_lr_metric", v)} />
                </div>
                {num("plateau_lr_factor", "Factor",
                  { min: 0.05, max: 0.99, step: 0.01,
                    hint: "Multiply the LR by this on each plateau cut. 0.5 = halve." })}
                {num("plateau_lr_patience", "Patience (steps)",
                  { min: 100, max: 100000, step: 1,
                    hint: "Steps of no metric progress before a cut fires." })}
                {num("plateau_lr_min_delta", "Min Δ (progress)",
                  { min: 0, max: 10, step: 0.0001,
                    hint: "Minimum metric change that counts as progress. Default 1e-4 suits combined_loss; ~0.05 dB for psnr_stretched." })}
                {num("plateau_lr_cooldown", "Cooldown (steps)",
                  { min: 0, max: 100000, step: 1,
                    hint: "Steps to wait after a cut before the stall counter re-arms." })}
                {num("plateau_lr_min_lr", "Min LR",
                  { min: 0, max: 1, step: 0.00001,
                    hint: "Absolute LR floor. Neither guard reduces the effective LR below this." })}
              </div>
            </CardBody>
          </Card>

          {note && (
            <div className={`job-panel job-panel--${note.ok ? "done" : "err"}`}>
              <LogTail text={(note.ok ? "✓ " : "✗ ") + note.text} />
            </div>
          )}
        </div>
      )}
    </Page>
  );
}
