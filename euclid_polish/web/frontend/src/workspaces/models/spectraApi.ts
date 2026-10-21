/* Typed endpoints of Models › Spectra: the universal models' power spectra
   and direction (isotropy) diagnostics on the synthetic test fields.
   - GET /ensemble/universal/spectra.json: every universal model's summary
     row (never starts a job);
   - GET /ensemble/universal/spectra/<name>.json: one model's record with
     its 2-D error map (404 when the model has no spectra).
   They are computed by the universal score job (POST
   /ensemble/universal/score with spectra=1: jobs.ts computeSpectra).
   Every payload is read through `normalise*` in ./spectra.ts: any number may
   be null (a non-finite ring, an empty angle bin, a map cell outside the
   measured disc) and a row may carry no spectra at all (its `reason` says
   why), so a partial payload never breaks the view. */
import { useResource } from "../../api/query";

/* ── the contract (what the server sends) ──────────────────────────────── */

/** current: computed for the current checkpoint and test records; stale: an
 *  earlier checkpoint or other test records; missing: none yet. */
export type SpectraStatus = "current" | "stale" | "missing";

/** Per band (VIS, Y, J, H) one value per ring or angle bin; null where the
 *  value is undefined. */
export type BandRows = (number | null)[][];

/** The radial spectra per band at each ring of a 128-px window (period
 *  128 / ring px): the output's power of asinh(x / 1 e⁻) (`power_sr`; white
 *  noise of variance σ² is flat at σ²), its coherence with the truth in
 *  [−1, 1] and its transfer, the amplitude ratio √(P_out / P_truth). */
export type SpectraRadial = { power_sr: BandRows; coherence: BandRows; transfer: BandRows };

/** Per band the error power [along the pixel axes, along the diagonals],
 *  each over its round expectation (1 = no preferred direction). */
export type AxisDiag = ([number | null, number | null] | null)[];

export type SpectraModel = {
  name: string; number: number | null; label: string | null;
  status: SpectraStatus | string;
  computed_at: string | null; n_fields: number | null;
  /** Knee-integrated PSNR from the score index (dB). */
  mean_psnr: number | null;
  steps: number | null;
  /** The checkpoint step the spectra were computed from. */
  served_step: number | null;
  depth: number | null; knees: number[] | null;
  /** The isotropy rule's weight in training (null: trained without it). */
  isotropy_penalty: number | null;
  /** Why the row has no spectra (e.g. the test fields are smaller than the windows). */
  reason: string | null;
  radial: SpectraRadial | null;
  /** Period band key ("2-3", "3-6", …) → per band [axis, diagonal]. */
  axis_diag: Record<string, AxisDiag> | null;
  /** Period band key → per band the error power by wave-vector direction
   *  (one value per `angle_deg` bin) over its round expectation. */
  angular: Record<string, BandRows> | null;
};

/** GET /ensemble/universal/spectra.json. */
export type SpectraPayload = {
  schema: number | null;
  /** Whether the synthetic test records are on this machine (the job needs them). */
  records_local: boolean;
  bands: string[];
  /** [shortest, longest] period of each direction band (px). */
  period_bands: [number, number][];
  /** The angle bins' centres (degrees; 0° and 90° are the pixel axes). */
  angle_deg: number[];
  /** The radial rings' periods (px), coarse to fine. */
  period_px: number[];
  /** The truth's radial power per band (as `power_sr`). */
  truth_power: BandRows | null;
  /** The HR pixel (arcsec). */
  pixel_scale_arcsec: number | null;
  /** The radial measurement: its window (px) and the asinh knee of its stretch (e⁻). */
  radial: { window_px: number; stretch_knee_e: number };
  /** The direction measurement: its window, the longer window of the
   *  coarsest period band (px), and its stretch's knee (e⁻). */
  direction: { window_px: number; long_window_px: number; stretch_knee_e: number };
  models: SpectraModel[];
  /** The models without current spectra. */
  missing: string[];
  notes: string[];
};

/** The 2-D mean error power spectrum (fftshifted): `values[band][row][col]`
 *  at (ky = k[row], kx = k[col]), each over its ring's mean; null outside
 *  the measured disc (the DC bin and |k| beyond the Nyquist circle). */
export type SpectraMap = { k: number[]; values: (number | null)[][][]; period_lines_px: number[]; n_windows: number | null };

/** GET /ensemble/universal/spectra/<name>.json. */
export type SpectraRecord = SpectraModel & { bands: string[]; map: SpectraMap | null };

/* ── URLs ──────────────────────────────────────────────────────────────── */

export const spectraUrl = {
  list: () => "/ensemble/universal/spectra.json",
  model: (name: string) => `/ensemble/universal/spectra/${encodeURIComponent(name)}.json`,
};

/* ── reads (GET only: opening the tab never starts a job) ──────────────── */

const MIN = 60_000;
/** The summary; `poll` (ms) while the score job writes it. */
export const useSpectra = (poll?: number) => useResource<unknown>(spectraUrl.list(), [], { ttl: 30_000, poll });
/** One model's record with its map (null name: idle); a new `stamp` (its
 *  row's computed_at) reads it again. */
export const useSpectraRecord = (name: string | null | undefined, stamp?: string | null) =>
  useResource<unknown>(name ? spectraUrl.model(name) : null, [name, stamp ?? null], { ttl: 5 * MIN });
