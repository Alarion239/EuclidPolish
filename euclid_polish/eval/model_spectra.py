"""Power spectra of a model's output against the truth, and the direction
structure of its error, accumulated field by field (Models › Spectra).

Per field (``sr`` the model output, ``truth`` the HR target blurred to the
target PSF, both ``(H, W, C)`` electrons; a ``BORDER``-pixel frame dropped):

* **Radial spectra** in 128-px round-Hann windows of asinh(x / 1 e⁻): the
  output's and the truth's ring-averaged power per ring of the window's
  frequency grid (period 128 / ring px; white noise of variance σ² gives a
  flat σ²), the coherence ``Re<S T*> / sqrt(<|S|²><|T|²>)`` (1 where the
  output follows the truth at that period) and the transfer
  ``sqrt(<|S|²> / <|T|²>)`` (1 where it has the truth's amplitude, < 1 too
  smooth, > 1 invented structure — as ``eval/power_spectrum.py`` defines
  both). Windows are summed, not medianed: in asinh space a bright star
  no longer dominates.
* **Direction of the error** in asinh(x / 0.01 e⁻), as the isotropy penalty
  measures it (``training/isotropy.py``): every window's error power over its
  own round expectation (ring means interpolated log-log to each bin's |k|),
  averaged. 1 means no preferred direction; a cross on kx = 0 / ky = 0 is the
  pixel-axis grid. 64-px windows give the 2-D map, the angular profiles and
  the axis / diagonal ratios of periods 2-21 px; 128-px windows those of
  21-42 px, which a 64-px window resolves too coarsely.

:func:`field_sums` is pure (a thread pool computes it per field);
:class:`SpectraAccumulator` adds the partial sums and
:meth:`SpectraAccumulator.result` turns them into the JSON-ready record the
console serves.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

SCHEMA = 1
BORDER = 32
RADIAL_WINDOW = 128
DIRECTION_WINDOW = 64
LONG_WINDOW = 128
RADIAL_KNEE_E = 1.0
DIRECTION_KNEE_E = 0.01
#: The power of a white pattern of this many e⁻ RMS near zero flux floors
#: every window's round expectation (as the training penalty's floor).
FLOOR_E = 1e-4
PERIOD_BANDS = ((2, 3), (3, 6), (6, 12), (12, 21))
LONG_PERIOD_BANDS = ((21, 42),)
ANGLE_BINS = 36
#: Periods (px) the direction map marks: the penalty's 3-21 px annulus.
MAP_PERIOD_LINES_PX = (21, 3)


def _round_hann(window: int) -> np.ndarray:
    yy, xx = np.mgrid[0:window, 0:window]
    rr = np.hypot(yy - (window - 1) / 2.0, xx - (window - 1) / 2.0)
    return np.where(rr < window / 2.0, 0.5 * (1.0 + np.cos(np.pi * rr / (window / 2.0))), 0.0)


@lru_cache(maxsize=4)
def _geometry(window: int) -> dict:
    """Frequency grid (cycles per window), rings 1 .. window/2 - 1 (all inside
    the Nyquist circle) and the log-log ring interpolation of one window."""
    f = np.fft.fftfreq(window) * window
    ky, kx = np.meshgrid(f, f, indexing="ij")
    radius = np.hypot(kx, ky).ravel()
    ring = np.rint(radius).astype(int)
    rings = np.arange(1, window // 2)
    member = (ring[None, :] == rings[:, None]).astype(np.float64)
    bins = member.sum(axis=1)
    member /= bins[:, None]
    log_centre = np.log(np.array([radius[ring == r].mean() for r in rings]))
    valid = (ring >= 1) & (radius <= window / 2.0)
    log_k = np.log(np.where(valid, radius, 1.0))
    j = np.clip(np.searchsorted(log_centre, log_k) - 1, 0, rings.size - 2)
    t = (log_k - log_centre[j]) / (log_centre[j + 1] - log_centre[j])
    interp = np.zeros((window * window, rings.size))
    interp[np.arange(window * window), j] = 1.0 - t
    interp[np.arange(window * window), j + 1] = t
    taper = _round_hann(window)
    return {"kx": kx.ravel(), "ky": ky.ravel(), "radius": radius, "ring": ring, "rings": rings,
            "bins": bins, "member": member, "interp": interp, "valid": valid, "taper": taper,
            "taper_sum": float(taper.sum()), "taper_sq_sum": float((taper ** 2).sum())}


def period_px() -> list[float]:
    """Period (px) of each radial-spectrum ring, longest first."""
    return [round(RADIAL_WINDOW / float(r), 3) for r in _geometry(RADIAL_WINDOW)["rings"]]


def angle_centres() -> list[float]:
    """Centre (degrees, 0 = along x, 90 = along y) of each angular bin."""
    edges = np.linspace(0.0, 180.0, ANGLE_BINS + 1)
    return [round(float(v), 2) for v in 0.5 * (edges[:-1] + edges[1:])]


def _n_windows(shape: tuple[int, ...], window: int) -> int:
    return (shape[0] // window) * (shape[1] // window)


def _windows(img: np.ndarray, window: int) -> np.ndarray:
    """``(H, W, C)`` → ``(N, C, window, window)``; trailing pixels dropped."""
    ny, nx = img.shape[0] // window, img.shape[1] // window
    c = img.shape[-1]
    cut = img[:ny * window, :nx * window]
    return cut.reshape(ny, window, nx, window, c).transpose(0, 2, 4, 1, 3).reshape(-1, c, window, window)


def _tapered_fft(win: np.ndarray, geo: dict) -> np.ndarray:
    """``(N, C, w, w)`` → ``(N, C, w²)`` FFT of each window, its taper-weighted
    mean removed and the round Hann taper applied."""
    taper = geo["taper"]
    win = win - (win * taper).sum(axis=(-2, -1), keepdims=True) / geo["taper_sum"]
    return np.fft.fft2(win * taper).reshape(win.shape[0], win.shape[1], -1)


def _normalised_power(residual: np.ndarray, window: int) -> np.ndarray:
    """Sum over the windows of each window's error power over its round
    expectation, ``(C, window²)`` (0 outside the measured disc)."""
    geo = _geometry(window)
    floor = (FLOOR_E / DIRECTION_KNEE_E) ** 2 * geo["taper_sq_sum"]
    power = np.abs(_tapered_fft(_windows(residual, window), geo)) ** 2            # (N, C, w²)
    expected = np.exp(np.log(power @ geo["member"].T + floor) @ geo["interp"].T)
    return np.where(geo["valid"], power / expected, 0.0).sum(axis=0)


def field_sums(sr: np.ndarray, truth: np.ndarray) -> dict:
    """One field's partial sums (pure): ``fields`` 1, and per measure that
    fits the field the window count and float64 sums."""
    s = np.asarray(sr, np.float64)[BORDER:-BORDER, BORDER:-BORDER]
    t = np.asarray(truth, np.float64)[BORDER:-BORDER, BORDER:-BORDER]
    out: dict = {"fields": 1}
    if _n_windows(s.shape, RADIAL_WINDOW):
        geo = _geometry(RADIAL_WINDOW)
        fs = _tapered_fft(_windows(np.arcsinh(s / RADIAL_KNEE_E), RADIAL_WINDOW), geo)
        ft = _tapered_fft(_windows(np.arcsinh(t / RADIAL_KNEE_E), RADIAL_WINDOW), geo)
        member = (geo["member"] * geo["bins"][:, None]).T                           # ring sums
        out["radial_windows"] = fs.shape[0]
        out["radial_ss"] = (np.abs(fs) ** 2).sum(axis=0) @ member
        out["radial_tt"] = (np.abs(ft) ** 2).sum(axis=0) @ member
        out["radial_st"] = np.real(fs * np.conj(ft)).sum(axis=0) @ member
    residual = np.arcsinh(s / DIRECTION_KNEE_E) - np.arcsinh(t / DIRECTION_KNEE_E)
    for key, window in (("direction", DIRECTION_WINDOW), ("long", LONG_WINDOW)):
        n = _n_windows(residual.shape, window)
        if n:
            out[f"{key}_windows"] = n
            out[f"{key}_map"] = _normalised_power(residual, window)
    return out


def _band_mask(geo: dict, window: int, shortest: float, longest: float) -> np.ndarray:
    radius = geo["radius"]
    return (radius >= window / longest) & (radius <= window / shortest) & geo["valid"]


def _axis_diag(mean_map: np.ndarray, window: int, bands) -> dict:
    """``{"a-b": [[axis, diagonal] per band]}`` of a mean normalised map: the
    mean over the bins within 1.5 bins of a pixel axis, and of a diagonal."""
    geo = _geometry(window)
    kx, ky = geo["kx"], geo["ky"]
    axis = (np.abs(kx) <= 1.5) | (np.abs(ky) <= 1.5)
    diag = (np.abs(np.abs(kx) - np.abs(ky)) <= 1.5) & ~axis
    out = {}
    for a, b in bands:
        sel = _band_mask(geo, window, a, b)
        if (sel & axis).any() and (sel & diag).any():
            out[f"{a}-{b}"] = [[round(float(m[sel & axis].mean()), 4), round(float(m[sel & diag].mean()), 4)]
                               for m in mean_map]
    return out


def _angular(mean_map: np.ndarray, window: int) -> dict:
    """``{"a-b": [[mean per angle bin (None when empty)] per band]}``."""
    geo = _geometry(window)
    theta = np.degrees(np.arctan2(geo["ky"], geo["kx"])) % 180.0
    idx = np.clip((theta / (180.0 / ANGLE_BINS)).astype(int), 0, ANGLE_BINS - 1)
    out = {}
    for a, b in PERIOD_BANDS:
        sel = _band_mask(geo, window, a, b)
        counts = np.bincount(idx[sel], minlength=ANGLE_BINS)
        rows = []
        for m in mean_map:
            sums = np.bincount(idx[sel], weights=m[sel], minlength=ANGLE_BINS)
            rows.append([round(float(v / c), 4) if c else None for v, c in zip(sums, counts, strict=True)])
        out[f"{a}-{b}"] = rows
    return out


def _rows(a: np.ndarray, digits: int) -> list:
    return [[round(float(v), digits) if np.isfinite(v) else None for v in row] for row in a]


def _significant(a: np.ndarray, digits: int) -> list:
    """Rows rounded to ``digits`` significant figures (powers span decades)."""
    return [[float(f"{v:.{digits}g}") if np.isfinite(v) else None for v in row] for row in a]


class SpectraAccumulator:
    """Sums of :func:`field_sums` over a model's fields (see the module)."""

    def __init__(self) -> None:
        self.sums: dict = {}

    def add(self, sr: np.ndarray, truth: np.ndarray) -> None:
        self.merge(field_sums(sr, truth))

    def merge(self, part: dict) -> None:
        for key, value in part.items():
            self.sums[key] = value if key not in self.sums else self.sums[key] + value

    def result(self, bands: list[str]) -> dict:
        """The JSON-ready record; ``available`` false (with a ``reason``) when
        the fields are smaller than the windows."""
        s = self.sums
        record: dict = {"schema": SCHEMA, "bands": list(bands), "n_fields": int(s.get("fields", 0)),
                        "available": "radial_ss" in s and "direction_map" in s}
        if not record["available"]:
            record["reason"] = (f"the test fields are smaller than the spectra's {RADIAL_WINDOW}-px "
                                f"windows (after a {BORDER}-px border)")
            return record
        geo = _geometry(RADIAL_WINDOW)
        ss, tt, st = s["radial_ss"], s["radial_tt"], s["radial_st"]
        norm = float(s["radial_windows"]) * geo["bins"][None, :] * geo["taper_sq_sum"]
        with np.errstate(invalid="ignore", divide="ignore"):
            coherence = st / np.sqrt(ss * tt)
            transfer = np.sqrt(ss / tt)
        record["radial"] = {"window_px": RADIAL_WINDOW, "stretch_knee_e": RADIAL_KNEE_E,
                            "n_windows": int(s["radial_windows"]), "period_px": period_px(),
                            "power_sr": _significant(ss / norm, 5), "power_truth": _significant(tt / norm, 5),
                            "coherence": _rows(coherence, 4), "transfer": _rows(transfer, 4)}
        mean64 = s["direction_map"] / float(s["direction_windows"])
        axis_diag = _axis_diag(mean64, DIRECTION_WINDOW, PERIOD_BANDS)
        if "long_map" in s:
            axis_diag.update(_axis_diag(s["long_map"] / float(s["long_windows"]), LONG_WINDOW,
                                        LONG_PERIOD_BANDS))
        record["direction"] = {"window_px": DIRECTION_WINDOW, "long_window_px": LONG_WINDOW,
                               "stretch_knee_e": DIRECTION_KNEE_E, "n_windows": int(s["direction_windows"]),
                               "angle_deg": angle_centres(), "angular": _angular(mean64, DIRECTION_WINDOW),
                               "axis_diag": axis_diag}
        valid = _geometry(DIRECTION_WINDOW)["valid"]
        shape = (len(bands), DIRECTION_WINDOW, DIRECTION_WINDOW)
        shifted = np.fft.fftshift(np.where(valid, mean64, np.nan).reshape(shape), axes=(-2, -1))
        k = np.fft.fftshift(np.fft.fftfreq(DIRECTION_WINDOW) * DIRECTION_WINDOW)
        record["map"] = {"k": [int(v) for v in k],
                         "period_lines_px": list(MAP_PERIOD_LINES_PX),
                         "n_windows": int(s["direction_windows"]),
                         "values": [_rows(band, 4) for band in shifted]}
        return record


__all__ = ["ANGLE_BINS", "LONG_PERIOD_BANDS", "PERIOD_BANDS", "SCHEMA", "SpectraAccumulator",
           "angle_centres", "field_sums", "period_px"]
