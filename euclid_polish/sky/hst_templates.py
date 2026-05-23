"""
HST-template renderer for the multi-band scene generator.

Replaces (or supplements) the analytic Sersic B+D path with real
morphology drawn from HST/ACS F814W cutouts in
:class:`HSTTemplateLibrary`. Each render call:

1. Loads one cutout from disk.
2. Background-subtracts (median over the outer annulus).
3. Clips negatives (a galaxy template is non-negative light by
   definition; residual negative pixels are sky noise).
4. Downsamples by a random factor ``f ∈ [MIN, MAX]`` — this both
   pushes the residual HST PSF to sub-pixel scale and reduces HST's
   per-pixel noise by ``~f``.
5. Normalises the patch to unit total flux.
6. Composites into the (H, W, 4) HR canvas at a chosen pixel position,
   per-band scaled by a length-4 flux vector pulled from the COSMOS
   catalog — so the template plays the *same photometric role* as the
   Sersic profile it replaces.

The deposit is broadcast across all four bands: the morphology is
band-independent (only the F814W cutout is available), while colour
information is preserved through the per-band flux scaling. This mirrors
the design of :func:`euclid_polish.sky.profiles.add_sersic_to_bands`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import zoom as ndimage_zoom

from euclid_polish.config import Config
from euclid_polish.hst.catalog import HSTTemplateLibrary
from euclid_polish.hst.types import HSTCutoutMetadata


# ---------------------------------------------------------------------------
# Per-render configuration
# ---------------------------------------------------------------------------

@dataclass
class HSTTemplateRenderConfig:
    """Knobs for one rendering session.

    Bound at simulator construction time; per-render values (rescale
    factor, deposit position) are sampled from these bounds inside
    :meth:`HSTTemplateRenderer.deposit_into_canvas`.
    """

    rescale_min:       float = Config.HST_RESCALE_FACTOR_MIN
    rescale_max:       float = Config.HST_RESCALE_FACTOR_MAX
    bg_annulus_pix:    int   = Config.HST_BACKGROUND_ANNULUS_PIX
    # Spline order for ``scipy.ndimage.zoom``. 3 = cubic spline — close
    # enough to Lanczos for the cleanup purpose at a fraction of the cost.
    # The existing :mod:`euclid_polish.sky.resample` Lanczos-3 helper is
    # integer-factor-only and not applicable here.
    spline_order:      int   = 3
    # If True, masks pixels brighter than the central peak × mask_thresh
    # outside a small "main galaxy" core stamp — prevents neighbouring
    # bright sources from polluting the template. The annulus-based
    # background subtraction usually handles this on its own.
    mask_neighbours:   bool  = False
    mask_thresh:       float = 0.2

    def validate(self) -> Tuple[bool, Optional[str]]:
        if not (0 < self.rescale_min <= self.rescale_max):
            return False, "rescale_min must be in (0, rescale_max]"
        if self.bg_annulus_pix < 0:
            return False, "bg_annulus_pix must be ≥ 0"
        if self.spline_order not in (0, 1, 2, 3, 4, 5):
            return False, "spline_order must be in [0..5]"
        return True, None


# ---------------------------------------------------------------------------
# Preprocessing helpers (module-level so they're testable in isolation)
# ---------------------------------------------------------------------------

def _annulus_median(img: np.ndarray, annulus_pix: int) -> float:
    """Median pixel value over the outer ``annulus_pix``-wide border.

    Used as a robust background estimate for square HST cutouts where the
    target galaxy sits at the centre and the outer frame is sky-dominated.
    Returns 0 if the annulus would be empty (cutout too small).
    """
    H, W = img.shape
    p = int(annulus_pix)
    if p <= 0 or H <= 2 * p or W <= 2 * p:
        return 0.0
    top    = img[:p,    :  ]
    bottom = img[-p:,   :  ]
    left   = img[p:-p,  :p ]
    right  = img[p:-p, -p:]
    flat = np.concatenate([
        top.ravel(), bottom.ravel(), left.ravel(), right.ravel(),
    ])
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


def _rescale_2d(img: np.ndarray, factor: float, *, order: int = 3) -> np.ndarray:
    """Resize a 2-D image by a real-valued downscale ``factor`` > 1.

    Output side ≈ ``input_side / factor`` (integer-rounded). Uses cubic
    spline interpolation by default. Returns ``float32``.
    """
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    z = 1.0 / float(factor)
    out = ndimage_zoom(
        img.astype(np.float32, copy=False),
        zoom=z, order=order, mode="nearest", grid_mode=False,
    )
    return np.ascontiguousarray(out, dtype=np.float32)


def _normalise_to_unit_flux(img: np.ndarray) -> Optional[np.ndarray]:
    """Clip negatives and divide by the total. ``None`` if input is blank."""
    nonneg = np.clip(img, 0.0, None)
    total = float(nonneg.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    return (nonneg / total).astype(np.float32)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class HSTTemplateRenderer:
    """Composite real HST-morphology templates into HR scenes.

    Wraps a :class:`HSTTemplateLibrary` so the simulator only needs to
    know "give me a template at this position with this per-band flux."
    The renderer is stateless across calls; all randomness comes from
    the ``rng`` argument so behaviour is reproducible per seed.
    """

    def __init__(
        self,
        library: HSTTemplateLibrary,
        config: Optional[HSTTemplateRenderConfig] = None,
    ):
        if len(library) == 0:
            raise ValueError(
                "HSTTemplateLibrary is empty. Run "
                "scripts/download_hst_templates.py to populate it."
            )
        self.library = library
        self.config  = config or HSTTemplateRenderConfig()
        ok, why = self.config.validate()
        if not ok:
            raise ValueError(f"Invalid HSTTemplateRenderConfig: {why}")
        # Cache the list of usable cosmos_ids once so sampling is O(1).
        self._cosmos_ids = self.library.cutout_ids()

    # ------------------------------------------------------------------ #
    # Sampling + preprocessing
    # ------------------------------------------------------------------ #

    def sample_metadata(self, rng: np.random.Generator) -> HSTCutoutMetadata:
        """Return one randomly chosen template entry from the library."""
        cid = int(rng.choice(self._cosmos_ids))
        meta = self.library.get(cid)
        assert meta is not None  # cached id is by construction present
        return meta

    def _prepare_patch(
        self, meta: HSTCutoutMetadata, rng: np.random.Generator,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Load + bg-subtract + rescale + normalise.

        Returns ``(patch, rescale_factor)``. ``patch`` is ``None`` if the
        prepared template is blank (all-zero / all-NaN after clip);
        callers should skip the deposit in that case.
        """
        img = self.library.load_image(meta)
        # 1. Background subtract (annulus median is robust to a central
        # bright galaxy and tolerant of moderate noise).
        bg = _annulus_median(img, self.config.bg_annulus_pix)
        img = img - bg
        # 2. Random downsample factor for THIS deposit.
        f = float(rng.uniform(self.config.rescale_min, self.config.rescale_max))
        rescaled = _rescale_2d(img, f, order=self.config.spline_order)
        # 3. Optional neighbour mask (off by default).
        if self.config.mask_neighbours and rescaled.size > 0:
            rescaled = _mask_neighbours(rescaled, self.config.mask_thresh)
        # 4. Unit-flux normalise (drops blank/all-NaN inputs).
        return _normalise_to_unit_flux(rescaled), f

    # ------------------------------------------------------------------ #
    # Public deposit API
    # ------------------------------------------------------------------ #

    def deposit_into_canvas(
        self,
        canvas_multi: np.ndarray,
        flux_per_band: np.ndarray | Tuple[float, ...],
        x_pix: float,
        y_pix: float,
        rng: np.random.Generator,
        *,
        meta: Optional[HSTCutoutMetadata] = None,
    ) -> Optional[dict]:
        """Deposit one template patch centred at ``(x_pix, y_pix)``.

        Parameters
        ----------
        canvas_multi
            ``(H, W, C)`` HR canvas; modified in place.
        flux_per_band
            Length-``C`` vector of total electrons per band for this
            galaxy. The patch is multiplied by this on a per-channel
            basis so each band gets the same morphology with its own
            normalisation.
        x_pix, y_pix
            HR-pixel position of the patch centre.
        rng
            Source of randomness for the rescale factor (and the
            template choice if ``meta`` is None).
        meta
            Optional pre-selected template; if None, sampled from the
            library.

        Returns
        -------
        dict | None
            Metadata about the deposit (template id, rescale factor,
            pixel position), or None if the template was blank/clipped.
        """
        if canvas_multi.ndim != 3:
            raise ValueError(
                f"canvas_multi must be (H, W, C); got {canvas_multi.shape}"
            )
        H, W, C = canvas_multi.shape
        flux = np.asarray(flux_per_band, dtype=np.float32)
        if flux.shape != (C,):
            raise ValueError(
                f"flux_per_band length {flux.shape} != canvas channels {C}"
            )
        if meta is None:
            meta = self.sample_metadata(rng)
        patch, rescale_factor = self._prepare_patch(meta, rng)
        if patch is None:
            return None

        # Composite at (x_pix, y_pix) — patch centred on the pixel.
        h, w = patch.shape
        i0 = int(round(y_pix)) - h // 2
        j0 = int(round(x_pix)) - w // 2
        i1, j1 = i0 + h, j0 + w
        # Clamp to canvas bounds.
        src_i0 = max(0, -i0)
        src_j0 = max(0, -j0)
        dst_i0 = max(0, i0)
        dst_j0 = max(0, j0)
        dst_i1 = min(H, i1)
        dst_j1 = min(W, j1)
        if dst_i1 <= dst_i0 or dst_j1 <= dst_j0:
            return None
        src_i1 = src_i0 + (dst_i1 - dst_i0)
        src_j1 = src_j0 + (dst_j1 - dst_j0)
        sub = patch[src_i0:src_i1, src_j0:src_j1]
        # Broadcast: (h_s, w_s) × (C,) → (h_s, w_s, C) without explicit reshape.
        canvas_multi[dst_i0:dst_i1, dst_j0:dst_j1, :] += (
            sub[:, :, None] * flux[None, None, :]
        )
        return {
            "cosmos_id":      int(meta.cosmos_id),
            "rescale_factor": float(rescale_factor),
            "x_pix":          float(x_pix),
            "y_pix":          float(y_pix),
            "patch_shape":    tuple(int(s) for s in patch.shape),
        }


# ---------------------------------------------------------------------------
# Optional neighbour-mask helper (off by default; here for completeness)
# ---------------------------------------------------------------------------

def _mask_neighbours(img: np.ndarray, thresh: float) -> np.ndarray:
    """Zero pixels in connected-component "islands" disconnected from the
    central peak when each is brighter than ``thresh × peak``.

    Cheap heuristic: floodfill from the brightest pixel into all pixels
    above ``thresh × peak``; pixels outside the floodfill are set to
    zero. The galaxy is assumed connected to its peak.
    """
    from scipy.ndimage import label
    peak = float(img.max())
    if peak <= 0:
        return img
    mask = img > thresh * peak
    labelled, n = label(mask)
    if n == 0:
        return img
    iy, ix = np.unravel_index(int(np.argmax(img)), img.shape)
    keep_lab = int(labelled[iy, ix])
    if keep_lab == 0:
        return img
    out = img.copy()
    # Pixels in any other connected component get zeroed out.
    out[(labelled > 0) & (labelled != keep_lab)] = 0.0
    return out
