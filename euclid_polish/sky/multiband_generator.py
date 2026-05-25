"""
Multi-band clean-HR scene generator.

Replaces the GalSim/donut path with a self-contained renderer that uses:

  * :mod:`euclid_polish.sky.profiles` for Sersic rasterisation (galaxies,
    bulge+disk, lens galaxy light, lensed source light)
  * :mod:`euclid_polish.sky.cosmos2025` as the parametric galaxy/source
    catalog
  * :mod:`euclid_polish.sky.lens_population` for the lens-population priors
    and lensed-source ray-shooting

The output of :meth:`MultiBandSimulator.simulate_field` is a single
:class:`MultiBandSkyImage` with ``data`` of shape ``(H, W, 4)`` in **raw
electrons** on the 0.05″ HR grid, one channel per band ordered as
:attr:`Config.LR_INPUT_BAND_NAMES` (``VIS, Y_E, J_E, H_E``).

Stars are drawn with a single fixed G-type SED — the per-band magnitude
offsets live in :attr:`Config.STAR_BAND_OFFSETS_MAG`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.cosmos2025 import CosmosCatalog
from euclid_polish.sky.lens_population import (
    LensPopulation, render_lens_to_multiband_canvas,
)
from euclid_polish.sky.profiles import add_sersic_to_bands, draw_sersic
from euclid_polish.sky.types import MultiBandSkyImage


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiBandGeneratorConfig:
    """Field-level config for the multi-band simulator.

    The synthetic generator is fully analytic: all galaxies render via
    Sersic B+D from the COSMOS catalog. Real-HST-morphology training
    data lives in a separate stream (``fasrc_generate_hst_tfrecords``)
    and round-trip data lives in another
    (``fasrc_generate_euclid_roundtrip_tfrecords``); both are mixed at
    the dataloader level, not inside the simulator.
    """
    image_size:               int   = Config.DEFAULT_IMAGE_SIZE
    pixel_scale:              float = Config.DEFAULT_PIXEL_SCALE     # arcsec/pix
    gal_density_arcmin2:      float = Config.DEFAULT_GAL_DENSITY_ARCMIN2
    star_density_arcmin2:     float = Config.DEFAULT_STAR_DENSITY_ARCMIN2
    lens_density_arcmin2:     float = Config.LENS_DENSITY_ARCMIN2

    def validate(self) -> Tuple[bool, Optional[str]]:
        if self.image_size <= 0:
            return False, "image_size must be positive"
        if self.pixel_scale <= 0:
            return False, "pixel_scale must be positive"
        if min(self.gal_density_arcmin2, self.star_density_arcmin2,
               self.lens_density_arcmin2) < 0:
            return False, "densities must be non-negative"
        return True, None


# ---------------------------------------------------------------------------
# Stars (point sources with fixed colour)
# ---------------------------------------------------------------------------

def _sample_star_mag(rng: np.random.Generator) -> float:
    """Sample one VIS magnitude from a three-bin (faint/mid/bright) prior.

    Thresholds and ranges come from ``Config.STAR_MAG_*``; the three bins
    are an empirical fit to the observed Wide-Survey stellar density per
    magnitude — most stars are faint, a fat tail extends bright.
    """
    u = rng.random()
    if u < Config.STAR_MAG_PROB_FAINT:
        return Config.STAR_MAG_FAINT_BASE  + Config.STAR_MAG_FAINT_RANGE  * rng.random()
    if u < Config.STAR_MAG_PROB_MID:
        return Config.STAR_MAG_MID_BASE    + Config.STAR_MAG_MID_RANGE    * rng.random()
    return Config.STAR_MAG_BRIGHT_BASE     + Config.STAR_MAG_BRIGHT_RANGE * rng.random()


def _deposit_star(
    canvas_4ch: np.ndarray, x_pix: float, y_pix: float, mag_vis: float,
) -> None:
    """Drop a point source at the nearest HR pixel, replicating across bands.

    The per-band magnitude is ``mag_vis + STAR_BAND_OFFSETS_MAG[band]`` and
    each band's flux is computed in its own per-stack zeropoint.
    """
    H, W, C = canvas_4ch.shape
    ix = int(round(x_pix))
    iy = int(round(y_pix))
    if not (0 <= ix < W and 0 <= iy < H):
        return
    for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        band = Config.get_band(band_name)
        mag_k = mag_vis + Config.STAR_BAND_OFFSETS_MAG[band_name]
        flux_k = 10.0 ** (-0.4 * (mag_k - band.sim_zeropoint_e))
        canvas_4ch[iy, ix, k] += np.float32(flux_k)


# ---------------------------------------------------------------------------
# Multi-band simulator
# ---------------------------------------------------------------------------

class MultiBandSimulator:
    """Generates ``(H, W, 4)`` HR clean fields in electrons.

    Each channel is the clean image of the same scene seen in band
    ``Config.LR_INPUT_BAND_NAMES[k]``. Geometry of every source is
    band-independent; per-band flux normalisations come from the catalog
    (galaxies/lenses) or from the fixed stellar colour (stars).
    """

    def __init__(
        self,
        catalog: CosmosCatalog,
        config: Optional[MultiBandGeneratorConfig] = None,
        *,
        lens_population: Optional[LensPopulation] = None,
    ):
        self.catalog = catalog
        self.config  = config or MultiBandGeneratorConfig()
        ok, why = self.config.validate()
        if not ok:
            raise ValueError(f"Invalid generator config: {why}")
        self.lens_population = lens_population or LensPopulation(catalog)

    # ------------------------------------------------------------------ #
    def _field_area_arcmin2(self) -> float:
        side_arcmin = self.config.image_size * self.config.pixel_scale / 60.0
        return side_arcmin ** 2

    def _random_pix(self, rng: np.random.Generator) -> Tuple[float, float]:
        N = self.config.image_size
        return float(rng.uniform(0.0, N - 1)), float(rng.uniform(0.0, N - 1))

    # ------------------------------------------------------------------ #
    def _add_galaxy(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
    ) -> dict:
        """Render one galaxy via Sersic B+D from the COSMOS catalog row.

        Geometry band-independent; per-band fluxes from the catalog
        drive the photometry. Each Sersic component is rendered once and
        broadcast-scaled into every channel by its per-band flux — cuts
        Sersic evaluations by ``NUM_LR_CHANNELS=4×`` without changing
        the result.
        """
        g = self.catalog.sample_galaxy(rng)
        x_pix, y_pix = self._random_pix(rng)
        add_sersic_to_bands(
            canvas_4ch, flux_per_band=g.bulge_flux_e, n=4.0,
            r_e=g.bulge_r_e_arcsec, q=g.bulge_axis_ratio,
            theta_rad=g.angle_rad, x0=x_pix, y0=y_pix,
            pixel_scale=self.config.pixel_scale,
        )
        add_sersic_to_bands(
            canvas_4ch, flux_per_band=g.disk_flux_e, n=1.0,
            r_e=g.disk_r_e_arcsec, q=g.disk_axis_ratio,
            theta_rad=g.angle_rad, x0=x_pix, y0=y_pix,
            pixel_scale=self.config.pixel_scale,
        )
        return {
            "type": "galaxy",
            "render": "sersic",
            "catalog_id": g.catalog_id,
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            "z_phot": float(g.z_phot),
            "bulge_re_arcsec": float(g.bulge_r_e_arcsec),
            "disk_re_arcsec":  float(g.disk_r_e_arcsec),
            "flux_e_per_band": list(map(float, [g.total_flux_e(k) for k in range(4)])),
        }

    def _add_star(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
    ) -> dict:
        x_pix, y_pix = self._random_pix(rng)
        mag = _sample_star_mag(rng)
        _deposit_star(canvas_4ch, x_pix, y_pix, mag)
        return {
            "type": "star",
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            "mag_vis": float(mag),
        }

    def _add_lens(
        self, canvas_4ch: np.ndarray, rng: np.random.Generator,
    ) -> Optional[dict]:
        try:
            lp = self.lens_population.sample(rng)
        except RuntimeError:
            return None
        x_pix, y_pix = self._random_pix(rng)
        from dataclasses import replace
        lp = replace(lp, centre_x_pix=x_pix, centre_y_pix=y_pix)
        # Fast path: render the lens once into the 4-channel canvas (geometry
        # is band-independent; only per-band fluxes differ).
        render_lens_to_multiband_canvas(
            canvas_4ch, params=lp, pixel_scale=self.config.pixel_scale,
        )
        return {
            "type": "lens",
            "x_pix": float(x_pix),
            "y_pix": float(y_pix),
            "z_lens": float(lp.z_lens),
            "z_source": float(lp.z_source),
            "theta_E_arcsec": float(lp.theta_E_arcsec),
            "sigma_v_proxy_q": float(lp.lens_q),
        }

    # ------------------------------------------------------------------ #
    def simulate_field(
        self,
        rng: np.random.Generator,
        *,
        n_galaxies: Optional[int] = None,
        n_stars:    Optional[int] = None,
        n_lenses:   Optional[int] = None,
    ) -> Tuple[MultiBandSkyImage, dict]:
        """Render one clean HR field in 4 bands.

        Returns
        -------
        sky_image : :class:`MultiBandSkyImage` with ``data`` shape
                    ``(image_size, image_size, 4)`` in raw electrons.
        metadata  : dict with per-source parameter records.
        """
        cfg = self.config
        N   = cfg.image_size

        area = self._field_area_arcmin2()
        if n_galaxies is None:
            n_galaxies = int(rng.poisson(cfg.gal_density_arcmin2  * area))
        if n_stars is None:
            n_stars    = int(rng.poisson(cfg.star_density_arcmin2 * area))
        if n_lenses is None:
            n_lenses   = int(rng.poisson(cfg.lens_density_arcmin2 * area))

        canvas = np.zeros((N, N, Config.NUM_LR_CHANNELS), dtype=np.float32)

        galaxies, stars, lenses = [], [], []
        for _ in range(n_galaxies):
            galaxies.append(self._add_galaxy(canvas, rng))
        for _ in range(n_stars):
            stars.append(self._add_star(canvas, rng))
        for _ in range(n_lenses):
            rec = self._add_lens(canvas, rng)
            if rec is not None:
                lenses.append(rec)

        meta = {
            "field_area_arcmin2":     float(area),
            "galaxy_density_arcmin2": float(cfg.gal_density_arcmin2),
            "star_density_arcmin2":   float(cfg.star_density_arcmin2),
            "lens_density_arcmin2":   float(cfg.lens_density_arcmin2),
            "n_galaxies": len(galaxies),
            "n_stars":    len(stars),
            "n_lenses":   len(lenses),
            "galaxies":   galaxies,
            "stars":      stars,
            "lenses":     lenses,
        }
        sky = MultiBandSkyImage(
            data=canvas,
            pixel_scale_arcsec=cfg.pixel_scale,
            band_names=Config.LR_INPUT_BAND_NAMES,
            is_clean=True,
            metadata=meta,
        )
        return sky, meta
