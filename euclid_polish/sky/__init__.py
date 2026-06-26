"""
Multi-band sky simulation + forward model.

Public API:

* :class:`MultiBandSimulator` — generate 4-channel clean HR fields
  (galaxies + stars + strong lenses) using the COSMOS2025 catalog and
  the project's custom Sersic renderer.
* :class:`MultiBandForward` — per-band PSF convolution + noise +
  NISP→VIS-LR resample, returning paired (LR 4-channel, HR-VIS 1-channel).
* :class:`Image` — typed container for ``(H, W, C)`` records.
* :class:`Cosmos2025Catalog` — galaxy catalog reader (mandatory).
* :class:`LensPopulation` — Collett 2015 lens population sampler.
"""

from euclid_polish.sky.multiband_generator import (
    MultiBandSimulator,
    MultiBandGeneratorConfig,
)
from euclid_polish.sky.multiband_forward import (
    MultiBandForward,
    MultiBandForwardConfig,
)
from euclid_polish.sky.cosmos2025 import (
    CosmosCatalog,
    Cosmos2025Catalog,
    open_cosmos2025,
    GalaxyParams,
)
from euclid_polish.sky.lens_population import (
    LensParams,
    LensPopulation,
    einstein_radius_sis,
    render_lens_to_canvas,
    render_lens_to_multiband_canvas,
)
from euclid_polish.image import Image
from euclid_polish.sky.profiles import (
    add_sersic_to_bands,
    compute_sersic_stamp,
    draw_bulge_disk,
    draw_sersic,
    evaluate_sersic_at_coords,
    sersic_b_n,
    sersic_amp_from_flux,
)
from euclid_polish.sky.resample import upsample, lanczos3_upsample, cubic_upsample

__all__ = [
    "MultiBandSimulator",
    "MultiBandGeneratorConfig",
    "MultiBandForward",
    "MultiBandForwardConfig",
    "Image",
    "CosmosCatalog",
    "Cosmos2025Catalog",
    "open_cosmos2025",
    "GalaxyParams",
    "LensParams",
    "LensPopulation",
    "einstein_radius_sis",
    "render_lens_to_canvas",
    "render_lens_to_multiband_canvas",
    "add_sersic_to_bands",
    "compute_sersic_stamp",
    "draw_bulge_disk",
    "draw_sersic",
    "evaluate_sersic_at_coords",
    "sersic_b_n",
    "sersic_amp_from_flux",
    "upsample",
    "lanczos3_upsample",
    "cubic_upsample",
]
